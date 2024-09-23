from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, auth
from flask_cors import CORS
from flask_mail import Mail, Message
from config import Config
from email_templates import NEW_ACCOUNT_TEMPLATE
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnablePassthrough
from langchain_core.agents import AgentFinish
from langgraph.graph import END, Graph
import os
import pytz
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)

# Initialize CORS for both local development and production URLs
CORS(app, origins=[
    "http://localhost:4200", 
    "http://127.0.0.1:5000", 
    "https://godrej-chat.web.app", 
    "https://godrej-chat.firebaseapp.com", 
    "https://godreja.onrender.com",
    "*"
    ])

# Initialize Firebase Admin SDK
cred = credentials.Certificate('/etc/secrets/credentials.json')
firebase_admin.initialize_app(cred)

# Initialize Flask-Mail
mail = Mail(app)
mail = Mail(app)

tools = [TavilySearchResults(max_results=5)]
prompt = hub.pull("hwchase17/openai-functions-agent")
llm = ChatOpenAI(model="gpt-3.5-turbo")
agent_runnable = create_openai_functions_agent(llm, tools, prompt)
agent = RunnablePassthrough.assign(agent_outcome=agent_runnable)


os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')

if not os.getenv('OPENAI_API_KEY'):
    raise ValueError("OPENAI_API_KEY is not set in the environment variables")
if not os.getenv('TAVILY_API_KEY'):
    raise ValueError("TAVILY_API_KEY is not set in the environment variables")


def execute_tools(data):
    agent_action = data.pop('agent_outcome')
    tools_to_use = {t.name: t for t in tools}[agent_action.tool]
    observation = tools_to_use.invoke(agent_action.tool_input)
    data['intermediate_steps'].append((agent_action, observation))
    return data


def should_continue(data):
    if isinstance(data['agent_outcome'], AgentFinish):
        return "exit"
    else:
        return "continue"

workflow = Graph()
workflow.add_node("agent", agent)
workflow.add_node("tools", execute_tools)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "exit": END
    }
)
workflow.add_edge('tools', 'agent')
chain = workflow.compile()

@app.route('/', methods=['GET'])
def index():
    return "Hello, API is running!"

@app.route('/generate', methods=['POST'])
def generate():
    try:
        # Extract Authorization header
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            logger.warning("Missing or invalid Authorization header")
            return jsonify({'error': 'Missing or invalid Authorization header'}), 401

        # Extract ID token
        id_token = auth_header.split(' ')[1]

        try:
            # Verify Firebase token
            decoded_token = auth.verify_id_token(id_token)
            user_id = decoded_token['uid']
            logger.debug(f"Authenticated user ID: {user_id}")
        except Exception as e:
            logger.error(f"Error verifying ID token: {e}")
            return jsonify({'error': 'Invalid or expired token'}), 401

        # Extract request data
        data = request.get_json()

        if not data:
            logger.warning("No JSON payload provided")
            return jsonify({'error': 'No input data provided'}), 400

        prompt = data.get('prompt')
        if not prompt:
            logger.warning("Prompt not provided in request")
            return jsonify({'error': 'Prompt is required'}), 400

        logger.debug(f"Received prompt: {prompt}")

        user_data = data.get('user_data')

        if not user_data:
            logger.warning(f"User data not found for user_id: {user_id}")
            return jsonify({'error': 'User data not found'}), 404

        logger.debug(f"User data: {user_data}")

        if is_relevant_query(prompt, user_data):
            response = chain.invoke({"input": prompt, "intermediate_steps": []})
            logger.debug(f"Chain response: {response}")

            search_results = []
            if isinstance(response, dict):
                intermediate_steps = response.get('intermediate_steps', [])
                if intermediate_steps and isinstance(intermediate_steps[0], tuple):
                    search_results = intermediate_steps[0][1]

            if not search_results or isinstance(search_results, str):
                logger.debug("No valid search results from chain, using TavilySearchResults")
                try:
                    search_tool = TavilySearchResults(max_results=5)
                    search_results = search_tool.invoke(prompt)
                except Exception as e:
                    logger.error(f"Error using TavilySearchResults: {e}")
                    search_results = []

            logger.debug(f"Search results: {search_results}")

            formatted_results = format_search_results(search_results)
            overall_summary = generate_overall_summary(search_results)

            if isinstance(response, dict):
                agent_outcome = response.get('agent_outcome', '')
                if isinstance(agent_outcome, dict):
                    ai_response = agent_outcome.get('output', '')
                elif isinstance(agent_outcome, str):
                    ai_response = agent_outcome
                else:
                    ai_response = str(agent_outcome)
            else:
                ai_response = str(response)

            logger.debug(f"AI response before formatting: {ai_response}")

            ai_response = f"{ai_response}\n\n{formatted_results}\nOverall Summary:\n{overall_summary}"
        else:
            ai_response = "This query doesn't seem to be related to your department or interests. Would you like to rephrase your question or ask something more relevant?"

        logger.debug(f"Final AI response: {ai_response}")

        return jsonify({'response': ai_response})

    except Exception as e:
        logger.exception("Error in generate function")
        return jsonify({'error': 'An internal error occurred'}), 500

def is_relevant_query(query, user_data):
    prompt = f"""
    Given the user's department: {user_data['department']}
    and interests: {', '.join(user_data['interests'])},
    is the following query relevant? Query: {query}
    Respond with 'Yes' or 'No'.
    """
    response = llm.invoke(prompt)  
    return response.content.strip().lower() == 'yes' 


def format_search_results(results):
    if isinstance(results, str):
        logger.error(f"Received error instead of search results: {results}")
        return "No search results available due to an error."
    
    if not results or not isinstance(results, list):
        return "No search results found."
    
    formatted_results = "Top 5 Sources:\n\n"
    for i, result in enumerate(results[:5], 1):
        if isinstance(result, dict):
            title = result.get('title', f'Reference {i}')
            url = result.get('url', 'No URL available')
            content = result.get('content', 'No content available')
            
            formatted_results += f"{i}. [{title}]({url})\n"
            summary = generate_three_line_summary(content)
            formatted_results += f"   {summary}\n\n"
        else:
            formatted_results += f"{i}. Unable to format this result.\n\n"
    
    return formatted_results

def generate_overall_summary(results):
    if isinstance(results, str):
        logger.error(f"Received error instead of search results: {results}")
        return "Unable to generate summary due to an error in search results."
    
    if not results or not isinstance(results, list):
        return "No information available to summarize."
    
    try:
        combined_content = " ".join([result.get('content', '') for result in results[:5] if isinstance(result, dict)])
        summary_prompt = f"Provide a concise overall summary of the following information:\n\n{combined_content}\n\nSummary:"
        summary = llm.invoke(summary_prompt)
        return summary.content
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        return "Unable to generate summary due to an unexpected error."


def generate_three_line_summary(content):
    summary_prompt = f"Provide a three-line summary of the following content:\n\n{content}\n\nSummary:"
    summary = llm.invoke(summary_prompt)
    return summary.content.strip()


@app.route('/recent-news',  methods=['POST'])
def recent_news():
    try:
        # Extract request data
        data = request.get_json()
        user_id = data.get('user_data', {}).get('uid')
        if not user_id:
            logger.warning("User ID not provided in request")
            return jsonify({'error': 'User ID is required'}), 400

        user_data = data.get('user_data')
        
        if not user_data:
            logger.warning(f"User not found for user_id: {user_id}")
            return jsonify({'error': 'User not found'}), 404
        
        news_articles = get_recent_news(user_data)
        
        if not news_articles:
            logger.info(f"No recent news found for user_id: {user_id}")
            return jsonify({'message': 'No recent news found', 'news': []}), 200
        
        logger.info(f"Successfully retrieved {len(news_articles)} news articles for user_id: {user_id}")
        return jsonify({'news': news_articles})
    except Exception as e:
        logger.exception(f"Error in recent_news: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

def get_recent_news(user_data, num_articles=10):
    # print(f"User ID: {user_data}")
    # print(f"User Type:",type(user_data))
    # {'department': 'IT', 'displayName': 'Avinash HBTU', 'email': '210231017@hbtu.ac.in', 'emailVerified': True, 'interests': ['AI', 'DATA'], 'joinAt': '2024-09-23 01:02:56', 'skills': ['Python', 'Java', 'AI'], 'uid': 'OtWX1ge1qOdltmRh8AwOzwp2uq42'}
    interests = ", ".join(user_data.get('interests', []))
    skills = ", ".join(user_data.get('skills', []))
    print(f"Interests: {interests}, Skills: {skills}")
    current_date = datetime.now(pytz.utc).strftime("%Y-%m-%d")
    
    search_query = f"latest news as of {current_date} related to {interests} and {skills}"
    
    try: 
        search_results = TavilySearchResults(
            max_results=20,
            include_domains=["bbc.com", "cnn.com", "reuters.com", "apnews.com", "bloomberg.com", "nytimes.com", "wsj.com"],
            exclude_domains=["wikipedia.org"],
            time_range="d"
        ).invoke(search_query)
    except Exception as e:
        logger.error(f"Error in Tavily search: {str(e)}")
        return []
    
    prompt = f"""
    Based on these search results, identify the {num_articles} most recent and relevant news articles related to the user's interests ({interests}) and skills ({skills}).
    Today's date is {current_date}. Only include articles from the past week, prioritizing the most recent ones.
    For each article, provide:
    1. A concise title (max 15 words)
    2. A brief summary (2-3 sentences)
    3. The source URL
    4. The exact publication date and time (if available, in UTC)
    5. The source name

    Format the output as a list of dictionaries, each containing 'title', 'summary', 'url', 'date', and 'source' keys.
    Ensure the 'date' field is in the format 'YYYY-MM-DD HH:MM:SS UTC' if available, or 'YYYY-MM-DD' if only the date is known.
    If the exact date is not available, use 'Recent' as the date value.
    
    Sort the articles by date, with the most recent first.

    Search results:
    {search_results}
    """
    
    try:
        news_articles = eval(llm.invoke(prompt).content)
    except Exception as e:
        logger.error(f"Error in LLM processing: {str(e)}")
        return []
    
    current_time = datetime.now(pytz.utc)
    filtered_articles = []
    
    for article in news_articles:
        article_date = parse_date(article['date'])
        if article_date and (current_time - article_date) <= timedelta(days=7):
            filtered_articles.append(article)
        elif article['date'] == 'Recent':
            filtered_articles.append(article)
    
    # Sort articles by date, most recent first
    filtered_articles.sort(key=lambda x: parse_date(x['date']) or datetime.max.replace(tzinfo=pytz.UTC), reverse=True)
    
    return filtered_articles[:num_articles]


def parse_date(date_str):
    """
    Parse a date string into a datetime object.
    """
    try:
        if date_str == 'Recent':
            return datetime.now(pytz.utc)
        elif ' ' in date_str:  # Assumes format is 'YYYY-MM-DD HH:MM:SS UTC'
            return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S %Z").replace(tzinfo=pytz.UTC)
        else:  # Assumes format is 'YYYY-MM-DD'
            return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=pytz.UTC)
    except ValueError:
        return None



@app.route('/send-mail', methods=['POST'])
def send_mail():
    # Extract Authorization header
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({'error': 'Missing or invalid auth token'}), 401

    # Extract ID token
    id_token = auth_header.split(' ')[1]

    try:
        # Verify Firebase token
        decoded_token = auth.verify_id_token(id_token)
        user_id = decoded_token['uid']
        print(f"Authenticated user ID: {user_id}")

        # Extract request data
        data = request.get_json()

        # Validate email and name fields
        name = data.get('name')
        email = data.get('email')

        if not name or not email:
            return jsonify({'error': 'Missing name or email field'}), 400

        # Prepare email content
        subject = "Welcome to Godrej AI"
        body = NEW_ACCOUNT_TEMPLATE.format(name=name, email=email)  # Using the correct template

        # Send email
        send_email(email, subject, body)

        return jsonify({'message': 'Email sent successfully'}), 200

    except Exception as e:
        print(f"Error verifying ID token or sending email: {e}")
        return jsonify({'error': 'Invalid token or internal error'}), 401


def send_email(to, subject, body):
    msg = Message(
        subject=subject,
        recipients=[to],
        html=body,
        sender=app.config['MAIL_USERNAME']
    )
    try:
        mail.send(msg)
        print(f'Email sent to {to}')
    except Exception as e:
        print(f'Error sending email: {e}')


if __name__ == '__main__':
    app.run(debug=True)

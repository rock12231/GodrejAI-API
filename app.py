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
    "https://godreja.onrender.com"
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
        body = NEW_EVENT_TEMPLATE.format(name=name, email=email)  # Using the correct template

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

from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, auth
from flask_cors import CORS
from flask_mail import Mail, Message
from config import Config
from email_templates import NEW_ACCOUNT_TEMPLATE

app = Flask(__name__)
app.config.from_object(Config)

CORS(app, origins=["http://localhost:4200", "https://godreja.onrender.com"])

# Initialize Firebase Admin SDK //     "/etc/secrets/<filename>"
cred = credentials.Certificate('credentials.json')
firebase_admin.initialize_app(cred)

# Initialize Flask-Mail
mail = Mail(app)

@app.route('/', methods=['GET'])
def index():
    return "Hello, API is running!"

@app.route('/send-mail', methods=['POST'])
def new_event():
    data = request.get_json()
    to = data.get('email')
    subject = "Welcome to Godrej AI"
    body = NEW_ACCOUNT_TEMPLATE.format(
        name=data.get('name'),
        email=data.get('email')
    )
    send_email(to, subject, body)
    return jsonify({'message': 'New event email sent'}), 200

def send_email(to, subject, body):
    msg = Message(subject=subject,
                  recipients=[to],
                  html=body,
                  sender=app.config['MAIL_USERNAME'])
    try:
        mail.send(msg)
        print(f'Email sent to {to}')
    except Exception as e:
        print(f'Error sending email: {e}')

if __name__ == '__main__':
    app.run(debug=True)

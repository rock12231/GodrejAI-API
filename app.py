from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, auth
from flask_cors import CORS
from flask_mail import Mail, Message
from config import Config
from email_templates import NEW_EVENT_TEMPLATE  # Corrected template import

app = Flask(__name__)
app.config.from_object(Config)

# Initialize CORS for both local development and production URLs
CORS(app, origins=["http://localhost:4200", "https://godreja.onrender.com", "https://godrej-chat.firebaseapp.com", "https://godreja.onrender.com"])

# Initialize Firebase Admin SDK
cred = credentials.Certificate('credentials.json')  # Ensure correct file path
firebase_admin.initialize_app(cred)

# Initialize Flask-Mail
mail = Mail(app)

@app.route('/', methods=['GET'])
def index():
    return "Hello, API is running!"

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

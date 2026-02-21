import hashlib
import os
import random
import string
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import streamlit as st
import time
from dotenv import load_dotenv

load_dotenv()

def hash_password(password: str, salt: str = None) -> tuple:
    """Hash a password with a salt"""
    if not salt:
        salt = os.urandom(16).hex()
    
    # Simple hash for demonstration - in production use bcrypt/argon2
    password_hash = hashlib.sha256((password + salt).encode('utf-8')).hexdigest()
    return password_hash, salt

def verify_password(stored_password: str, stored_salt: str, provided_password: str) -> bool:
    """Verify a password against stored hash and salt"""
    password_hash, _ = hash_password(provided_password, stored_salt)
    return password_hash == stored_password

def generate_otp(length: int = 6) -> str:
    """Generate a numeric OTP"""
    return ''.join(random.choices(string.digits, k=length))

def send_otp_email(email: str, otp: str):
    """
    Send OTP via Email.
    Uses SMTP_EMAIL and SMTP_PASSWORD from env if available.
    Otherwise falls back to Mock (print/UI).
    """
    smtp_email = os.getenv("SMTP_EMAIL")
    smtp_password = os.getenv("SMTP_PASSWORD")
    
    # 1. Try Real Email if Configured
    # Check if they are not default placeholders
    if smtp_email and smtp_password and "your_email" not in smtp_email:
        try:
            msg = MIMEMultipart()
            msg['From'] = smtp_email
            msg['To'] = email
            msg['Subject'] = "Your Verification Code - UniMate"
            
            body = f"""
            <html>
                <body>
                    <h2>Verification Code</h2>
                    <p>Your OTP code is: <strong>{otp}</strong></p>
                    <p>This code expires in 5 minutes.</p>
                </body>
            </html>
            """
            msg.attach(MIMEText(body, 'html'))
            
            # SMTP Setup (Gmail defaults)
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(smtp_email, smtp_password)
            text = msg.as_string()
            server.sendmail(smtp_email, email, text)
            server.quit()
            
            print(f"✅ Real email sent to {email}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to send real email: {e}")
            # Fallback to mock
    
    # 2. Mock Email (Fallback)
    # Simulate network delay
    time.sleep(1)
    
    # Log to Streamlit for user visibility during dev
    print(f"============================================")
    print(f"📧 EMAIL TO: {email}")
    print(f"🔢 OTP CODE: {otp}")
    print(f"============================================")
    
    # Store in session state for UI display (Mock purpose only)
    if 'mock_email_box' not in st.session_state:
        st.session_state.mock_email_box = []
    
    st.session_state.mock_email_box.insert(0, {
        "to": email,
        "otp": otp,
        "time": time.strftime("%H:%M:%S")
    })
    
    return True

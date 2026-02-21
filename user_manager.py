import sqlite3
import time
import uuid
from pathlib import Path
from typing import Optional, Dict, Tuple
from auth_utils import hash_password, verify_password, generate_otp, send_otp_email

class UserManager:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.db_path = self.data_dir / "users.db"
        self._init_db()
        
    def _init_db(self):
        """Initialize user database"""
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)
            
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Users table
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                is_verified INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            )
        ''')
        
        # Pending verifications table
        c.execute('''
            CREATE TABLE IF NOT EXISTS pending_verifications (
                email TEXT PRIMARY KEY,
                otp TEXT NOT NULL,
                expires_at REAL NOT NULL
            )
        ''')
        
        # Sessions table for persistence
        c.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                expires_at REAL NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def register_init(self, email: str, password: str) -> Tuple[bool, str]:
        """Initialize registration by sending OTP"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # Check if already registered
            c.execute("SELECT id FROM users WHERE email = ?", (email,))
            if c.fetchone():
                return False, "Email already registered"
            
            # Generate OTP
            otp = generate_otp()
            expires_at = time.time() + 300  # 5 minutes
            
            # Store/Update pending verification
            c.execute("""
                INSERT OR REPLACE INTO pending_verifications (email, otp, expires_at)
                VALUES (?, ?, ?)
            """, (email, otp, expires_at))
            
            # Temporary store password hash (in a real app, maybe store this differently or just pass it through)
            # For simplicity, we'll re-ask for password or store it in session state on frontend
            # Here we just handle the OTP part
            
            conn.commit()
            conn.close()
            
            # Send Email
            send_otp_email(email, otp)
            
            return True, "OTP sent to email"
        except Exception as e:
            return False, str(e)

    def register_confirm(self, email: str, password: str, otp: str) -> Tuple[bool, str]:
        """Confirm registration with OTP"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # Verify OTP
            c.execute("SELECT otp, expires_at FROM pending_verifications WHERE email = ?", (email,))
            result = c.fetchone()
            
            if not result:
                return False, "No pending verification found"
                
            stored_otp, expires_at = result
            
            if time.time() > expires_at:
                return False, "OTP expired"
                
            if stored_otp != otp:
                return False, "Invalid OTP"
            
            # Create User
            user_id = str(uuid.uuid4())
            salt = "" # hash_password generates salt
            p_hash, salt = hash_password(password, None)
            created_at = time.strftime("%Y-%m-%d %H:%M:%S")
            
            c.execute("""
                INSERT INTO users (id, email, password_hash, salt, is_verified, created_at)
                VALUES (?, ?, ?, ?, 1, ?)
            """, (user_id, email, p_hash, salt, created_at))
            
            # Cleanup pending
            c.execute("DELETE FROM pending_verifications WHERE email = ?", (email,))
            
            conn.commit()
            conn.close()
            
            return True, "Registration successful"
        except Exception as e:
            return False, str(e)
            
    def login(self, email: str, password: str) -> Tuple[Optional[Dict], str]:
        """Login user"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            c.execute("SELECT id, password_hash, salt, is_verified FROM users WHERE email = ?", (email,))
            result = c.fetchone()
            
            conn.close()
            
            if not result:
                return None, "User not found"
                
            user_id, stored_hash, salt, is_verified = result
            
            if not verify_password(stored_hash, salt, password):
                return None, "Invalid password"
                
            if not is_verified:
                return None, "Email not verified"
                
            return {"id": user_id, "email": email}, "Success"
            
        except Exception as e:
            return None, str(e)
            
    def create_session(self, user_id: str, duration_days: int = 30) -> str:
        """Create a persistent session token"""
        token = str(uuid.uuid4())
        expires_at = time.time() + (duration_days * 24 * 3600)
        
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("INSERT INTO sessions (token, user_id, expires_at) VALUES (?, ?, ?)", 
                  (token, user_id, expires_at))
        conn.commit()
        conn.close()
        
        return token
        
    def validate_session(self, token: str) -> Optional[Dict]:
        """Validate a session token and return user info"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            c.execute("""
                SELECT u.id, u.email, s.expires_at 
                FROM sessions s 
                JOIN users u ON s.user_id = u.id 
                WHERE s.token = ?
            """, (token,))
            
            result = c.fetchone()
            conn.close()
            
            if not result:
                return None
                
            user_id, email, expires_at = result
            
            if time.time() > expires_at:
                # Clean up expired
                self.delete_session(token)
                return None
                
            return {"id": user_id, "email": email}
            
        except Exception:
            return None
            
    def delete_session(self, token: str):
        """Logout/Delete session"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("DELETE FROM sessions WHERE token = ?", (token,))
            conn.commit()
            conn.close()
        except:
            pass

import sqlite3
import hashlib
import os
import jwt
from datetime import datetime, timedelta
from functools import wraps
from flask import request, redirect, url_for, session, flash

# Secret key for JWT token generation
SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key-here')

def init_db():
    """Initialize the database with users table"""
    try:
        # Remove existing database file if it exists and is corrupted
        if os.path.exists('users.db'):
            try:
                # Try to connect to test if it's a valid database
                conn = sqlite3.connect('users.db')
                conn.close()
            except sqlite3.DatabaseError:
                # If connection fails, remove the corrupted file
                os.remove('users.db')
        
        # Create new database connection
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        
        # Create users table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        conn.commit()
        conn.close()
        print("Database initialized successfully")
    except Exception as e:
        print(f"Error initializing database: {str(e)}")
        raise

def hash_password(password):
    """Hash the password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password, email):
    """Register a new user"""
    hashed_password = hash_password(password)
    
    try:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO users (username, password, email) VALUES (?, ?, ?)",
            (username, hashed_password, email)
        )
        
        conn.commit()
        conn.close()
        return True, "User registered successfully"
    except sqlite3.IntegrityError:
        return False, "Username or email already exists"
    except Exception as e:
        return False, f"Error: {str(e)}"

def verify_user(username, password):
    """Verify user credentials"""
    hashed_password = hash_password(password)
    
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT id, username FROM users WHERE username = ? AND password = ?",
        (username, hashed_password)
    )
    
    user = cursor.fetchone()
    conn.close()
    
    if user:
        return True, {"id": user[0], "username": user[1]}
    else:
        return False, "Invalid username or password"

def generate_token(user_id, username):
    """Generate JWT token for authenticated user"""
    payload = {
        'user_id': user_id,
        'username': username,
        'exp': datetime.utcnow() + timedelta(days=1)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm='HS256')

def login_required(f):
    """Decorator to check if user is logged in"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Initialize the database when the module is imported
init_db() 
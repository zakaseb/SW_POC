import sqlite3
import bcrypt
import os
import pickle
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'users.db')

def init_db():
    """
    Initializes the database and creates tables if they don't exist.
    """
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL
            )
        """)
        # Jobs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                job_type TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                completed_at TIMESTAMP,
                result_path TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        # User sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_sessions (
                user_id INTEGER PRIMARY KEY,
                session_data BLOB NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        conn.commit()

def add_user(username, password):
    """Adds a new user to the database with a hashed password."""
    password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username, password_hash.decode('utf-8'))
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

def verify_user(username, password):
    """Verifies a user's credentials."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
        result = cursor.fetchone()
        if result:
            password_hash = result[0].encode('utf-8')
            return bcrypt.checkpw(password.encode('utf-8'), password_hash)
        return False

def get_user_id_by_username(username):
    """Retrieves a user's ID by their username."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        result = cursor.fetchone()
        return result[0] if result else None

def list_users():
    """Returns a list of all usernames."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT username FROM users")
        return [row[0] for row in cursor.fetchall()]

# --- Job Management ---

def create_job(user_id, job_type):
    """Creates a new job entry and returns the job ID."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        now = datetime.utcnow()
        cursor.execute(
            "INSERT INTO jobs (user_id, job_type, status, created_at) VALUES (?, ?, ?, ?)",
            (user_id, job_type, 'pending', now)
        )
        conn.commit()
        return cursor.lastrowid

def update_job_status(job_id, status, result_path=None):
    """Updates the status and result of a job."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        now = datetime.utcnow()
        cursor.execute(
            "UPDATE jobs SET status = ?, result_path = ?, completed_at = ? WHERE id = ?",
            (status, result_path, now, job_id)
        )
        conn.commit()

def get_user_jobs(user_id):
    """Retrieves all jobs for a given user."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, job_type, status, created_at, completed_at, result_path FROM jobs WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,)
        )
        return cursor.fetchall()

# --- Session Management ---

def save_session(user_id, session_data):
    """Saves or updates a user's session data."""
    serialized_data = pickle.dumps(session_data)
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        now = datetime.utcnow()
        cursor.execute(
            "INSERT OR REPLACE INTO user_sessions (user_id, session_data, updated_at) VALUES (?, ?, ?)",
            (user_id, serialized_data, now)
        )
        conn.commit()

def load_session(user_id):
    """Loads a user's session data."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT session_data FROM user_sessions WHERE user_id = ?", (user_id,))
        result = cursor.fetchone()
        if result:
            return pickle.loads(result[0])
        return None

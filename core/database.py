import sqlite3
import bcrypt
import os

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'users.db')

def init_db():
    """
    Initializes the database, creates the users table if it doesn't exist,
    and adds a sample user if the table is empty.
    """
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL
            )
        """)

        # Check if the users table is empty
        cursor.execute("SELECT COUNT(*) FROM users")
        count = cursor.fetchone()[0]

        if count == 0:
            # Add a sample user if the table is empty
            username = "jane.doe"
            password = "password123"
            password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            cursor.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username, password_hash.decode('utf-8'))
            )

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
            return False  # Username already exists

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

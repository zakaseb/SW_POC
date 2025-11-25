import sqlite3
import bcrypt
import os
import pickle
import json
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
        # User sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_sessions (
                user_id INTEGER PRIMARY KEY,
                session_data BLOB NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        # Requirement generation jobs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS requirement_jobs (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                status TEXT NOT NULL,
                result_path TEXT,
                metadata TEXT,
                error_message TEXT,
                created_at TIMESTAMP NOT NULL,
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

def delete_session(user_id):
    """Deletes a user's session data."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM user_sessions WHERE user_id = ?", (user_id,))
        conn.commit()


# --- Requirement Job Helpers ---

def _job_row_to_dict(row):
    if not row:
        return None
    metadata = {}
    if row[4]:
        try:
            metadata = json.loads(row[4])
        except json.JSONDecodeError:
            metadata = {}
    return {
        "id": row[0],
        "user_id": row[1],
        "status": row[2],
        "result_path": row[3],
        "metadata": metadata,
        "error_message": row[5],
        "created_at": row[6],
        "updated_at": row[7],
    }


def create_requirement_job(job_id, user_id, status="queued", metadata=None):
    """Creates a new requirement generation job entry."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        now = datetime.utcnow()
        cursor.execute(
            """
            INSERT INTO requirement_jobs (id, user_id, status, result_path, metadata, error_message, created_at, updated_at)
            VALUES (?, ?, ?, NULL, ?, NULL, ?, ?)
            """,
            (
                job_id,
                user_id,
                status,
                json.dumps(metadata or {}),
                now,
                now,
            ),
        )
        conn.commit()


def update_requirement_job(job_id, status=None, result_path=None, error_message=None, metadata=None):
    """Updates fields on an existing requirement job entry."""
    fields = []
    values = []
    if status is not None:
        fields.append("status = ?")
        values.append(status)
    if result_path is not None:
        fields.append("result_path = ?")
        values.append(result_path)
    if error_message is not None:
        fields.append("error_message = ?")
        values.append(error_message)
    if metadata is not None:
        fields.append("metadata = ?")
        values.append(json.dumps(metadata))

    if not fields:
        return

    values.append(datetime.utcnow())
    fields.append("updated_at = ?")
    values.append(job_id)

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"UPDATE requirement_jobs SET {', '.join(fields)} WHERE id = ?",
            values,
        )
        conn.commit()


def get_requirement_job(job_id):
    """Fetches a single requirement job by id."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, user_id, status, result_path, metadata, error_message, created_at, updated_at
            FROM requirement_jobs
            WHERE id = ?
            """,
            (job_id,),
        )
        row = cursor.fetchone()
    return _job_row_to_dict(row)


def get_latest_requirement_job(user_id):
    """Returns the most recent job for a user."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, user_id, status, result_path, metadata, error_message, created_at, updated_at
            FROM requirement_jobs
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (user_id,),
        )
        row = cursor.fetchone()
    return _job_row_to_dict(row)


def list_requirement_jobs(user_id, limit=5):
    """Returns up to `limit` recent jobs for a user."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, user_id, status, result_path, metadata, error_message, created_at, updated_at
            FROM requirement_jobs
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        )
        rows = cursor.fetchall()
    return [_job_row_to_dict(row) for row in rows]

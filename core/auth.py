import streamlit as st
from .database import verify_user, init_db, get_user_id_by_username, load_session
from .session_utils import unpack_session_from_storage

def show_login_form():
    """Displays the login form and returns True if the user is authenticated."""
    init_db()
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.title("Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login"):
            if verify_user(username, password):
                st.session_state.authenticated = True
                user_id = get_user_id_by_username(username)
                st.session_state.user_id = user_id
                st.session_state.username = username

                # Load the user's session
                loaded_state = load_session(user_id)
                if loaded_state:
                    unpack_session_from_storage(loaded_state)

                st.rerun()
            else:
                st.error("Invalid username or password")

    return st.session_state.authenticated

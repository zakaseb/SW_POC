import streamlit as st
from .database import verify_user, init_db

def show_login_form():
    """Displays the login form and returns True if the user is authenticated."""
    init_db()
    st.session_state.authenticated = st.session_state.get('authenticated', False)

    if not st.session_state.authenticated:
        st.title("Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login"):
            if verify_user(username, password):
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid username or password")

    return st.session_state.authenticated

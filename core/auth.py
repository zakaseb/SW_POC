import streamlit as st
from .database import verify_user, init_db, get_user_id_by_username, load_session
from .session_utils import unpack_session_from_storage
from .document_processing import re_index_documents_from_session

def show_login_form():
    """Displays the login form and returns True if the user is authenticated."""
    init_db()
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.title("Log in")
        with st.form("login_form"):
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Log in", type="primary")

        if submitted:
            if verify_user(username, password):
                st.session_state.authenticated = True
                user_id = get_user_id_by_username(username)
                st.session_state.user_id = user_id
                st.session_state.username = username

                # Load the user's session
                loaded_state = load_session(user_id)
                if loaded_state:
                    unpack_session_from_storage(loaded_state)
                    # If documents were processed in the loaded session, re-index them
                    if st.session_state.get("document_processed"):
                        re_index_documents_from_session()

                st.rerun()
            else:
                st.error("Invalid username or password")

    return st.session_state.authenticated

import argparse
import getpass
from core.database import init_db, add_user, list_users

def main():
    parser = argparse.ArgumentParser(description="User management script.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 'init' command
    parser_init = subparsers.add_parser("init", help="Initialize the database.")

    # 'add' command
    parser_add = subparsers.add_parser("add", help="Add a new user.")

    # 'list' command
    parser_list = subparsers.add_parser("list", help="List all users.")

    args = parser.parse_args()

    if args.command == "init":
        init_db()
        print("Database initialized successfully.")

    elif args.command == "add":
        init_db()  # Ensure DB is initialized before adding a user
        username = input("Enter username (firstname.lastname): ")
        password = getpass.getpass("Enter password: ")
        password_confirm = getpass.getpass("Confirm password: ")

        if password != password_confirm:
            print("Passwords do not match.")
            return

        if add_user(username, password):
            print(f"User '{username}' added successfully.")
        else:
            print(f"User '{username}' already exists.")

    elif args.command == "list":
        init_db()  # Ensure DB is initialized before listing users
        users = list_users()
        if users:
            print("Existing users:")
            for user in users:
                print(f"- {user}")
        else:
            print("No users found.")

if __name__ == "__main__":
    main()

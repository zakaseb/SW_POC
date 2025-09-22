import os
import time
from .database import update_job_status, load_session
from .generation import generate_requirements_json, generate_excel_file
from .model_loader import get_language_model

# Define the storage path for job results
JOB_RESULTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'job_results')
os.makedirs(JOB_RESULTS_PATH, exist_ok=True)

def process_requirements_job(job_id, user_id):
    """
    The main function for the background process to generate requirements.
    """
    try:
        # 1. Update job status to 'running'
        update_job_status(job_id, 'running')

        # 2. Load user session to get context
        session_state = load_session(user_id)
        if not session_state:
            raise ValueError("Could not load user session to process job.")

        # 3. Load the language model within the new process
        language_model = get_language_model()
        if not language_model:
            raise RuntimeError("Failed to load language model in background process.")

        # 4. Get the required data from the loaded session state
        requirements_chunks = session_state.get('requirements_chunks', [])
        if not requirements_chunks:
            raise ValueError("No requirement chunks found in the user's session data to process.")

        # 5. Perform the long-running task
        all_requirements_json = []
        for chunk in requirements_chunks:
            # This is the time-consuming part
            json_response = generate_requirements_json(language_model, chunk)
            all_requirements_json.append(json_response)

        excel_data = generate_excel_file(all_requirements_json)
        if not excel_data:
            raise ValueError("Generated requirements but failed to create Excel file.")

        # 6. Save the result file
        result_filename = f"job_{job_id}_requirements.xlsx"
        result_filepath = os.path.join(JOB_RESULTS_PATH, result_filename)
        with open(result_filepath, 'wb') as f:
            f.write(excel_data)

        # 7. Update job status to 'completed'
        update_job_status(job_id, 'completed', result_path=result_filepath)

    except Exception as e:
        # 8. Handle errors
        error_message = f"Job failed: {e}"
        print(error_message) # Log to console for debugging
        # The `result_path` can be used to store the error message
        update_job_status(job_id, 'failed', result_path=error_message)

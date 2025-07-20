# app.py

"""
The main unified Flask application for the Smart Insurance Claim Advisor.
This single application serves the HTML/CSS/JS frontend and provides all backend APIs.
- It replaces the separate Flask API (api.py) and Streamlit UI (app.py).
- It uses Server-Sent Events (SSE) for real-time streaming of chat responses.
"""

import os
import json
from flask import Flask, request, jsonify, Response, render_template
from werkzeug.utils import secure_filename

from config.settings import settings
from src.ingest import process_and_embed_documents
from src.conversation import get_full_response
from utils.logging_config import get_logger

logger = get_logger(__name__)

# Initialize Flask App
# The 'template_folder' points to where index.html is located.
app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = settings.DOCUMENTS_DIR
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB upload limit

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def home():
    """Renders the main chat interface."""
    return render_template('index.html')

@app.route('/ingest', methods=['POST'])
def ingest_documents_route():
    """
    Endpoint to upload and process documents.
    Expects 'multipart/form-data' with one or more files.
    """
    if 'files' not in request.files:
        logger.warning("Ingest request received with no files part.")
        return jsonify({"error": "No files part in the request"}), 400

    files = request.files.getlist('files')
    if not files or all(f.filename == '' for f in files):
        logger.warning("Ingest request received with no selected files.")
        return jsonify({"error": "No selected files"}), 400

    saved_file_paths = []
    for file in files:
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            saved_file_paths.append(file_path)
    
    logger.info(f"Received {len(saved_file_paths)} files for ingestion.")

    try:
        # For production, this should be offloaded to a background worker (e.g., Celery)
        process_and_embed_documents(file_paths=saved_file_paths)
        return jsonify({
            "message": f"Successfully ingested {len(saved_file_paths)} files.",
            "filenames": [os.path.basename(p) for p in saved_file_paths]
        }), 200
    except Exception as e:
        logger.error(f"Error during ingestion process: {e}", exc_info=True)
        return jsonify({"error": "An error occurred during document processing."}), 500

@app.route('/chat', methods=['POST'])
def chat_route():
    """
    Endpoint to handle chat interactions using Server-Sent Events (SSE) for streaming.
    """
    data = request.get_json()
    query = data.get('query')
    chat_history = data.get('chat_history', [])

    if not query:
        return jsonify({"error": "Query is a required field."}), 400

    logger.info(f"Received chat query for SSE stream: '{query}'")

    def generate_stream():
        """Generates the SSE stream with enhanced logging."""
        logger.info("Stream generation started.")
        try:
            # get_full_response is a generator yielding structured ClaimDecision objects
            for i, response_chunk in enumerate(get_full_response(query, chat_history)):
                logger.info(f"Received chunk {i} from LLM.")
                # Format for SSE: 'data: {json_string}\n\n'
                sse_data = f"data: {response_chunk.json()}\n\n"
                yield sse_data
            logger.info("Stream generation finished successfully.")
        except Exception as e:
            logger.error(f"Error during stream generation: {e}", exc_info=True)
            error_response = {
                "error": "An error occurred on the server while generating the response.",
                "details": str(e)
            }
            yield f"data: {json.dumps(error_response)}\n\n"

    # The mimetype 'text/event-stream' is crucial for SSE to work.
    return Response(generate_stream(), mimetype='text/event-stream')

if __name__ == '__main__':
    # We've added `use_reloader=False` to prevent the server from restarting
    # itself and loading the AI model twice. This will significantly speed up
    # the startup time during development.
    app.run(host='0.0.0.0', port=settings.API_PORT, debug=True, use_reloader=False)

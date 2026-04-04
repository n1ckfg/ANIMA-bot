import os
import logging
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from rag_system import get_rag_system

app = Flask(__name__)

# Initialize the RAG system globally
config_path = os.environ.get("ANIMA_CONFIG", "./config.yaml")
rag_system = None


def get_rag():
    """Lazy initialization of RAG system."""
    global rag_system
    if rag_system is None:
        rag_system = get_rag_system(config_path)
    return rag_system


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint for monitoring."""
    status = {"status": "healthy", "components": {}}

    try:
        rag = get_rag()
        status["components"]["rag_system"] = "ok"
        status["components"]["index_loaded"] = rag.index is not None
        status["components"]["llm"] = "ok" if rag.llm else "not_initialized"
        status["components"]["embeddings"] = "ok" if rag.embed_model else "not_initialized"
        status["components"]["cache_enabled"] = rag.cache.enabled if rag.cache else False
        status["components"]["hyde_enabled"] = rag.hyde is not None
    except Exception as e:
        status["status"] = "unhealthy"
        status["error"] = str(e)
        return jsonify(status), 503

    return jsonify(status)


@app.route('/query', methods=['POST'])
def query():
    """API endpoint to query the RAG system."""
    data = request.get_json()
    query_text = data.get('query', '').strip()
    use_cache = data.get('use_cache', True)

    if not query_text:
        return jsonify({'error': 'Query cannot be empty'}), 400

    try:
        rag = get_rag()
        response = rag.query(query_text, use_cache=use_cache)
        return jsonify({
            'query': query_text,
            'response': response
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/query/stream', methods=['POST'])
def query_stream():
    """Streaming API endpoint for real-time responses."""
    data = request.get_json()
    query_text = data.get('query', '').strip()

    if not query_text:
        return jsonify({'error': 'Query cannot be empty'}), 400

    def generate():
        try:
            rag = get_rag()
            for chunk in rag.query_stream(query_text):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )


@app.route('/cache/clear', methods=['POST'])
def clear_cache():
    """Clear the query cache."""
    try:
        rag = get_rag()
        rag.clear_cache()
        return jsonify({'status': 'ok', 'message': 'Cache cleared'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/reindex', methods=['POST'])
def reindex():
    """Trigger a full reindex of documents."""
    try:
        rag = get_rag()
        rag.reindex()
        return jsonify({'status': 'ok', 'message': 'Reindex complete'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Initialize RAG system on startup
    print("Initializing RAG system...")
    get_rag()
    print("RAG system ready!")

    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    app.run(debug=True, host='0.0.0.0', port=8080)

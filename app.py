from flask import Flask, render_template, request, jsonify
from rag_system import setup_rag

app = Flask(__name__)

# Initialize the RAG query engine globally
query_engine = setup_rag()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    """API endpoint to query the RAG system."""
    data = request.get_json()
    query_text = data.get('query', '').strip()

    if not query_text:
        return jsonify({'error': 'Query cannot be empty'}), 400

    try:
        response = query_engine.query(query_text)
        return jsonify({
            'query': query_text,
            'response': str(response)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)

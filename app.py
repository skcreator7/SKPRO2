from flask import Flask, send_from_directory, jsonify, request
import os

app = Flask(__name__)

@app.route('/')
def index():
    """Serve home page"""
    return send_from_directory('.', 'index.html')

@app.route('/search.html')
def search():
    """Serve search page"""
    return send_from_directory('.', 'search.html')

@app.route('/post.html')
def post():
    """Serve post page"""
    try:
        return send_from_directory('.', 'post.html')
    except:
        return send_from_directory('.', 'index.html')

@app.route('/health')
def health():
    """Health check"""
    return jsonify({
        "status": "healthy",
        "service": "SK4FiLM Frontend",
        "timestamp": "2025-11-14"
    })

@app.route('/<path:path>')
def static_proxy(path):
    """Serve any static file"""
    try:
        return send_from_directory('.', path)
    except Exception as e:
        print(f"Error serving {path}: {e}")
        # Return index for SPA routing
        if not '.' in path:
            return send_from_directory('.', 'index.html')
        return jsonify({"error": "Not found"}), 404

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

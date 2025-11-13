from flask import Flask, send_from_directory, jsonify
import os

app = Flask(__name__)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/search.html')
def search():
    return send_from_directory('.', 'search.html')

@app.route('/post.html')
def post():
    return send_from_directory('.', 'post.html')

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "service": "SK4FiLM Frontend"})

@app.route('/<path:path>')
def static_files(path):
    try:
        return send_from_directory('.', path)
    except:
        return send_from_directory('.', 'index.html')

if __name__ == '__main__':
    app.run()

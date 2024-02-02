from flask import Flask, request, jsonify
import subprocess
import json

app = Flask(__name__)

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/rag', methods=['POST'])
def run_rag():
    data = request.json
    
    # Extract parameters or use default values
    file_path = data.get('file_path', 'sibs.pdf')
    query = data.get('query', 'Whats so civil about war anyway?')
    top_k = data.get('top_k', 10)

    # Run your script with the provided arguments
    try:
        process = subprocess.run(
            ['python3', 'main.py', '--file_path', file_path, '--query', query, '--top_k', str(top_k)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        response = {
            "status": "success",
            "output": process.stdout,
            "error": process.stderr
        }
    except subprocess.CalledProcessError as e:
        response = {
            "status": "error",
            "output": e.stdout,
            "error": e.stderr
        }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=9001)

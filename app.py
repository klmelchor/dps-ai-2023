from flask import Flask, request
import json

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello World"

@app.route('/api/task', methods=['POST'])
def get_query():
    query_data = request.get_json()
    response = {
        'prediction': query_data['year']
    }
    return json.dumps(response)
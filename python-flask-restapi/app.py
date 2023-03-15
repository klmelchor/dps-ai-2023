from flask import Flask, request
import json
import accident_prediction as acc_pred

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello World"

@app.route('/api/task', methods=['POST'])
def get_query():
    query_data = request.get_json()
    year = int(query_data['year'])
    month = int(query_data['month'])
    prediction = acc_pred.main_func(year, month)
    response = {
        'prediction': prediction
    }
    return json.dumps(response)
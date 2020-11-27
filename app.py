import sys, os, json, pdb
import hashlib
import flask
from flask import request, jsonify
from model import Summariser

app = flask.Flask(__name__)
app.config["DEBUG"] = False
app.config['Summariser'] = Summariser()

@app.route('/api/v2/summariser', methods=['GET'])
def answer_question():
    """
    This api calls the summarisation module and performs summarisation

    Inputs
    ------
    Expects a api call of the form : 
    {
        body: "The body of the text that must be highlighted"
    }
    
    Query : String
        The string from which we need to summarises

    Outputs
    -------
    Json Object : 
        The form of the json object is as follows : -
    {
        markdown_text : "The text which is editted to highlight text in markdown"
        summary : "Summary Sentences"
    }
    """
    request_json = json.loads(request.data, strict=False)
    requires = [
            'body',
        ]
    for x in requires:
        if x not in request_json.keys():
            return jsonify({"message":"given request does not have a "+x })
    
    body = request_json['body']
    summary, markdown_text = app.config['Summariser'].cluster(body)

    response = {
        "summary":summary,
        "markdown_text":markdown_text,
    }

    return jsonify(response)

@app.route('/')
def hello_world():
    return 'Hello, World! The service is up for serving summarisation highlighting'
        
if __name__ == '__main__':
    app.run(host='0.0.0.0', port = 5000)
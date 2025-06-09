from flask import Flask, jsonify
from flask import request
import json

app = Flask(__name__)

@app.route('/list-tools', methods=['GET'])
def list_tools():
    with open('tools.json') as f:
        tools = json.load(f)
    return jsonify(tools)

def process_video_and_generate_report(video_url, vhc_json, debug):
    return video_url.replace("https", "http")

def generate_vehicle_report(video_url, vhc_json, debug=False):
    # You already have this logic in your existing pipeline
    # Call the backend function that processes video + VHC
    output_url = process_video_and_generate_report(video_url, vhc_json, debug)
    return {"video_url": output_url}


@app.route('/call-tool', methods=['POST'])
def call_tool():
    data = request.json
    tool_name = data.get('tool_name')
    arguments = data.get('arguments', {})

    # Mapping between tool names and handler functions
    TOOL_MAP = {
        "generate_vehicle_report": generate_vehicle_report
        # add other tool handlers here
    }

    if tool_name not in TOOL_MAP:
        return jsonify({"error": "Unknown tool name"}), 400

    try:
        result = TOOL_MAP[tool_name](**arguments)
        return jsonify({"output": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
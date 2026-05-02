import os
import json
import pandas as pd
from flask import Flask, render_template, request, jsonify
from groq import Groq
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

load_dotenv()
app = Flask(__name__)

# Core Config
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)
llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile", temperature=0)

global_df = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/process', methods=['POST'])
def process():
    global global_df
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"})
    
    file = request.files['file']
    try:
        # Smart Reading
        if file.filename.lower().endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            df = pd.read_csv(file, encoding='latin1')
        
        global_df = df.dropna(how='all')
        
        # Professional Statistical Extraction
        numeric_df = global_df.select_dtypes(include=['number'])
        if numeric_df.empty:
            return jsonify({"success": False, "error": "No numeric data found for charts."})

        stats = {
            "cols": numeric_df.columns.tolist()[:5],
            "means": [round(float(val), 2) for val in numeric_df.mean().tolist()[:5]],
            "maxs": [round(float(val), 2) for val in numeric_df.max().tolist()[:5]],
            "count": len(global_df)
        }
        
        return jsonify({"success": True, "stats": stats})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/chat', methods=['POST'])
def chat():
    global global_df
    if global_df is None: return jsonify({"success": False, "error": "Upload data first"})
    
    query = request.json.get('query')
    try:
        # Multilingual Expert Instruction
        instruction = "System: Respond in the language used by the user (Arabic or English). Be a senior data consultant. Use precise data."
        agent = create_pandas_dataframe_agent(llm, global_df, verbose=False, allow_dangerous_code=True)
        response = agent.run(f"{instruction}\nUser Question: {query}")
        return jsonify({"success": True, "answer": response})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
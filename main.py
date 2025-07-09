# main.py

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal
import google.generativeai as genai
from dotenv import load_dotenv

app = FastAPI(
    title="AgriBot API",
    description="A multilingual API chatbot for agricultural questions.",
    version="1.0.0",
)

# confiiiiiiiiig
def setup_gemini():
    """Loads API key and configures the Gemini model."""
    try:
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google API Key not found in .env file.")
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        return model
    except Exception as e:
        print(f"Error during Gemini setup: {e}")
        return None

# Load the model on startup
gemini_model = setup_gemini()

#limitn


OFF_TOPIC_MESSAGES = {
    'en': "I am an agricultural assistant. My knowledge is focused on topics like farming, crops, and livestock. Please ask a question related to agriculture.",
    'es': "Soy un asistente agrícola. Mi conocimiento se centra en temas como la agricultura, los cultivos y el ganado. Por favor, haga una pregunta relacionada con la agricultura.",
    'bn': "আমি একজন কৃষি সহায়ক। আমার জ্ঞান চাষ, ফসল এবং পশুপালনের মতো বিষয়ে সীমাবদ্ধ। অনুগ্রহ করে কৃষি সম্পর্কিত একটি প্রশ্ন জিজ্ঞাসা করুন।"
}


# pydnant fro req and res and fast api for automated validation 
class ChatRequest(BaseModel):
    query: str
    language: Literal['en', 'es', 'bn'] = 'en' # Default to English

# Defines the structure of the outgoing response.
class ChatResponse(BaseModel):
    response: str


# core

def is_agricultural_question(user_question: str, model: genai.GenerativeModel) -> bool:
    """Uses the LLM to determine if a question is related to agriculture."""
    gatekeeper_prompt = f"""
    Is the following user question related to agriculture, farming, crops, soil science,
    livestock, horticulture, agricultural technology, or aquaculture?
    Please answer with only a single word: 'YES' or 'NO'.

    User Question: "{user_question}"
    """
    try:
        response = model.generate_content(gatekeeper_prompt)
        answer = response.text.strip().upper()
        return answer == "YES"
    except Exception as e:
        print(f"Error in gatekeeper check: {e}")
        # Fail safe: if unsure, treat it as a valid question to be answered.
        # An alternative would be to return False and give an error.
        return True

def get_agricultural_answer(user_question: str, model: genai.GenerativeModel) -> str:
    """
    Generates a helpful answer, responding in the same language as the user's question.
    """
    main_prompt = f"""
    You are AgriBot, a friendly and knowledgeable AI assistant specializing in agriculture.
    Your goal is to provide accurate, helpful, and concise answers.

    IMPORTANT: First, identify the language of the user's question below.
    Then, respond fully and helpfully in that SAME language.

    User Question: "{user_question}"
    """
    try:
        response = model.generate_content(main_prompt)
        return response.text
    except Exception as e:
        print(f"Error in answer generation: {e}")
        raise HTTPException(status_code=500, detail="Error generating response from the AI model.")


#  end point 
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main endpoint to handle user chat requests.
    It validates the topic and returns an appropriate response.
    """
    if not gemini_model:
        raise HTTPException(status_code=503, detail="Gemini model is not available.")

    # 1. Gatekeeper Check \ limitn
    if is_agricultural_question(request.query, gemini_model):
        # 2. If agricultural, get the real answer from Gemini
        answer = get_agricultural_answer(request.query, gemini_model)
    else:
        # 3. If not, provide the pre-written guidance message in the requested language
        answer = OFF_TOPIC_MESSAGES.get(request.language, OFF_TOPIC_MESSAGES['en'])
    
    return ChatResponse(response=answer)

# A simple root endpoint to confirm the API is running
@app.get("/")
def root():
    return {"message": "AgriBot API is running. Go to /docs for the interactive API documentation."}
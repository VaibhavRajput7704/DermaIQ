# gemini_chat_helper.py

import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

def init_chat():
    genai.configure(api_key=API_KEY)

    system_prompt = (
        "You are a medical AI assistant named DermaIQ. Only provide concise, medically accurate treatment suggestions "
        "for first, second, or third-degree skin burns. Keep responses short and to the point (2–4 sentences max). "
        "Politely refuse to answer unrelated or non-medical questions."
    )

    model = genai.GenerativeModel("models/gemini-2.0-flash",
                                  system_instruction=system_prompt)

    return model.start_chat(history=[])

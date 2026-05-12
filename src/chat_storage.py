import json
import os

CHAT_FILE = "chat_history/chat_history.json"

# Create folder if not exists
os.makedirs("chat_history", exist_ok=True)

def save_chat(messages):

    with open(CHAT_FILE, "w", encoding="utf-8") as file:
        json.dump(messages, file, indent=4)

def load_chat():

    if os.path.exists(CHAT_FILE):

        with open(CHAT_FILE, "r", encoding="utf-8") as file:
            return json.load(file)

    return []
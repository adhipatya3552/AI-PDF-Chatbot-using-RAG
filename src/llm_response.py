import os
import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


def generate_response(prompt):

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "mistralai/mistral-7b-instruct",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    try:

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=60
        )

        result = response.json()

        print("API RESPONSE:", result)

        # Check if choices exist
        if "choices" in result:

            return result["choices"][0]["message"]["content"]

        # Handle API errors gracefully
        elif "error" in result:

            return f"❌ API Error: {result['error']['message']}"

        else:

            return "❌ Unexpected API response received."

    except Exception as e:

        return f"❌ Error generating response: {str(e)}"

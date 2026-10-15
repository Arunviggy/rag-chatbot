
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("ERROR: GEMINI_API_KEY was not found in .env")
else:
    try:
        client = genai.Client(api_key=api_key)

        print("\nModels available for text generation:\n")

        found = False
        for model in client.models.list():
            actions = getattr(model, "supported_actions", []) or []

            if "generateContent" in actions:
                print(model.name)
                found = True

        if not found:
            print("No models supporting generateContent were listed.")

    except Exception as error:
        print("Could not list models:", error)

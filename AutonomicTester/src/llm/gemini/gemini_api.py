import google.generativeai as genai
import keyring


def check_gemini_api():
    # Service name for keyring (choose a descriptive name)
    SERVICE_NAME = "gemini-api-key"

    # Try to get the API key from keyring
    API_KEY = keyring.get_password(SERVICE_NAME, None)

    # If not found, prompt the user
    if not API_KEY:
        API_KEY = input("Enter your Google API Key: ")
        keyring.set_password(SERVICE_NAME, None, API_KEY)
    
    return API_KEY

def call_gemini_api(prompt: str, model: str = 'gemini-pro'):

    GOOGLE_API_KEY = check_gemini_api()

    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel(model)
    response = model.generate_content(prompt)

    return response.text
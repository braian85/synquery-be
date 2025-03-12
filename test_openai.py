import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Get OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY", "")
print(f"OpenAI API key: {openai_api_key[:5]}...{openai_api_key[-5:] if openai_api_key else ''}")

if not openai_api_key:
    print("WARNING: OpenAI API key is not set!")
    exit(1)

try:
    # Initialize the OpenAI client
    client = OpenAI(api_key=openai_api_key)
    
    # Make a simple API call
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt="Hello, how are you?",
        max_tokens=10
    )
    
    print("API call successful!")
    print(f"Response: {response.choices[0].text}")
except Exception as e:
    import traceback
    print(f"Error calling OpenAI API: {e}")
    print(traceback.format_exc()) 
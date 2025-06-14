import os
import sys
from google import genai
from google.genai import types
from dotenv import load_dotenv


def main():
    # Load environment variables
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")

    # Check for verbose flag and extract user prompt
    verbose = False
    args = sys.argv[1:]

    if not args:
        print("Error: No prompt provided.")
        sys.exit(1)

    if "--verbose" in args:
        verbose = True
        args.remove("--verbose")

    if not args:
        print("Error: No prompt provided.")
        sys.exit(1)

    user_prompt = " ".join(args)

    # Create Gemini-compatible message
    messages = [
        types.Content(role="user", parts=[types.Part(text=user_prompt)]),
    ]

    # Set up Gemini client and send request
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=messages,
    )

    # Conditionally print verbose output
    if verbose:
        print(f'User prompt: {user_prompt}')
        print(f'Prompt tokens: {response.usage_metadata.prompt_token_count}')
        print(f'Response tokens: {response.usage_metadata.candidates_token_count}')

    # Always print the model's response
    print("Response:")
    print(response.text)


if __name__ == "__main__":
    main()

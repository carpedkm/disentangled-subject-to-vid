import os
import json
import time
from tqdm import tqdm
from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI

# Azure OpenAI Setup
credential = AzureCliCredential()
token_provider = get_bearer_token_provider(
    credential,
    "https://cognitiveservices.azure.com/.default"
)

aoiclient = AzureOpenAI(
    azure_endpoint="https://t2vgoaigpt4o.openai.azure.com/",
    azure_ad_token_provider=token_provider,
    api_version="2024-02-15-preview",
    max_retries=5,
)

# Function to generate "foreground_prompt" using GPT with retry mechanism
def generate_foreground_prompt_with_retry(text, objects, retries=5):
    # Convert objects list to string format expected by GPT
    objects_string = '.'.join(objects.split('.')).strip('.')

    # Prepare the prompt
    prompt = f"""
    Your task is to generate a foreground-focused version of a given sentence by removing words that refer to the background. Use the following instructions:

    1. Given a sentence, identify all nouns referring to foreground objects based on the provided list of objects.
    2. Remove all nouns, phrases, or words in the sentence that refer to background elements or settings.
    3. Ensure that the output sentence retains only those parts relevant to the provided objects, preserving proper grammar and flow.
    4. Do not include any additional words, descriptions, or alterations beyond removing background references.

    Here is an example:
    - Input sentence: "Father carrying his son on a beach."
    - Objects list: ["father", "son"]
    - Output: "Father carrying his son."

    Another example:
    - Input sentence: "A ray of the sun on the forest ground."
    - Objects list: ["ray", "sun"]
    - Output: "A ray of the sun."

    Input sentence: "{text}"
    Objects list: [{objects_string}]
    Provide only the filtered sentence as the result.
    """

    for attempt in range(retries):
        try:
            # Send request to Azure OpenAI
            response = aoiclient.chat.completions.create(
                model="gpt4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=100,
                top_p=0.5,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
            )
            # Extract response content
            result = response.choices[0].message.content.strip()
            if result:  # Check if the result is valid
                return result
        except Exception as e:
            print(f"Error processing text (Attempt {attempt + 1}/{retries}): {text} - {e}")
        time.sleep(2)  # Wait before retrying
    return None  # Return None if all retries fail

# Main function to process JSON and add "foreground_prompt"
def process_json_file(json_file_path, output_file_path):
    # Load the JSON file
    with open(json_file_path, "r") as f:
        video_dict = json.load(f)

    # Iterate through the video entries and generate "foreground_prompt"
    for video_id, data in tqdm(video_dict.items()):
        text = data.get("text", "")
        objects = data.get("objects", "")
        
        # Generate foreground prompt with retry
        foreground_prompt = generate_foreground_prompt_with_retry(text, objects, retries=5)
        if foreground_prompt:
            video_dict[video_id]["foreground_prompt"] = foreground_prompt
        else:
            video_dict[video_id]["foreground_prompt"] = "Error"

    # Save the updated JSON
    with open(output_file_path, "w") as f:
        json.dump(video_dict, f, indent=4)

# Paths to input and output JSON files
input_json_path = "/root/daneul/projects/refactored/CogVideo/annotation/video_dict_subset128.json"
output_json_path = "/root/daneul/projects/refactored/CogVideo/annotation/video_dict_foreground2_subset128.jsonn"

# Process the JSON file
process_json_file(input_json_path, output_json_path)

print("Foreground prompts added successfully!")
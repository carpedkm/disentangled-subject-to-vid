# %%
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

# %%
# Function to generate "foreground_prompt" using GPT with retry mechanism
def refine_prompt_with_retry(text, object_, retries=5):
    # Convert objects list to string format expected by GPT
    # objects_string = '.'.join(objects.split('.')).strip('.')

    # Prepare the prompt
    prompt = f"""
Your task is to generate a given sentence with key subject word removed. Use the following instructions:

1. Given a sentence, identify the key noun referring to the given item.
2. Remove all subject nouns, phrases, or words in the sentence that refer to given items and use 'it', 'the item' or 'this item'.
3. Ensure that the output sentence retains the whole part. Do not modify other part of the sentence.
4. Do not include any additional words, descriptions, or alterations beyond removing subject word.
5. Do not include or modify unnecessary part.
6. Do not shorten the sentence. Give out the whole sentence.

Here is an example:
- Input sentence: "The Eames Lounge Chair is placed in a modern city living room, 
visible through a large glass window that overlooks a bustling cityscape. 
The angle of the photo captures a side profile of the chair from a slightly elevated perspective, 
emphasizing its sleek contours and iconic design."
- Key item given : "Eames Lounge Chair"
- Output: "The item is placed in a modern city living room, visible through a 
large glass window that overlooks a bustling cityscape.
The angle of the photo captures a side profile of the chair from a slightly elevated perspective,
emphasizing its sleek contours and iconic design."

Another example:
- Input sentence: "The bunk bed is placed in a vibrant children's room in a suburban home. 
Shot from a slight overhead perspective, the photo captures the lively colors of the bed's frame 
and the cheerful bedspreads."
- Key item given : "Bunk Bed"
- Output: "It is placed in a vibrant children's room in a suburban home.
Shot from a slight overhead perspective, the photo captures the lively colors of the bed's frame"

Last example:
- Input sentence: "Captured in a rustic countryside dining room, 
the sideboard basks in the soft, warm light of an early evening golden hour."
- Key item given : "Sideboard"
- Output: "Captured in a runstic countryside dining room, 
the item basks in the soft, warm light of an early evening golden hour."

Key item given : "{object_}"
Provide only the filtered sentence as the result.
    """

    for attempt in range(retries):
        try:
            # Send request to Azure OpenAI
            response = aoiclient.chat.completions.create(
                model="gpt4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=260,
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
        meta_dict = json.load(f)
    
    object_ = meta_dict['item']
    description_0 = meta_dict['description_0']
    description_1 = meta_dict['description_1']
    
    # Prepare the text for GPT
    refined_text0 = refine_prompt_with_retry(description_0, object_, retries=3)
    refined_text1 = refine_prompt_with_retry(description_1, object_, retries=3)

    meta_dict["description_0_refined"] = refined_text0
    meta_dict["description_1_refined"] = refined_text1

    with open(output_file_path, "w") as f:
        json.dump(meta_dict, f, indent=4)


# Paths to input and output JSON files
input_json_path = "/root/daneul/projects/refactored/CogVideo/output/metadata_updated"
output_json_path = "/root/daneul/projects/refactored/CogVideo/output/metadata_update_refined/"

os.makedirs(output_json_path, exist_ok=True)
# Process the JSON file
input_json_lists = os.listdir(input_json_path)
for input_json_list in tqdm(input_json_lists):
    input_json_list_path = os.path.join(input_json_path, input_json_list)
    output_json_list_path = output_json_path  + input_json_list
    process_json_file(input_json_list_path, output_json_list_path)

# %%




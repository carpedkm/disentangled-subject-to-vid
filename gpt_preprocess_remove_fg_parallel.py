import os
import json
import time
from tqdm import tqdm
import multiprocessing
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
def generate_foreground_prompt_with_retry(args):
    video_id, text, objects, retries = args
    # objects_string = '.'.join(objects.split('.')).strip('.')
    results_for_the_all_objs = []
    for obj in objects:

        prompt = f"""
        Your task is to generate a background-focused version of a given sentence by removing words that refer to each object. Use the following instructions:

        1. Given a sentence, identify all nouns referring to objects based on the provided list of objects.
        2. Remove all nouns, phrases, or words in the sentence that refer to foreground elements or settings.
        3. Ensure that the output sentence retains only those parts relevant to the provided objects, preserving proper grammar and flow.
        4. Do not include any additional words, descriptions, or alterations beyond removing background references.
        5. If direct removal of word leads to awkward sentence, paraphrase

        Here is an example:
        - Input sentence: "Father carrying his son on a beach."
        - Objects list: ["Father"]
        - Output: "A boy on a beach"

        Another example:
        - Input sentence: "A ray of the sun on the forest ground."
        - Objects list: ["ray", "sun"]
        - Output: "Forest ground"

        Input sentence: "{text}"
        Objects list: [{obj}]
        Provide only the filtered sentence as the result.
        """

        for attempt in range(retries):
            try:
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
                # print(response.choices[0].message.content)
                result = response.choices[0].message.content.strip()
                if result:
                    results_for_the_all_objs.append(result)
                    break
            except Exception as e:
                print(f"Error processing video_id {video_id} (Attempt {attempt + 1}/3): {e}")
            time.sleep(2)  # Wait before retrying

        results_for_the_all_objs.append("Error")
    return video_id, results_for_the_all_objs

# Main function to process JSON and add "foreground_prompt" using multiprocessing
def process_json_file_multiprocessing(json_file_path, output_file_path, num_workers=8):
    # Load the JSON file
    with open(json_file_path, "r") as f:
        video_dict = json.load(f)

    
    foreground_preprocessed_image_path = "/mnt/carpedkm_data/preprocessed_4k_with_foreground/foreground_objects"
    foreground_preprocessed_image_list = os.listdir(foreground_preprocessed_image_path)
    vid_id_in_foreground_preprocessed_image_list = [str(id_.split('.')[0].split('_')[0].strip()) for id_ in foreground_preprocessed_image_list]
    vid_id_in_foreground_preprocessed_image_list = list(set(vid_id_in_foreground_preprocessed_image_list))
    
    # print(video_dict[])
    for vid_id in video_dict.keys():
        # print(video_dict[vid_id].keys())
        video_dict[vid_id]['ref_obj'] = []

    for image_path in foreground_preprocessed_image_list:
        vid_id = str(image_path.split('.')[0].split('_')[0].strip())
        if vid_id not in vid_id_in_foreground_preprocessed_image_list:
            video_dict[vid_id]['ref_obj'] = [image_path.split('.')[0].split('_')[-1].strip()]
        else:
            video_dict[vid_id]['ref_obj'].append(image_path.split('.')[0].split('_')[-1].strip())
            # print(image_path.split('.')[0].split('_')[-1].strip())
    tasks = [
        (video_id, data.get("text", ""), data.get("ref_obj", ""), 3)
        for video_id, data in video_dict.items()
    ]      
    # Use multiprocessing to process entries in parallel
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(generate_foreground_prompt_with_retry, tasks), total=len(tasks)))

    # Update the video_dict with foreground_prompts
    
    for video_id, background_prompt_list in results:
        # check errors
        video_dict[video_id]["background_prompts"] = []
        for background_prompt in background_prompt_list:
            if "Error" in background_prompt_list:
                video_dict[video_id]["background_prompts"].append(video_dict[video_id]["text"])
            else:
                video_dict[video_id]["background_prompts"].append(background_prompt)

    # Save the updated JSON
    with open(output_file_path, "w") as f:
        json.dump(video_dict, f, indent=4)

# Paths to input and output JSON files
input_json_path = "/root/daneul/projects/refactored/CogVideo/annotation/video_dict_foreground_subset4000.json"
output_json_path = "/root/daneul/projects/refactored/CogVideo/annotation/video_dict_fg_bg_subset4000.json"

# Process the JSON file with multiprocessing
process_json_file_multiprocessing(input_json_path, output_json_path, num_workers=16)

print("Foreground prompts added successfully with multiprocessing!")
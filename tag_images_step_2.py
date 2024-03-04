import os
import numpy as np
from PIL import Image
import shutil
import subprocess
import uuid
import argparse
import json
from advcaption.taggers import ImageTagger

# Adapted from OpenAI's Vision example 
from openai import OpenAI
import base64
import json
import demjson3

# Point to the local server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")


class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)

def copy_template_to_image_directory(image_directory, template_file='./template_boilerplate.json'):
    """
    Copies the template JSON file './template.json' into the specified image directory.

    Parameters:
    - image_directory: The path to the directory where the template JSON file will be copied.
    """

    # Check if the image directory exists
    if not os.path.isdir(image_directory):
        print(f"The directory {image_directory} does not exist.")
        return
    
    # Check if the template file exists
    if not os.path.isfile(template_file):
        print("Template file './template.json' does not exist.")
        return
    
    # Construct the destination path for the template file in the image directory
    destination_path = os.path.join(image_directory, 'template.json')
    
    # Copy the template file to the destination
    shutil.copy(template_file, destination_path)
    
    print(f"Template file copied to {destination_path}.")
    
    # Load the JSON content into a dictionary
    try:
        with open(destination_path, 'r') as file:
            template_data = json.load(file)
            print("Template JSON loaded into dictionary.")
            return template_data
    except json.JSONDecodeError as e:
        print(f"Error loading JSON from template file: {e}")
        return None

def load_image(image_path, error_directory):
    try:
        image = Image.open(image_path)  # Load the image using PIL
        img = np.array(image, np.uint8)  # Attempt the conversion to numpy array
    except OSError as e:
        if "image file is truncated" in str(e):  # Check for specific error message
            print(f"Error: {image_path} is truncated. Moving to error directory.")
            if not os.path.exists(error_directory):  # Create error directory if it doesn't exist
                os.makedirs(error_directory)
            shutil.move(image_path, os.path.join(error_directory, os.path.basename(image_path)))  # Move the erroneous image
            return None
    return img

def validate_images_imagemagick(image_directory, error_directory):
    if not os.path.exists(error_directory):  # Create the error directory if it doesn't exist
        os.makedirs(error_directory)
    for filename in os.listdir(image_directory):
        # Update the condition to include .webp and .webm
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp', '.webm')):
            image_path = os.path.join(image_directory, filename)
            try:
                subprocess.run(['identify', image_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)  # Validate image
            except subprocess.CalledProcessError as e:
                print(f"Error occurred with file {filename}: {e}")
                shutil.move(image_path, os.path.join(error_directory, filename))  # Move the erroneous image

def rename_files_in_directory(directory, pattern):
    """
    Renames all files in the specified directory to a given pattern followed by an incrementing number.

    Parameters:
    - directory: The directory containing the files to be renamed.
    - pattern: The pattern to use for renaming the files, e.g., 'image_'
    """
    if not os.path.isdir(directory):
        print(f"The directory {directory} does not exist.")
        return
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    files.sort()
    counter = 1
    new_files = []
    
    for filename in files:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp', '.webm')):
            extension = os.path.splitext(filename)[1]
            new_filename = f"{pattern}_{counter}{extension}"
            old_file = os.path.join(directory, filename)
            new_file = os.path.join(directory, new_filename)
            os.rename(old_file, new_file)
            new_files.append(new_file)
            counter += 1

    print(f"All files in {directory} have been renamed according to the ID '{pattern}'.")
    return new_files

def generate_unique_id(length=8):
    """
    Generates a unique identifier using a substring of a UUID4.
    
    Parameters:
    - length: The desired length of the identifier. Default is 8 characters.
    
    Returns:
    A string representing the unique identifier.
    """
    # Generate a random UUID
    unique_id_full = uuid.uuid4()
    
    # Convert the UUID to a string, remove hyphens, and take a substring of the specified length
    unique_id_short = str(unique_id_full).replace('-', '')[:length]
    
    return unique_id_short










parser = argparse.ArgumentParser(description="Process images in a directory.")
parser.add_argument("--image_directory", help="The directory containing images to process.")
args = parser.parse_args()

with open(r".\\prompts\\system_prompt_COT.txt", 'r', encoding='utf-8') as file:
    system_prompt = file.read()
chat_prompt = 'Describe this image using your template AND all of the system prompt instructions.'

template_file = os.path.join(args.image_directory, 'template.json')
with open(template_file, 'r', encoding='utf-8') as file:
    template_data = json.load(file)
    config = template_data

for filename in os.listdir(args.image_directory):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp', '.webm')):
        image_path = os.path.join(args.image_directory, filename)
        with open(os.path.join(args.image_directory, os.path.splitext(filename)[0]+'.json'), 'r', encoding='utf-8') as file:
            combined_results = json.load(file)

        # Check if 'caption' exists and is neither None nor an empty string
        if combined_results.get('caption'):
            continue
        
        template_internal = config['llava_config']['template_internal']
        template_blank = {key: '' for key in config['llava_config']['template_internal']}
        concept_focus = config['llava_config']['concept_focus']
        template = 'OUTPUT_TEMPLATE_WITH_INTERNAL_INSTRUCTIONS: ' + json.dumps(template_internal) + ' OUTPUT_TEMPLATE: ' + json.dumps(template_blank) + ' CONCEPT_FOCUS: ' + concept_focus + ' CAPTION_FILE: ' + combined_results['processed']
        
        path = image_path
        base64_image = ""
        try:
            image = open(path.replace("'", ""), "rb").read()
            base64_image = base64.b64encode(image).decode("utf-8")
        except:
            print("Couldn't read the image. Make sure the path is correct and the file exists.")
            continue

        try_again = True
        temp_modifier = 0.0
        while try_again:
            completion = client.chat.completions.create(
                model="local-model", # not used
                messages=[
                {
                    "role": "system",
                    "content": system_prompt + template,
                },
                {
                    "role": "user",
                    "content": [
                    {"type": "text", "text": chat_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                    ],
                }
                ],
                max_tokens=32000,
                stream=False,
                temperature=0.2 + temp_modifier,
                presence_penalty=1.1,
                top_p=0.95
            )

            # content = completion.choices[0].message.content
            # returned_json = json.loads(content)

            returned_json = None
            for choice in completion.choices:
                try:
                    # Attempt to parse the JSON content
                    content = choice.message.content.replace("\n", "")
                    returned_json = json.loads(content)
                    # If the above line does not raise an error, break out of the loop
                    break
                except json.JSONDecodeError:
                    try:
                        # Attempt to parse the JSON content
                        returned_json = demjson3.decode(content)
                        # If the above line does not raise an error, break out of the loop
                        break
                    except demjson3.JSONDecodeError as e:
                        print(f"Error parsing JSON: {e}")
                        continue

            # Check if we successfully parsed any JSON
            if returned_json:
                print("Successfully parsed JSON:", returned_json)
                try_again = False
            temp_modifier += 0.01
        
        combined_results['llm'] = returned_json
        cot_substrings = ["chain", "of", "thought"]
        # Iterate over a list of keys (to avoid RuntimeError for changing dict size during iteration)
        for key in list(returned_json.keys()):
            # Check if all substrings are present in the key, ignoring non-alphanumeric characters in the key
            if all(sub in ''.join(filter(str.isalnum, key)).lower() for sub in cot_substrings):
                # If found, pop the key and break, assuming only one such key needs to be removed
                returned_json.pop(key)
                break
            
        conceptfocus_substrings = ["concept", "focus"]
        # Iterate over a list of keys (to avoid RuntimeError for changing dict size during iteration)
        for key in list(returned_json.keys()):
            # Check if all substrings are present in the key, ignoring non-alphanumeric characters in the key
            if all(sub in ''.join(filter(str.isalnum, key)).lower() for sub in conceptfocus_substrings):
                # If found, pop the key and break, assuming only one such key needs to be removed
                returned_json.pop(key)
                break
        
        relevant_tags = returned_json.pop('relevant_tags')
        
        # Part 1: Process the dictionary and concatenate values
        concatenated_values = ', '.join([
            ', '.join([subvalue.replace('.', ',').strip() for subvalue in value]) if isinstance(value, list) 
            else value.replace('.', ',').strip()
            for value in returned_json.values()
        ])

        # Part 2: Process the comma delimited string, compare, and append if necessary
        # Convert the comma delimited string into a list
        comma_delimited_list = combined_results['processed'].split(',')
        comma_delimited_list = [value.replace('.', '').strip() for value in comma_delimited_list]
        relevant_tags = [value.replace('.', '').strip() for value in relevant_tags]
        combined_set = set(comma_delimited_list + relevant_tags)
        comma_delimited_list = list(combined_set)

        # Trim whitespace and check if each element is not in the concatenated string
        elements_not_in_concatenated = [element.strip() for element in comma_delimited_list if element.strip() not in concatenated_values]

        # If there are elements not in the concatenated string, append them
        if elements_not_in_concatenated:
            if concatenated_values:  # Check if concatenated_values is not empty
                concatenated_values += ', '  # Add a separator before appending new elements
            concatenated_values += ', '.join(elements_not_in_concatenated)
            
        # 'concatenated_values' now contains all substrings processed as per the instructions
        combined_results['caption'] = concatenated_values

        json_path = os.path.splitext(image_path)[0] + '.json'
        with open(json_path, 'w') as json_file:
            json.dump(combined_results, json_file, cls=NumpyEncoder, indent=4)
            
        txt_path = os.path.splitext(image_path)[0] + '.txt'
        with open(txt_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(concatenated_values)
            
        print(f"Saved combined results to {json_path} and {txt_path}.")
import os
import numpy as np
from PIL import Image
import shutil
import subprocess
import uuid
import argparse
from advcaption.taggers import ImageTagger
import io

# Adapted from OpenAI's Vision example 
from openai import OpenAI
import base64
import yaml

# Point to the local server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

def load_or_initialize_template(dirpath, backup_dirpath, default_template_path='./template/template.yaml'):
    """
    Loads a template YAML file from the specified directory. If it doesn't exist,
    copies a default template into the directory, updates its 'llm_config'->'concept_focus'
    to the directory name, and returns the template data.

    Parameters:
    - dirpath: The directory path to check for the template file and potentially update with default template.
    - default_template_path: Path to the default template file.
    
    Returns:
    A dictionary representing the loaded or initialized template YAML data.
    """
    template_file = os.path.join(dirpath, 'template.yaml')

    if not os.path.isfile(template_file):
        print(f"'{template_file}' does not exist. Using backup template instead.")
        template_file = os.path.join(backup_dirpath, 'template.yaml')
        if not os.path.isfile(template_file):
            print(f"'{template_file}' does not exist. Using default template instead.")
            template_file = default_template_path
        else:
            print(f"Found '{template_file}'. Loading...")
    else:
        print(f"Found '{template_file}'. Loading...")

    # Load the template data from either the existing or the copied default template file
    with open(template_file, 'r', encoding='utf-8') as file:
        template_data = yaml.safe_load(file)
    
    # Update 'concept_focus' with the directory's name
    dir_name = os.path.basename(dirpath)  # Extracts the folder name
    if 'llm_config' in template_data and 'concept_focus' not in template_data['llm_config']:
        template_data['llm_config']['concept_focus'] = dir_name

    # Return the updated template data
    return template_data


# class NumpyEncoder(YAML.YAMLEncoder):
#     """ Custom encoder for numpy data types """
#     def default(self, obj):
#         if isinstance(obj, np.integer):
#             return int(obj)
#         elif isinstance(obj, np.floating):
#             return float(obj)
#         elif isinstance(obj, np.ndarray):
#             return obj.tolist()
#         else:
#             return super(NumpyEncoder, self).default(obj)

def copy_template_to_image_directory(image_directory, template_file='./templates/template.yaml'):
    """
    Copies the template YAML file './templates.yaml' into the specified image directory.

    Parameters:
    - image_directory: The path to the directory where the template YAML file will be copied.
    """

    # Check if the image directory exists
    if not os.path.isdir(image_directory):
        print(f"The directory {image_directory} does not exist.")
        return
    
    # Check if the template file exists
    if not os.path.isfile(template_file):
        print("Template file './templates.yaml' does not exist.")
        return
    
    # Construct the destination path for the template file in the image directory
    destination_path = os.path.join(image_directory, 'template.yaml')
    
    # Copy the template file to the destination
    shutil.copy(template_file, destination_path)
    
    print(f"Template file copied to {destination_path}.")
    
    # Load the YAML content into a dictionary
    try:
        with open(destination_path, 'r') as file:
            template_data = yaml.safe_load(file)
            print("Template YAML loaded into dictionary.")
            return template_data
    except yaml.YAMLDecodeError as e:
        print(f"Error loading YAML from template file: {e}")
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




## system prompt
## expert list
## default message 'Describe this image using your template AND all of the system prompt instructions.'
## expert attachment to the message
## 





parser = argparse.ArgumentParser(description="Process images in a directory.")
parser.add_argument("--image_directory", help="The directory containing images to process.")
parser.add_argument("--overwrite", action="store_true", help="Overwrite the LLM captions from a previous run.")
parser.add_argument("--default_template", help="Choose the template file to use as your default.", default="./templates/template.yaml")
parser.add_argument("--skipconcept", action="store_true", help="Skip adding a customized concept per subdirectory.")
args = parser.parse_args()

for dirpath, dirnames, filenames in os.walk(args.image_directory):
    for filename in filenames:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp', '.webm')):
            image_path = os.path.join(dirpath, filename)
            print(f"Processing file: {image_path}")

            config = load_or_initialize_template(dirpath, args.image_directory, args.default_template)
            
            try:
                with open(os.path.join(dirpath, os.path.splitext(filename)[0]+'.yaml'), 'r', encoding='utf-8') as file:
                    image_meta_yaml = yaml.safe_load(file)
            except FileNotFoundError as e:
                print(f"Warning: Could not find file {e.filename}. Continuing with the next file.")
                continue  # Skip to the next iteration of the loop, effectively ignoring the missing file
            # Check if 'caption' exists and is neither None nor an empty string
            if image_meta_yaml.get('caption') and args.overwrite == False:
                continue
            
            system_prompt_paths = config['llm_config']['system_prompt'] if config['llm_config']['system_prompt'] else []
            default_prompt_path = config['llm_config']['default_prompt']
            expert_list = config['llm_config']['expert_list']
            concept_focus = config['llm_config']['concept_focus']
            
            # template = config['llm_config']['template']
            # template_blank = {key: '' for key in config['llm_config']['template']}
            
            # Initialize an empty string to hold the concatenated contents of all files
            system_prompt = ""

            # Iterate over each path in the system_prompt_paths list
            for path in system_prompt_paths:
                # Construct the full path to the file
                file_path = os.path.join('.', 'prompts', path)
                
                # Open and read the file, then append its content to system_prompt
                with open(file_path, 'r', encoding='utf-8') as file:
                    # If system_prompt is not empty, add a newline for separation before appending more content
                    if system_prompt:
                        system_prompt += "\n"
                    system_prompt += file.read()

            with open(os.path.join('.','prompts',default_prompt_path), 'r', encoding='utf-8') as file:
                chat_prompt = file.read()

            path = image_path
            base64_image = ""
            
            try:
            
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp', '.webm')):
                        
                    with Image.open(image_path) as img:
                        # Convert the image to PNG by saving it to a bytes buffer
                        # This avoids the need to save and read the file from disk
                        buffer = io.BytesIO()
                        img.save(buffer, format="PNG")
                        
                        # Seek to the beginning of the buffer
                        buffer.seek(0)
                        base64_image = base64.b64encode(buffer.read()).decode('utf-8')
                else:
                    image = open(path.replace("'", ""), "rb").read()
                    base64_image = base64.b64encode(image).decode("utf-8")
            except:
                print("Couldn't read the image. Make sure the path is correct and the file exists.")
                continue

            for expert_path in expert_list:
                with open(os.path.join('.','prompts',expert_path), 'r') as file:
                    expert_data = yaml.safe_load(file)
                    
                expert_name = expert_data['expert_name']
                expert_system_prompt = expert_data['expert_system_prompt']
                expert_conversation_prompt = expert_data['expert_conversation_prompt']

                expert_system_append = expert_name + '\n' + expert_system_prompt

                if args.skipconcept:
                    template = expert_system_append + '\TAG_FILE: ' + image_meta_yaml['general']
                else:
                    template = expert_system_append + '\nCONCEPT_FOCUS: ' + concept_focus + '\TAG_FILE: ' + image_meta_yaml['general']

                system_prompt_combined = system_prompt + template
                print('system_prompt_combined')
                print(system_prompt_combined)
                print('chat_prompt')
                print(chat_prompt)
                print('expert_conversation_prompt')
                print(expert_conversation_prompt)
                
                
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
                            {"type": "text", "text": chat_prompt + expert_conversation_prompt},
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
                        presence_penalty=1.2,
                        top_p=0.95
                    )
                    
                    returned_message = completion.choices[0].message.content
                    try_again = False
                    temp_modifier += 0.01
                    
                image_meta_yaml.setdefault('expert_list', {})
                image_meta_yaml['expert_list'][expert_name] = returned_message

                yaml_path = os.path.splitext(image_path)[0] + '.yaml'
                with open(yaml_path, 'w') as yaml_file:
                    yaml.dump(image_meta_yaml, yaml_file, allow_unicode=True, default_flow_style=False, indent=4)
                print(f"Saved combined results to {yaml_path}.")
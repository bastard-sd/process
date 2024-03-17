import os
import numpy as np
from PIL import Image
import shutil
import subprocess
import uuid
import argparse
import yaml
from advcaption.taggers import ImageTagger
import json
from advcaption.nlp import load_model as loadmodel, loadinflectmodel, process_input_tags

# class NumpyEncoder(json.JSONEncoder):
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
    Copies the template yaml file './templates/template.yaml' into the specified image directory.

    Parameters:
    - image_directory: The path to the directory where the template YAML file will be copied.
    """

    if not os.path.isdir(image_directory):
        print(f"The directory {image_directory} does not exist.")
        return {}
    
    if not os.path.isfile(template_file):
        print("Template file './templates/template.yaml' does not exist.")
        return {}
    
    destination_path = os.path.join(image_directory, os.path.basename(template_file))
    
    if os.path.isfile(destination_path):
        print(f"Template file already exists at {destination_path}.")
        try:
            with open(destination_path, 'r') as file:
                template_data = yaml.safe_load(file)
                print("Template YAML loaded into dictionary.")
                return template_data
        except yaml.YAMLError as e:
            print(f"Error loading YAML from template file: {e}")
            return {}
    
    shutil.copy(template_file, destination_path)
    print(f"Template file copied to {destination_path}.")
    try:
        with open(destination_path, 'r') as file:
            template_data = yaml.safe_load(file)
            print("Template YAML loaded into dictionary.")
            return template_data
    except yaml.YAMLError as e:
        print(f"Error loading YAML from template file: {e}")
        return {}

def load_image(image_path, error_directory):
    try:
        image = Image.open(image_path)  # Load the image using PIL
        img = np.array(image, np.uint8)  # Attempt the conversion to numpy array
    except OSError as e:
        print(f"Error: {e}. Moving to error directory.")
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


def rename_files_in_directory(directory, rename=True):
    """
    Recursively renames all image files in the specified directory and its subdirectories.
    Each subdirectory will have its files renamed to a unique pattern followed by an incrementing number.

    Parameters:
    - directory: The root directory containing the files to be renamed.
    """
    if not os.path.isdir(directory):
        print(f"The directory {directory} does not exist.")
        return

    new_files = []
    for dirpath, dirnames, filenames in os.walk(directory):
        # Generate a unique pattern for each subdirectory
        pattern = generate_unique_id(length=5)
        counter = 1

        # Sort files for consistent renaming across runs
        filenames.sort()

        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp', '.webm')):
                if rename:
                    extension = os.path.splitext(filename)[1]
                    new_filename = f"{pattern}_{counter}{extension}"
                    old_file = os.path.join(dirpath, filename)
                    new_file = os.path.join(dirpath, new_filename)
                    os.rename(old_file, new_file)
                    new_files.append(new_file)
                    counter += 1
                else:
                    old_file = os.path.join(dirpath, filename)
                    new_files.append(old_file)
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
parser.add_argument("--error_directory", help="The directory where error images will be moved.")
parser.add_argument("--rename", action="store_true", help="Enable renaming of error images.")
args = parser.parse_args()

# Validate Part 1
for dirpath, dirnames, filenames in os.walk(args.image_directory):
    for filename in filenames:
        # Check if the file extension indicates an image file
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp', '.webm')):
            image_path = os.path.join(dirpath, filename)
            _ = load_image(image_path, args.error_directory)

# Validate Part 2
# validate_images_imagemagick(args.image_directory, args.error_directory)


renamed_filelist = rename_files_in_directory(args.image_directory, rename=args.rename)

template_file = os.path.join(args.image_directory, './templates.yaml')
if os.path.exists(template_file):
    try:
        with open(template_file, 'r', encoding='utf-8') as file:
            template_data = yaml.safe_load(file)
            print("Template YAML loaded into dictionary.")
            config = template_data
    except Exception as e:
        print(f"Failed to load template: {e}")
        config = {}
else:
    config = copy_template_to_image_directory(args.image_directory)

tagger_config = config.get('tagger_config', {})
tag_threshold = tagger_config.get('tag_threshold', 0.35)
character_threshold = tagger_config.get('character_threshold', 0.85)
processor = ImageTagger(tag_threshold=tag_threshold, ratio_threshold=tag_threshold, character_threshold=character_threshold)

spacy_model = loadmodel("en_core_web_lg", 'cuda:0')
inflect_model = loadinflectmodel()

for filename in renamed_filelist:
    print(filename)

    # Check if a corresponding YAML file already exists
    yaml_path = os.path.splitext(filename)[0] + '.yaml'
    if os.path.exists(yaml_path):
        print(f"YAML file already exists for {filename}. Skipping.")
        continue

    # Proceed with processing the image if no YAML file exists
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp', '.webm')):
        image_path = filename
        try:
            combined_results = processor.process_image(image_path)
            # try:
            #     fully_processed_prompt = process_input_tags(spacy_model, inflect_model, [], combined_results['general'])
            #     combined_results['processed'] = fully_processed_prompt
            # except Exception as e:
            #     print(f"Error processing tags for {filename}: {e}. Skipping this step.")
            #     print(f"Moving {filename} to error directory.")
            #     shutil.move(image_path, os.path.join(args.error_directory, os.path.basename(filename)))
            #     # Optionally, you can decide to continue, break, or take any specific action here
            #     continue  # or use 'break' to stop the loop, or 'pass' to do nothing further
            
            # Assuming success, proceed to save the combined_results to a YAML file
            with open(yaml_path, 'w') as yaml_file:
                yaml.dump(combined_results, yaml_file, allow_unicode=True, default_flow_style=False, indent=4)
            print(f"Saved combined results to {yaml_path}.")
        except Exception as e:  # Catching a general exception to handle any kind of failure in process_image
            print(f"Failed to process image {image_path}. Error: {e}")
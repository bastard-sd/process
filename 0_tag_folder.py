import os
import requests
import yaml
from pathlib import Path

# Specify the directory containing the image files
IMAGE_DIR = 'H:/cannabis/leafly'
API_ENDPOINT = 'http://127.0.0.1:9999/tag-image/'

def send_image_and_save_response(image_path: Path, url: str):
    yaml_path = image_path.with_suffix('.yaml')
    if yaml_path.exists():
        print(f"{yaml_path.name} already exists. Skipping reprocessing.")
        return  # Exit the function if the YAML file exists
    
    with open(image_path, 'rb') as image_file:
        # Prepare the 'file' part of the multipart/form-data request
        files = {'file': (image_path.name, image_file, 'image/jpeg')}
        # Make the POST request to the FastAPI endpoint
        response = requests.post(url, files=files)
        
        if response.status_code == 200:
            # Convert the response to YAML format
            response_yaml = yaml.dump(response.json(), allow_unicode=True, sort_keys=False)
            # Define the path for the YAML file (same name as the image, but with .yaml extension)
            yaml_path = image_path.with_suffix('.yaml')
            # Write the YAML data to the file
            with open(yaml_path, 'w') as yaml_file:
                yaml_file.write(response_yaml)
        else:
            print(f"Error with {image_path.name}: {response.text}")

if __name__ == "__main__":
    # Iterate through all files in the directory
    for item in os.listdir(IMAGE_DIR):
        full_path = Path(IMAGE_DIR) / item
        # Check if the current item is a file and has an image extension
        if full_path.is_file() and full_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            print(f"Processing {full_path.name}")
            send_image_and_save_response(full_path, API_ENDPOINT)

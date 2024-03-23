from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import argparse
import yaml
import os
from typing import Dict, Any
from wd14_tagger import ImageTagger # Make sure the path to your module is correct
import os
import json

# Getting environment variables for configuration
DEVICE = os.getenv("DEVICE", "cuda")
WD14_MODEL = os.getenv("WD14_MODEL", "wd14-vit.v3")
WD14_THRESHOLD = float(os.getenv("WD14_THRESHOLD", 0.35))
WD14_REPLACE_UNDERSCORE = os.getenv("WD14_REPLACE_UNDERSCORE", "true") == "true"

app = FastAPI()

# CORS middleware для кросс-оригин запросов
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
current_dir = os.path.dirname(__file__)
with open(os.path.join(current_dir, 'server.yaml'), 'r') as file:
    server_data = yaml.load(file, Loader=yaml.SafeLoader)
    
taggers = {}
for tagger, config in server_data.items():
    if config['activate']:
        taggers[tagger] = ImageTagger(config=config)

@app.post("/tag-image/")
async def tag_image(file: UploadFile = File(...)) -> Dict[str, Any]:
    temp_file_path = f"temp_{file.filename}"
    taggers_results = {}
    taggers_results['tags'] = {}
    taggers_results['tags']['raw'] = {}
    
    with open(temp_file_path, 'wb') as buffer:
        buffer.write(file.file.read())

    for model_name, tagger in taggers.items():
        try:
            print(model_name)
            print(tagger)
            tags_result = tagger.image_interrogate(temp_file_path)
            # tags_str = ", ".join(tags_result.keys())
            taggers_results['tags']['raw'][model_name] = tags_result
        except Exception as e:
            taggers_results['tags']['raw'][model_name] = f"Error: {str(e)}"
            print(e)

    # Initialize master dictionary
    master = {"character": {}, "general": {}, "rating": {}}

    # Function to update master dictionary
    def update_master(category, data):
        for key, value in data.items():
            if key in master[category]:
                master[category][key].append(value)
            else:
                master[category][key] = [value]

    # Iterate through each element and update master dictionary
    for key, element in taggers_results['tags']["raw"].items():
        print(key)
        print(element)
        for category in ["character", "general", "rating"]:
            update_master(category, element.get(category, {}))

    # Average the lists
    for category in master:
        for key in master[category]:
            master[category][key] = sum(master[category][key]) / len(master[category][key])
        # Sort them
        master[category] = dict(sorted(master[category].items(), key=lambda item: item[1], reverse=True))

    taggers_results['tags']['combined'] = master

    os.remove(temp_file_path)
    return taggers_results

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9999)

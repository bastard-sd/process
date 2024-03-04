from __future__ import annotations
import os
import numpy as np
import cv2
from PIL import Image
import onnxruntime as rt
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

NETWORKS_DIR = "./networks"
MOAT_MODEL_REPO = "wd-v1-4-moat-tagger-v2"
SWIN_MODEL_REPO = "wd-v1-4-swinv2-tagger-v2"
CONV_MODEL_REPO = "wd-v1-4-convnextv2-tagger-v2"
VIT_MODEL_REPO = "wd-v1-4-vit-tagger-v2"
MODEL_FILENAME = "model.onnx"
LABEL_FILENAME = "selected_tags.csv"

MODEL_PATHS = {
    'moat': os.path.join(NETWORKS_DIR, MOAT_MODEL_REPO, MODEL_FILENAME),
    'swin': os.path.join(NETWORKS_DIR, SWIN_MODEL_REPO, MODEL_FILENAME),
    'conv': os.path.join(NETWORKS_DIR, CONV_MODEL_REPO, MODEL_FILENAME),
    'vit': os.path.join(NETWORKS_DIR, VIT_MODEL_REPO, MODEL_FILENAME),
}
MODEL_LABEL_PATHS = {
    'moat': os.path.join(NETWORKS_DIR, MOAT_MODEL_REPO, LABEL_FILENAME),
    'swin': os.path.join(NETWORKS_DIR, SWIN_MODEL_REPO, LABEL_FILENAME),
    'conv': os.path.join(NETWORKS_DIR, CONV_MODEL_REPO, LABEL_FILENAME),
    'vit': os.path.join(NETWORKS_DIR, VIT_MODEL_REPO, LABEL_FILENAME),
}

class ImageTagger:
    def __init__(self, tag_threshold=0.35, ratio_threshold=0.1, character_threshold=0.85):
        self.tag_threshold = tag_threshold
        self.ratio_threshold = ratio_threshold
        self.character_threshold = character_threshold
        self.models = {
            "moat": None,
            "swin": None,
            "conv": None,
            "vit": None,
        }
        self.load_all_tagger_models()

    def smart_imread(self, img, flag=cv2.IMREAD_UNCHANGED):
        if img.endswith(".gif"):
            img = Image.open(img)
            img = img.convert("RGB")
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        else:
            img = cv2.imread(img, flag)
        return img

    def smart_24bit(self, img):
        if img.dtype == np.dtype(np.uint16):
            img = (img / 257).astype(np.uint8)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            trans_mask = img[:, :, 3] == 0
            img[trans_mask] = [255, 255, 255, 255]
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def make_square(self, img, target_size):
        old_size = img.shape[:2]
        desired_size = max(old_size)
        desired_size = max(desired_size, target_size)
        delta_w = desired_size - old_size[1]
        delta_h = desired_size - old_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        color = [255, 255, 255]
        new_im = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return new_im

    def smart_resize(self, img, size):
        if img.shape[0] > size:
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        elif img.shape[0] < size:
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
        return img

    def load_tagger_model(self, path: str) -> rt.InferenceSession:
        model = rt.InferenceSession(path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        return model

    def load_all_tagger_models(self):
        for name, path in MODEL_PATHS.items():
            self.models[name] = self.load_tagger_model(path)

    def load_labels(self, path) -> tuple[list[str], list[int], list[int], list[int]]:
        # LABEL_FILENAME = "selected_tags.csv"
        df = pd.read_csv(path)
        tag_names = df["name"].tolist()
        rating_indexes = list(np.where(df["category"] == 9)[0])
        general_indexes = list(np.where(df["category"] == 0)[0])
        character_indexes = list(np.where(df["category"] == 4)[0])
        return tag_names, rating_indexes, general_indexes, character_indexes

    def predict(self, raw_image: Image, image: np.ndarray, model_name: str, path: str):
        model = self.models[model_name]
        input_name = model.get_inputs()[0].name
        label_name = model.get_outputs()[0].name
        tag_names, rating_indexes, general_indexes, character_indexes = self.load_labels(path)

        # Prepare image for model prediction
        _, height, width, _ = model.get_inputs()[0].shape
        img_th = self.make_square(image[:, :, ::-1], height)  # Convert to BGR and make square
        img_th = self.smart_resize(img_th, height)  # Resize according to model's input size
        img_th = img_th.astype(np.float32)
        img_th = np.expand_dims(img_th, 0)  # Add batch dimension

        # Perform prediction
        probs = model.run([label_name], {input_name: img_th})[0]
        labels = list(zip(tag_names, probs[0].astype(float)))

        # Process prediction results
        # rating = {tag_names[i]: probs[0][i] for i in rating_indexes}
        highest_prob_index = max(rating_indexes, key=lambda i: probs[0][i])
        
        general = {tag_names[i]: probs[0][i] for i in general_indexes if probs[0][i] > self.tag_threshold}
        character = {tag_names[i]: probs[0][i] for i in character_indexes if probs[0][i] > self.character_threshold}

        return {
            'model': model_name,
            'rating': tag_names[highest_prob_index],
            'general': general,
            'character': character,
        }

    def combine_dicts(self, results):
        combined_result = {
            'rating': '',
            'general': {},
            'character': {},
        }
        
        for result in results:
            for key in ['general', 'character']:
                for tag, value in result[key].items():
                    if tag in combined_result[key]:
                        combined_result[key][tag] = max(combined_result[key][tag], value)
                    else:
                        combined_result[key][tag] = value
        
        
        a = dict(sorted(combined_result['general'].items(), key=lambda item: item[1], reverse=True))
        combined_result['general'] = (
            ", ".join(list(a.keys()))
            .replace("_", " ")
            .replace("(", "\(")
            .replace(")", "\)")
        )

        b = dict(sorted(combined_result['character'].items(), key=lambda item: item[1], reverse=True))
        combined_result['character'] = (
            ", ".join(list(b.keys()))
            .replace("_", " ")
            .replace("(", "\(")
            .replace(")", "\)")
        )
        combined_result['rating'] = results[0]['rating']
        
        return combined_result

    def process_image(self, img_path):
        raw_image = Image.open(img_path).convert("RGBA")
        new_image = Image.new("RGBA", raw_image.size, "WHITE")  # Convert alpha to white
        new_image.paste(raw_image, mask=raw_image)
        image = np.array(new_image.convert("RGB"))

        result_tags = []
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.predict, raw_image, image, model_name, path)
                for path, model_name in zip([MODEL_LABEL_PATHS['moat'], MODEL_LABEL_PATHS['swin'], MODEL_LABEL_PATHS['conv'], MODEL_LABEL_PATHS['vit']], self.models.keys())
            ]
            for future in as_completed(futures):
                result_tags.append(future.result())

        combined_results = self.combine_dicts(result_tags)
        return combined_results


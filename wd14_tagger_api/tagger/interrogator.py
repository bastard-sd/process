import os
import pandas as pd
import numpy as np

from typing import Tuple, List, Dict
from io import BytesIO
from PIL import Image

from pathlib import Path
from huggingface_hub import hf_hub_download
import re
import json

from numpy import asarray, float32, expand_dims, exp

tag_escape_pattern = re.compile(r'([\\()])')

import tagger.dbimutils as dbimutils

class Interrogator:
    @staticmethod
    def postprocess_tags(
        tags: Dict[str, float],
        threshold=0.35,
        additional_tags: List[str] = [],
        exclude_tags: List[str] = [],
        sort_by_alphabetical_order=False,
        add_confident_as_weight=False,
        replace_underscore=False,
        replace_underscore_excludes: List[str] = [],
        escape_tag=False,
        *args, **kwargs
    ) -> Dict[str, float]:
        for t in additional_tags:
            tags[t] = 1.0

        # those lines are totally not "pythonic" but looks better to me
        tags = {
            t: c

            # sort by tag name or confident
            for t, c in sorted(
                tags.items(),
                key=lambda i: i[0 if sort_by_alphabetical_order else 1],
                reverse=not sort_by_alphabetical_order
            )

            # filter tags
            if (
                c >= threshold
                and t not in exclude_tags
            )
        }

        new_tags = []
        for tag in list(tags):
            new_tag = tag

            if replace_underscore and tag not in replace_underscore_excludes:
                new_tag = new_tag.replace('_', ' ')

            if escape_tag:
                new_tag = tag_escape_pattern.sub(r'\\\1', new_tag)

            if add_confident_as_weight:
                new_tag = f'({new_tag}:{tags[tag]})'

            new_tags.append((new_tag, tags[tag]))
        tags = dict(new_tags)

        return tags

    def __init__(self, name: str) -> None:
        self.name = name

    def load(self):
        raise NotImplementedError()

    def unload(self) -> bool:
        unloaded = False

        if hasattr(self, 'model') and self.model is not None:
            del self.model
            unloaded = True
            print(f'Unloaded {self.name}')

        if hasattr(self, 'tags'):
            del self.tags

        return unloaded

    def interrogate(
        self,
        image: Image,
        device: str
    ) -> Tuple[
        Dict[str, float],  # rating confidents
        Dict[str, float]  # tag confidents
    ]:
        raise NotImplementedError()

class WaifuDiffusionInterrogator(Interrogator):
    def __init__(
        self,
        name,
        model_path,
        onnx_path='model.onnx',
        tags_path='selected_tags.csv',
        use_cpu=True,
        **kwargs
    ) -> None:
        super().__init__(name)
        self.model_path = model_path
        self.tags_path = os.path.join(self.model_path, tags_path)
        self.onnx_path = os.path.join(self.model_path, onnx_path)
        self.kwargs = kwargs
        self.use_cpu = use_cpu

    def download(self) -> Tuple[os.PathLike, os.PathLike]:
        print(f"Loading {self.name} model file from {self.kwargs['repo_id']}")

        model_path = Path(hf_hub_download(
            **self.kwargs, filename=self.model_path))
        tags_path = Path(hf_hub_download(
            **self.kwargs, filename=self.tags_path))
        return model_path, tags_path

    def load(self) -> None:
        # model_path, tags_path = self.download()

        from onnxruntime import InferenceSession

        # https://onnxruntime.ai/docs/execution-providers/
        # https://github.com/toriato/stable-diffusion-webui-wd14-tagger/commit/e4ec460122cf674bbf984df30cdb10b4370c1224#r92654958
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        if self.use_cpu:
            providers.pop(0)

        self.model = InferenceSession(str(self.onnx_path), providers=providers)

        print(f'Loaded {self.name} model from {self.onnx_path}')

        self.tags = pd.read_csv(self.tags_path)

    def load_labels(self, path) -> tuple[list[str], list[int], list[int], list[int]]:
        # LABEL_FILENAME = "selected_tags.csv"
        df = pd.read_csv(path)
        tag_names = df["name"].tolist()
        rating_indexes = list(np.where(df["category"] == 9)[0])
        
        general_indexes = list(np.where(df["category"] == 0)[0])
        species_indexes = list(np.where(df["category"] == 5)[0])
        meta_indexes = list(np.where(df["category"] == 7)[0])
        lore_indexes = list(np.where(df["category"] == 8)[0])
        
        character_indexes = list(np.where(df["category"] == 4)[0])
        copyright_indexes = list(np.where(df["category"] == 3)[0])
        artist_indexes = list(np.where(df["category"] == 1)[0])
        return (tag_names, rating_indexes, general_indexes, species_indexes, meta_indexes, lore_indexes, character_indexes, copyright_indexes, artist_indexes)

    def interrogate(
        self,
        image: Image
    ) -> Tuple[
        Dict[str, float],  # rating confidents
        Dict[str, float]  # tag confidents
    ]:
        # init model
        if not hasattr(self, 'model') or self.model is None:
            self.load()

        # code for converting the image and running the model is taken from the link below
        # thanks, SmilingWolf!
        # https://huggingface.co/spaces/SmilingWolf/wd-v1-4-tags/blob/main/app.py

        # convert an image to fit the model
        _, height, _, _ = self.model.get_inputs()[0].shape

        # alpha to white
        image = image.convert('RGBA')
        new_image = Image.new('RGBA', image.size, 'WHITE')
        new_image.paste(image, mask=image)
        image = new_image.convert('RGB')
        image = np.asarray(image)

        # PIL RGB to OpenCV BGR
        image = image[:, :, ::-1]

        image = dbimutils.make_square(image, height)
        image = dbimutils.smart_resize(image, height)
        image = image.astype(np.float32)
        image = np.expand_dims(image, 0)

        # evaluate model
        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        tag_names, rating_indexes, general_indexes, species_indexes, meta_indexes, lore_indexes, character_indexes, copyright_indexes, artist_indexes = self.load_labels(self.tags_path)
        
        # Perform prediction
        probs = self.model.run([label_name], {input_name: image})[0]
        labels = list(zip(tag_names, probs[0].astype(float)))


        return (probs, tag_names, rating_indexes, general_indexes, species_indexes, meta_indexes, lore_indexes, character_indexes, copyright_indexes, artist_indexes)

class MLDanbooruInterrogator(Interrogator):
    """ Interrogator for the MLDanbooru model. """
    def __init__(
        self,
        name: str,
        model_path: str,
        repo_id: str = '',
        tags_path='classes.json',
        use_cpu = True,
    ) -> None:
        super().__init__(name)
        self.model_path = model_path
        model_directory = os.path.dirname(self.model_path)
        self.tags_path = os.path.join(model_directory, tags_path)
        self.repo_id = repo_id
        self.tags = None
        self.model = None
        self.use_cpu = use_cpu

    def download(self) -> Tuple[str, str]:
        print(f"Loading {self.name} model file from {self.repo_id}")

        model_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=self.model_path
        )
        tags_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=self.tags_path,
        )
        return model_path, tags_path

    def load(self) -> None:
        # model_path, tags_path = self.download()
        from onnxruntime import InferenceSession
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        if self.use_cpu:
            providers.pop(0)
        self.model = InferenceSession(self.model_path, providers=providers)
        print(f'Loaded {self.name} model from {self.model_path}')

        with open(self.tags_path, 'r', encoding='utf-8') as filen:
            self.tags = json.load(filen)

    def interrogate(
        self,
        image: Image
    ) -> Tuple[
        Dict[str, float],  # rating confidents
        Dict[str, float]  # tag confidents
    ]:
        # init model
        if self.model is None:
            self.load()

        image = dbimutils.fill_transparent(image)
        image = dbimutils.resize(image, 448)  # TODO CUSTOMIZE

        x = asarray(image, dtype=float32) / 255
        # HWC -> 1CHW
        x = x.transpose((2, 0, 1))
        x = expand_dims(x, 0)

        input_ = self.model.get_inputs()[0]
        output = self.model.get_outputs()[0]
        # evaluate model
        y, = self.model.run([output.name], {input_.name: x})

        # Softmax
        y = 1 / (1 + exp(-y))

        tags = {tag: float(conf) for tag, conf in zip(self.tags, y.flatten())}
        return tags

    def large_batch_interrogate(self, images: List, dry_run=False) -> str:
        raise NotImplementedError()

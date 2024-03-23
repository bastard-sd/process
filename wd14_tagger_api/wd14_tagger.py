from tagger.interrogator import Interrogator
from PIL import Image
from pathlib import Path

from tagger.interrogators import interrogators

class ImageTagger:
    def __init__(self, config):
        self.threshold = config.get('threshold', 0.35)
        self.device = config.get('device','cpu')
        self.config = config
        print(config['model'])
        self.load_model(config['model'])

    def load_model(self, model_name):
        """
        Loads the selected model.
        """
        if model_name in interrogators.keys():
            self.model_name = model_name
            self.interrogator = interrogators[model_name]
            self.interrogator.use_cpu = self.device == "cpu"
        else:
            raise ValueError(f"Model {model_name} not available.")

    def change_model(self, new_model_name):
        """
        Changes the current model to the new model.
        """
        print(f"Changing model from {self.model_name} to {new_model_name}")
        self.load_model(new_model_name)

    def image_interrogate(self, image_path: Path):
        """
        Performs prediction on an image path.
        """
        im = Image.open(image_path)
        results = self.interrogator.interrogate(im)
        
        if len(results) == 10:
            probs, tag_names, rating_indexes, general_indexes, species_indexes, meta_indexes, lore_indexes, character_indexes, copyright_indexes, artist_indexes = results
            
            tags_with_probs = {tag_names[i]: probs[0][i].astype(float) for i in range(len(tag_names))}
            
            general_tags = {tag_names[i]: tags_with_probs[tag_names[i]] for i in general_indexes}
            general = Interrogator.postprocess_tags(
                general_tags,
                **self.config
            )

            species_tags = {tag_names[i]: tags_with_probs[tag_names[i]] for i in species_indexes}
            species = Interrogator.postprocess_tags(
                species_tags,
                **self.config
            )

            meta_tags = {tag_names[i]: tags_with_probs[tag_names[i]] for i in meta_indexes}
            meta = Interrogator.postprocess_tags(
                meta_tags,
                **self.config
            )

            character_tags = {tag_names[i]: tags_with_probs[tag_names[i]] for i in character_indexes}
            character = Interrogator.postprocess_tags(
                character_tags,
                **self.config
            )

            copyright_tags = {tag_names[i]: tags_with_probs[tag_names[i]] for i in copyright_indexes}
            copyright = Interrogator.postprocess_tags(
                copyright_tags,
                **self.config
            )

            artist_tags = {tag_names[i]: tags_with_probs[tag_names[i]] for i in artist_indexes}
            artist = Interrogator.postprocess_tags(
                artist_tags,
                **self.config
            )


            print(rating_indexes)
            if len(rating_indexes) > 0:
                rating = max(tags_with_probs.items(), key=lambda x: x[1] if x[0] in [tag_names[i] for i in rating_indexes] else 0)
                rating = {rating[0]: rating[1]}
            else:
                rating = {}
            
            combined_general_info = {**general, **species, **meta}
            
            combined_general_info = Interrogator.postprocess_tags(combined_general_info, threshold=self.threshold)
            character = Interrogator.postprocess_tags(character, threshold=0.85)
            copyright = Interrogator.postprocess_tags(copyright, threshold=self.threshold)
            artist = Interrogator.postprocess_tags(artist, threshold=self.threshold)
            
            combined_character_info = {**character, **copyright, **artist}
        else: 
            tags  = results
            combined_general_info = Interrogator.postprocess_tags(
                tags,
                threshold=0.75,
                replace_underscore=True
            )
            rating = {}
            combined_character_info = {}

        return {
            'rating':rating,
            'general':combined_general_info,
            'character':combined_character_info,
        }
        


# TEST
# tags = Interrogator.postprocess_tags(result[1], threshold=self.threshold)
# return tags

# def process_file(self, file_path):
# tags = self.image_interrogate(Path(file_path))
# print("\nDetected Tags:", ", ".join(tags.keys()))

# # Example usage:
# if __name__ == "__main__":
# tagger = ImageTagger(model_name='wd14-vit.v3', threshold=0.35) 
# file_path = "/path/to/your/image.jpg" 
# tagger.process_file(file_path)  

# tagger.change_model('new-model-name') 

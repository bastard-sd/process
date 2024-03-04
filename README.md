```
git clone https://github.com/bastard-sd/process.git
cd process
python -m venv venv
# activate your venv
pip install -r requirements.txt

python -m spacy download en_core_web_lg
cd .\networks\
git clone https://huggingface.co/SmilingWolf/wd-v1-4-moat-tagger-v2
git clone https://huggingface.co/SmilingWolf/wd-v1-4-convnextv2-tagger-v2
git clone https://huggingface.co/SmilingWolf/wd-v1-4-swinv2-tagger-v2
git clone https://huggingface.co/SmilingWolf/wd-v1-4-vit-tagger-v2
```

```
copy 'template_boilerplate.json' into the image directory you are captioning
rename it 'template.json'
customize it if necessary.
```

```
python .\tag_images.py --image_directory=\path\to\images --error_directory=\path\to\errors
python .\tag_images_step_2.py --image_directory=\path\to\images
```
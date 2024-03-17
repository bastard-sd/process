```
git clone https://github.com/bastard-sd/process.git
cd process
python -m venv venv
# activate your venv
pip install -r requirements.txt

python -m spacy download en_core_web_lg
cd .\networks\
git clone https://huggingface.co/SmilingWolf/wd-convnext-tagger-v3 conv
git clone https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3 swin
git clone https://huggingface.co/SmilingWolf/wd-vit-tagger-v3 vit
git clone https://huggingface.co/SmilingWolf/wd-v1-4-moat-tagger-v2 moat
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

# Troubleshooting
If you see `UserWarning: Specified provider 'CUDAExecutionProvider' is not in available provider names.Available providers: 'AzureExecutionProvider, CPUExecutionProvider'`
Do the following
```
pip uninstall onnxruntime onnxruntime-gpu
pip install onnxruntime onnxruntime-gpu
```
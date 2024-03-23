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
mkdir mldb
```

```
cd .\network\mldb\
Invoke-WebRequest -Uri "https://huggingface.co/deepghs/ml-danbooru-onnx/blob/main/ml_caformer_m36_dec-5-97527.onnx" -OutFile ".\ml_caformer_m36_dec-5-97527.onnx"
Invoke-WebRequest -Uri "https://huggingface.co/deepghs/ml-danbooru-onnx/resolve/main/TResnet-D-FLq_ema_6-30000.onnx" -OutFile ".\TResnet-D-FLq_ema_6-30000.onnx"
Invoke-WebRequest -Uri "https://huggingface.co/deepghs/ml-danbooru-onnx/blob/main/classes.json" -OutFile ".\classes.json"
```
or 
```
cd .\network\mldb\
curl -o ml_caformer_m36_dec-5-97527.onnx "https://huggingface.co/deepghs/ml-danbooru-onnx/blob/main/ml_caformer_m36_dec-5-97527.onnx"
curl -o TResnet-D-FLq_ema_6-30000.onnx "https://huggingface.co/deepghs/ml-danbooru-onnx/resolve/main/TResnet-D-FLq_ema_6-30000.onnx"
curl -o classes.json "https://huggingface.co/deepghs/ml-danbooru-onnx/blob/main/classes.json"
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

```
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir

python -m llama_cpp.server --config_file ./llm_api/server.json
```

# Troubleshooting
If you see `UserWarning: Specified provider 'CUDAExecutionProvider' is not in available provider names.Available providers: 'AzureExecutionProvider, CPUExecutionProvider'`
Do the following
```
pip uninstall onnxruntime onnxruntime-gpu
pip install onnxruntime onnxruntime-gpu
```
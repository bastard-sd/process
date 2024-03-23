from typing import List, Dict

from tagger.interrogator import Interrogator, WaifuDiffusionInterrogator, MLDanbooruInterrogator

interrogators: Dict[str, Interrogator] = {
    'wd14-vit.v3': WaifuDiffusionInterrogator(
        'WD14 ViT v3',
        model_path = r"./networks/vit"
        # repo_id='SmilingWolf/wd-vit-tagger-v3',
    ),
    'wd14-convnext.v3': WaifuDiffusionInterrogator(
        'WD14 ConvNeXT v3',
        model_path = r"./networks/conv"
        # repo_id='SmilingWolf/wd-convnext-tagger-v3',
    ),
    'wd14-swin.v3': WaifuDiffusionInterrogator(
        'WD14 Swin v3',
        model_path = r"./networks/swin"
        # repo_id='SmilingWolf/wd-swinv2-tagger-v3',
    ),
    'wd14-moat.v2': WaifuDiffusionInterrogator(
        'WD14 Moat v2',
        model_path = r"./networks/moat"
        # repo_id='SmilingWolf/wd-v1-4-moat-tagger-v2'
    ),
    'wd14-z3d.v2': WaifuDiffusionInterrogator(
        'WD14 Z3D v2',
        model_path = r"./networks/z3d"
    ),
    'mldb-caformer': MLDanbooruInterrogator(
        'ML-Danbooru Caformer dec-5-97527',
        model_path = r"./networks/mldb/ml_caformer_m36_dec-5-97527.onnx",
        # repo_id='deepghs/ml-danbooru-onnx'
    ),
    'mldb-tresnetd': MLDanbooruInterrogator(
        'ML-Danbooru TResNet-D 6-30000',
        model_path = r"./networks/mldb/TResnet-D-FLq_ema_6-30000.onnx",
        # repo_id='deepghs/ml-danbooru-onnx'
    ),
}

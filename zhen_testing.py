
import os
import random

import numpy
from cosyvoice.cli.cosyvoice import CosyVoice
import torchaudio
import torch
from cosyvoice.utils.file_utils import load_wav
cosy_voice = CosyVoice("pretrained_models/CosyVoice-300M")
store_dir = os.path.abspath("randomseed")
prompt_audio_name = "zpl_11"
# store_dir = os.path.abspath("英-多英少中")
# tts_text = "<|zh|>你好，hello，这是一个关于cross-lingual 性能的测试。主题是TTS和 AI，并且中文需要占多数文本。啊"
# tts_text = "<|en|> there will be a very important presentation deadline tomorrow. what a surprise. "
tts_text = "<|zh|>好烦啊，明天有一个很重要的presentation的deadline，我还需要带一个MP3去展示一个TTS模型"
# tts_text = "<|zh|>好烦啊，这是一个新的测试，用于观察随机种子对于音色和预期的影响。在此之前我需要随机来点带感情的文本。"
# tts_text ="<|en|>  there will be a very important presentation deadline tomorrow。 没想到吧，一段中文。what a surprise."
# tts_text = "<|en|>my number is 1539822115."
# tts_text = "<|zh|>你好，你好，这是一个关于跨语种的性能测试。主题是人工智能和语音生成，并且中文需要占多数文本。换了个模板呢。"

    

for j in ["zpl_37","zpl_42","zpl_43"]:
    for seed in [42,123,196]:
        store_path = os.path.join(store_dir,str(seed))
        if not os.path.exists(store_path):
            os.mkdir(store_path)
        random.seed(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
        prompt_audio = load_wav(f"examples/libritts/cosyvoice/zpl_cleandata-denoise/{j }.wav",16000)
        result =cosy_voice.inference_cross_lingual(tts_text=tts_text,prompt_speech_16k=prompt_audio)
        torchaudio.save(os.path.join(store_path,f"{ j}.wav"), result['tts_speech'], 22050)
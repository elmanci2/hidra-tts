
from qwen_tts import Qwen3TTSModel
import torch
import soundfile as sf

class Generate:
    def __init__(self):
        self.model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            device_map="cuda",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )

    def generate(self, text, audio_ref_path, output_path, **kwargs):
        # Default parameters
        params = {
            "language": "Spanish",
            "x_vector_only_mode": True,
            "max_new_tokens": 2048,
            "repetition_penalty": 1.1,
            "temperature": 0.5,
        }
        # Update with provided kwargs
        params.update(kwargs)

        wavs, sr = self.model.generate_voice_clone(
            text=text,
            ref_audio=audio_ref_path,
            **params
        )
        sf.write(output_path, wavs[0], sr)
        return output_path

from qwen_tts import Qwen3TTSModel
import torch
import soundfile as sf

class Generate:
    def __init__(self):
        self.model = Qwen3TTSModel.from_pretrained(
            # "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            device_map="cuda",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

    def generate(self, text, audio_ref_path, output_path, **kwargs):
        # Use defaults that favor high expression and rhythm
        params = {
            "language": "Spanish",
            "x_vector_only_mode": False,  
            "max_new_tokens": 512,
            "repetition_penalty": 1.05,   # Closer to Qwen3 default
            "temperature": 0.9,           # Default for maximizing expression/rhythm
            "top_p": 1.0,
        }
        params.update(kwargs)
        
        # Enforce quality overrides regardless of client request
        params["x_vector_only_mode"] = False

        if audio_ref_path.endswith(".pt"):
            prompt = torch.load(audio_ref_path, weights_only=False)
            wavs, sr = self.model.generate_voice_clone(
                text=text,
                voice_clone_prompt=prompt,
                **params
            )
        else:
            wavs, sr = self.model.generate_voice_clone(
                text=text,
                ref_audio=audio_ref_path,
                **params
            )
        sf.write(output_path, wavs[0], sr)
        return output_path

    def extract_voice(self, audio_ref_path, output_path, ref_text=""):
        # Always force highest quality extraction (ICL mode)
        prompt = self.model.create_voice_clone_prompt(
            ref_audio=audio_ref_path,
            ref_text=ref_text,
            x_vector_only_mode=False
        )
        torch.save(prompt, output_path)
        return output_path

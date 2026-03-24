
from qwen_tts import Qwen3TTSModel
import torch
import soundfile as sf
import io
import gc
import threading
from typing import List, Optional

try:
    import flash_attn  # noqa: F401
    ATTN_IMPL = "flash_attention_2"
    print("usando flash_attention_2")
except ImportError:
    ATTN_IMPL = "sdpa"
    print("usando sdpa")


class ModelManager:
    """Thread-safe singleton model manager with lazy loading."""
    def __init__(self):
        self.current_model_name: Optional[str] = None
        self.model = None
        self._lock = threading.Lock()

    def get_model(self, model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"):
        with self._lock:
            if self.model is None or self.current_model_name != model_name:
                if self.model is not None:
                    del self.model
                    gc.collect()
                    torch.cuda.empty_cache()
                print(f"📦 Loading model: {model_name}")
                self.model = Qwen3TTSModel.from_pretrained(
                    model_name,
                    device_map="cuda",
                    dtype=torch.bfloat16,
                    attn_implementation=ATTN_IMPL,
                )
                self.current_model_name = model_name
                print(f"✅ Model {model_name} loaded successfully.")
            return self.model

    def get_vram_after_model(self) -> float:
        """Returns free VRAM in GB after the model is loaded."""
        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info()
            return free / (1024**3)
        return 0.0


model_manager = ModelManager()


class Generate:
    """Single-item generation (kept for backward compatibility and for the batch engine)."""

    def __init__(self, model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"):
        self.model_name = model_name

    def _resolve_x_vector_mode(self, ref_text: Optional[str]) -> bool:
        """Auto-detect: if ref_text is empty, use x_vector_only_mode."""
        if not ref_text or (isinstance(ref_text, str) and ref_text.strip() == ""):
            return True
        return False

    def generate(self, text: str, audio_ref_path: Optional[str] = None,
                 voice_clone_prompt_bytes: Optional[bytes] = None, **kwargs) -> bytes:
        model = model_manager.get_model(self.model_name)
        params = {
            "language": "Spanish",
            "x_vector_only_mode": False,
            "max_new_tokens": 512,
            "repetition_penalty": 1.07,
            "temperature": 0.9,
            "top_p": 1.0,
        }
        params.update(kwargs)
        params["x_vector_only_mode"] = self._resolve_x_vector_mode(params.get("ref_text"))

        if voice_clone_prompt_bytes is not None:
            prompt = torch.load(io.BytesIO(voice_clone_prompt_bytes), weights_only=False)
            wavs, sr = model.generate_voice_clone(
                text=text, voice_clone_prompt=prompt, **params
            )
        elif audio_ref_path is not None:
            if isinstance(audio_ref_path, str) and audio_ref_path.endswith(".pt"):
                prompt = torch.load(audio_ref_path, weights_only=False)
                wavs, sr = model.generate_voice_clone(
                    text=text, voice_clone_prompt=prompt, **params
                )
            else:
                wavs, sr = model.generate_voice_clone(
                    text=text, ref_audio=audio_ref_path, **params
                )
        else:
            raise ValueError("Must provide either audio_ref_path or voice_clone_prompt_bytes")

        buffer = io.BytesIO()
        sf.write(buffer, wavs[0], sr, format="WAV")
        buffer.seek(0)
        return buffer.read()

    def generate_batch(self, texts: List[str], audio_ref_path: Optional[str] = None,
                       voice_clone_prompt_bytes: Optional[bytes] = None, **kwargs) -> List[bytes]:
        """
        True batched inference: sends N texts into Qwen's generate_voice_clone as a single
        forward pass. The GPU processes all texts simultaneously in one matrix operation.
        Returns a list of WAV bytes, one per text.
        """
        model = model_manager.get_model(self.model_name)
        params = {
            "language": "Spanish",
            "x_vector_only_mode": False,
            "max_new_tokens": 512,
            "repetition_penalty": 1.07,
            "temperature": 0.9,
            "top_p": 1.0,
        }
        params.update(kwargs)
        params["x_vector_only_mode"] = self._resolve_x_vector_mode(params.get("ref_text"))

        if voice_clone_prompt_bytes is not None:
            prompt = torch.load(io.BytesIO(voice_clone_prompt_bytes), weights_only=False)
            wavs, sr = model.generate_voice_clone(
                text=texts, voice_clone_prompt=prompt, **params
            )
        elif audio_ref_path is not None:
            if isinstance(audio_ref_path, str) and audio_ref_path.endswith(".pt"):
                prompt = torch.load(audio_ref_path, weights_only=False)
                wavs, sr = model.generate_voice_clone(
                    text=texts, voice_clone_prompt=prompt, **params
                )
            else:
                wavs, sr = model.generate_voice_clone(
                    text=texts, ref_audio=audio_ref_path, **params
                )
        else:
            raise ValueError("Must provide either audio_ref_path or voice_clone_prompt_bytes")

        results: List[bytes] = []
        for wav in wavs:
            buffer = io.BytesIO()
            sf.write(buffer, wav, sr, format="WAV")
            buffer.seek(0)
            results.append(buffer.read())
        return results

    def extract_voice(self, audio_ref_path: str, ref_text: str = "") -> bytes:
        model = model_manager.get_model(self.model_name)
        x_vector_only = self._resolve_x_vector_mode(ref_text)
        prompt = model.create_voice_clone_prompt(
            ref_audio=audio_ref_path,
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only
        )
        buffer = io.BytesIO()
        torch.save(prompt, buffer)
        buffer.seek(0)
        return buffer.read()
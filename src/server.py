import uvicorn
import time
import torch
from pydantic import BaseModel
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException

# Config & Logic
from src.config.conf import SERVER_HOST, SERVER_PORT
from src.controllers.generate_tts import Generate

app = FastAPI()

# --- Initialize TTS Module ---
# Loading model globally to avoid reloading on every request
try:
    print("🚀 Cargando modelo TTS...")
    tts_module = Generate()
    print("✅ Modelo cargado correctamente.")
except Exception as e:
    print(f"❌ Error loading TTS module: {e}")
    tts_module = None

# --- Pydantic Models ---
class ExtractRequest(BaseModel):
    audio_ref_path: str
    output_path: str
    ref_text: str = ""

class TTSRequest(BaseModel):
    text: str
    audio_ref_path: str
    output_path: str
    ref_text: str = ""
    language: str = "Spanish"
    max_new_tokens: int = 2048
    repetition_penalty: float = 1.1

# --- Routes ---
@app.get("/")
def read_root():
    return {"Hello": "World", "Service": "Hidra-TTS", "Model": "Qwen3-TTS"}

@app.post("/tts/extract_voice")
def extract_voice(request: ExtractRequest):
    if tts_module is None:
        raise HTTPException(status_code=500, detail="TTS Module not initialized")

    try:
        print(f"📦 Extrayendo vector de voz (Calidad Máxima) a: {request.output_path}")
        output_file = tts_module.extract_voice(
            audio_ref_path=request.audio_ref_path,
            output_path=request.output_path,
            ref_text=request.ref_text
        )
        return {
            "status": "success", 
            "output_path": output_file,
            "message": "Voice profile extracted successfully"
        }
    except Exception as e:
        print(f"❌ Error extraction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tts/generate")
def generate_audio(request: TTSRequest):
    if tts_module is None:
        raise HTTPException(status_code=500, detail="TTS Module not initialized")

    try:
        start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        print(f"🎙️ Generando audio para: {request.output_path}")
        
        # Prepare kwargs from request
        gen_kwargs = {
            "language": request.language,
            "max_new_tokens": request.max_new_tokens,
            "repetition_penalty": request.repetition_penalty,
            "ref_text": request.ref_text,
        }

        output_file = tts_module.generate(
            text=request.text,
            audio_ref_path=request.audio_ref_path,
            output_path=request.output_path,
            **gen_kwargs
        )
        
        end_time = time.time()
        duration = round(end_time - start_time, 2)
        
        vram_mb = 0
        if torch.cuda.is_available():
            vram_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)

        return {
            "status": "success", 
            "output_path": output_file,
            "message": "Audio generated successfully",
            "time_seconds": duration,
            "vram_used_mb": vram_mb
        }

    except Exception as e:
        print(f"❌ Error generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def start_server():
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
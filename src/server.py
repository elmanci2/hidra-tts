import uvicorn
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
class TTSRequest(BaseModel):
    text: str
    audio_ref_path: str
    output_path: str
    max_new_tokens: int = 2048
    repetition_penalty: float = 1.1
    temperature: float = 0.5
    x_vector_only_mode: bool = True
    # conf: Optional[Dict[str, Any]] = None # Deprecated in favor of explicit fields

# --- Routes ---
@app.get("/")
def read_root():
    return {"Hello": "World", "Service": "Hidra-TTS", "Model": "Qwen3-TTS"}

@app.post("/tts/generate")
def generate_audio(request: TTSRequest):
    if tts_module is None:
        raise HTTPException(status_code=500, detail="TTS Module not initialized")

    try:
        print(f"🎙️ Generando audio para: {request.output_path}")
        
        # Prepare kwargs from request, filtering out non-gen args
        gen_kwargs = {
            "max_new_tokens": request.max_new_tokens,
            "repetition_penalty": request.repetition_penalty,
            "temperature": request.temperature,
            "x_vector_only_mode": request.x_vector_only_mode
        }

        output_file = tts_module.generate(
            text=request.text,
            audio_ref_path=request.audio_ref_path,
            output_path=request.output_path,
            **gen_kwargs
        )
        
        return {
            "status": "success", 
            "output_path": output_file,
            "message": "Audio generated successfully"
        }

    except Exception as e:
        print(f"❌ Error generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def start_server():
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
import os
# ── Blackwell (sm_120) compatibility ─────────────────────────────────────────
# The @use_kernel_forward_from_hub("RMSNorm") decorator in the Qwen3TTS model
# tries to load a pre-compiled CUDA binary from HuggingFace Hub. That binary
# has no kernel image for Blackwell (sm_120 / RTX 5080+), causing:
#   "CUDA error: no kernel image is available for execution on the device"
# Setting this env var BEFORE any transformers import disables Hub kernels
# entirely and falls back to native PyTorch — safe and correct on all GPUs.
os.environ["TRANSFORMERS_USE_KERNELS"] = "0"
# (El parámetro PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" se ha
# eliminado porque provoca colapso de init de CUDA dentro de RunPod Docker).
# ─────────────────────────────────────────────────────────────────────────────
import uvicorn
import time
import torch
import io
import tempfile
from typing import Optional
import asyncio
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import Response, JSONResponse

# Config & Logic
from src.config.conf import SERVER_HOST, SERVER_PORT
from src.controllers.generate_tts import Generate, model_manager
from src.controllers.batch_engine import BatchEngine

app = FastAPI(
    title="Hidra-TTS API",
    description="Production-grade TTS API with Dynamic Batching",
    version="2.0.0",
)

# --- Initialize TTS Controller ---
try:
    print("🚀 Inicializando controlador TTS...")
    tts_controller = Generate()
    print("✅ Controlador TTS listo.")
except Exception as e:
    print(f"❌ Error loading TTS controller: {e}")
    tts_controller = None

# --- VRAM Calibration ---
MAX_VRAM_GB = 12.0
try:
    if torch.cuda.is_available():
        free_vram, total_vram = torch.cuda.mem_get_info()
        MAX_VRAM_GB = free_vram / (1024**3)
except Exception:
    pass

# El modelo base toma ~3.8 GB en bf16.
AVAILABLE_FOR_TASKS = MAX_VRAM_GB - 3.8
VRAM_PER_ITEM_GB = 0.8  # Cada item en un batch usa ~0.8GB para evitar OOM con textos largos
MAX_BATCH_SIZE = max(1, min(128, int(AVAILABLE_FOR_TASKS / VRAM_PER_ITEM_GB))) if AVAILABLE_FOR_TASKS > 0 else 1

vram_info = f"{round(MAX_VRAM_GB, 2)}GB"
print(f"⚙️ VRAM detectada: {vram_info} -> Max Batch Size dinámico: {MAX_BATCH_SIZE}")

# --- Batch Engine ---
batch_engine: Optional[BatchEngine] = None

@app.on_event("startup")
async def startup_event():
    global batch_engine
    if tts_controller is not None:
        print("🚀 Precargando el modelo TTS...")
        await asyncio.to_thread(model_manager.get_model)
        batch_engine = BatchEngine(
            generator=tts_controller,
            max_batch_size=MAX_BATCH_SIZE,
            max_wait_ms=10000,  # Esperar máximo 10 segundos recolectando items para el batch
            vram_per_item_gb=VRAM_PER_ITEM_GB,
        )
        batch_engine.start()
        print("🚀 Batch Engine iniciado y listo para recibir solicitudes.")

@app.on_event("shutdown")
async def shutdown_event():
    if batch_engine:
        batch_engine.stop()
        print("🛑 Batch Engine detenido.")

# --- Routes ---
@app.get("/")
def read_root():
    stats = {}
    if batch_engine:
        stats = {
            "total_batches": batch_engine.total_batches_processed,
            "total_items": batch_engine.total_items_processed,
            "queue_size": batch_engine.queue.qsize(),
            "max_batch_size": batch_engine._calculate_dynamic_batch_size(),
        }
    return {
        "service": "Hidra-TTS",
        "version": "2.0.0 (Dynamic Batching)",
        "status": "online",
        "vram_detected": vram_info,
        "batch_stats": stats,
    }

@app.post("/tts/extract_voice")
async def extract_voice(
    audio_file: UploadFile = File(...),
    ref_text: str = Form(""),
    model_name: str = Form("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
):
    if tts_controller is None:
        raise HTTPException(status_code=500, detail="TTS Controller not initialized")

    try:
        print(f"📦 Extrayendo vector de voz usando {model_name}...")
        audio_bytes = await audio_file.read()

        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".wav") as tmp_audio:
            tmp_audio.write(audio_bytes)
            tmp_path = tmp_audio.name

        tts_controller.model_name = model_name
        prompt_bytes = await asyncio.to_thread(
            tts_controller.extract_voice,
            audio_ref_path=tmp_path,
            ref_text=ref_text
        )

        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        return Response(
            content=prompt_bytes,
            media_type="application/octet-stream",
            headers={"Content-Disposition": "attachment; filename=voice_prompt.pt"}
        )
    except Exception as e:
        print(f"❌ Error extraction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tts/generate")
async def generate_audio(
    text: str = Form(...),
    audio_file: Optional[UploadFile] = File(None),
    prompt_file: Optional[UploadFile] = File(None),
    ref_text: str = Form(""),
    language: str = Form("Spanish"),
    max_new_tokens: int = Form(2048),
    repetition_penalty: float = Form(1.0),
    temperature: float = Form(0.5),
    top_p: float = Form(0.85),
    model_name: str = Form("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
):
    """
    Generate a single audio. 
    
    With Dynamic Batching enabled, this request is automatically grouped 
    with other concurrent requests into a single GPU batch for maximum throughput.
    """
    if batch_engine is None:
        raise HTTPException(status_code=500, detail="Batch Engine not initialized")

    if not audio_file and not prompt_file:
        raise HTTPException(status_code=400, detail="Debe proporcionar audio_file (.wav/.mp3) o prompt_file (.pt)")

    try:
        tmp_path = None
        voice_clone_prompt_bytes = None

        if prompt_file:
            voice_clone_prompt_bytes = await prompt_file.read()
        elif audio_file:
            audio_bytes = await audio_file.read()
            with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".wav") as tmp_audio:
                tmp_audio.write(audio_bytes)
                tmp_path = tmp_audio.name

        tts_controller.model_name = model_name

        # Ignorar los parámetros enviados por el cliente y forzar calidad extrema
        forced_max_new_tokens = 2048
        forced_repetition_penalty = 1.0
        forced_temperature = 0.5
        forced_top_p = 0.85

        gen_kwargs = {
            "language": language,
            "max_new_tokens": forced_max_new_tokens,
            "repetition_penalty": forced_repetition_penalty,
            "temperature": forced_temperature,
            "top_p": forced_top_p,
            "do_sample": False,
            "subtalker_dosample": False,
            "ref_text": ref_text,
        }

        print("\n" + "="*50)
        print("====== LOGS DE ENDPOINT (SERVER.PY) ======")
        print(f"Texto principal: '{text}'")
        print(f"Texto de referencia (ref_text): '{ref_text}'")
        print(f"¿Tiene .pt enviado?: {bool(voice_clone_prompt_bytes)}")
        print(f"¿Tiene audio directo?: {bool(tmp_path)}")
        print(f"Gen Kwargs:")
        for k,v in gen_kwargs.items():
            print(f"  {k}: {v}")
        print("="*50 + "\n")

        # Submit to the Batch Engine queue. The engine will automatically
        # group this with other concurrent requests into a single GPU batch.
        wav_bytes = await batch_engine.submit(
            text=text,
            audio_ref_path=tmp_path,
            voice_clone_prompt_bytes=voice_clone_prompt_bytes,
            **gen_kwargs
        )

        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=generated_audio.wav"}
        )

    except Exception as e:
        print(f"❌ Error generation: {e}")
        if 'tmp_path' in locals() and tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise HTTPException(status_code=500, detail=str(e))


def start_server():
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
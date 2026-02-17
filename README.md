# Hidra TTS API

API de clonación de voz y texto a voz (TTS) construida con **FastAPI** y el modelo **Qwen3-TTS**.

## Arquitectura

```
main.py                          → Punto de entrada
├── src/
│   ├── server.py                → FastAPI app, rutas y modelos Pydantic
│   ├── config/conf.py           → HOST / PORT
│   └── controllers/
│       └── generate_tts.py      → Lógica de generación (clase Generate)
└── qwen_tts/                    → Módulo Qwen3-TTS (modelo + tokenizer)
    └── inference/
        ├── qwen3_tts_model.py   → Qwen3TTSModel wrapper
        └── qwen3_tts_tokenizer.py
```

## Requisitos

- Python 3.11+
- GPU NVIDIA con CUDA 12.x (recomendado)

## Instalación Local

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate

# Instalar proyecto y dependencias
pip install .
```

## Docker (GPU)

Requiere [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

```bash
# Construir y arrancar
docker compose up --build

# Solo construir
docker compose build
```

El servicio estará en `http://localhost:8000`.

> **Nota:** La primera ejecución descargará el modelo (~3.4 GB). El cache se persiste en un volumen Docker (`hf_cache`) para evitar re-descargas.

## Uso

```bash
python main.py
```

Servidor en `http://0.0.0.0:8000`.

## Endpoints

### `GET /`

Health check. Retorna info del servicio.

### `POST /tts/generate`

Genera audio clonando la voz de una referencia.

**Request Body (JSON):**

| Campo | Tipo | Requerido | Default | Descripción |
| :--- | :--- | :--- | :--- | :--- |
| `text` | string | ✅ | — | Texto a sintetizar |
| `audio_ref_path` | string | ✅ | — | Ruta al audio de referencia |
| `output_path` | string | ✅ | — | Ruta del archivo de salida (.wav) |
| `ref_text` | string | — | `""` | Transcripción del audio de referencia (mejora la calidad en modo ICL) |
| `language` | string | — | `"Spanish"` | Idioma: `"Spanish"`, `"English"`, `"Auto"`, etc. |
| `max_new_tokens` | int | — | `2048` | Máximo de tokens a generar |
| `repetition_penalty` | float | — | `1.1` | Penalización por repetición |
| `temperature` | float | — | `0.5` | Temperatura de muestreo |
| `x_vector_only_mode` | bool | — | `true` | `true`: solo usa embedding del hablante. `false`: modo ICL (usa `ref_text`) |

**Ejemplo:**

```bash
curl -X POST "http://localhost:8000/tts/generate" \
     -H "Content-Type: application/json" \
     -d '{
           "text": "Hola, esto es una prueba de clonación de voz.",
           "audio_ref_path": "/ruta/a/referencia.mp3",
           "output_path": "/ruta/a/resultado.wav"
         }'
```

**Respuesta exitosa:**

```json
{
  "status": "success",
  "output_path": "/ruta/a/resultado.wav",
  "message": "Audio generated successfully"
}
```

**Respuesta de error:**

```json
{
  "detail": "descripción del error"
}
```

## Variables de Entorno

| Variable | Default | Descripción |
| :--- | :--- | :--- |
| `HF_HOME` | `~/.cache/huggingface` | Directorio de cache de modelos HuggingFace |
| `NVIDIA_VISIBLE_DEVICES` | — | GPUs visibles para el contenedor |

## Licencia

Apache-2.0

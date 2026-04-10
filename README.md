# Hidra TTS API v2.0

API de producción para clonación de voz y texto a voz (TTS) construida con **FastAPI**, **Dynamic Batching** y el modelo **Qwen3-TTS**.

Diseñada para ejecutarse remotamente a través de VPN, escalar a miles de audios por hora y exprimir hasta el último CUDA core de tu GPU.

---

## Características Principales

- 🧠 **Dynamic Batching** — Agrupa automáticamente múltiples solicitudes HTTP en un solo forward pass de GPU
- ⚡ **Inferencia Vectorizada** — El modelo Qwen3-TTS procesa N textos simultáneamente como una única matriz de tensores
- 🔄 **Cola Asíncrona** — Las peticiones se encolan sin bloquear, se agrupan y se procesan por lotes
- 📊 **Calibración Dinámica de VRAM** — El tamaño del batch se calcula en tiempo real según la VRAM libre
- 🌐 **VPN Ready** — Los audios y vectores de voz se transfieren como bytes sobre HTTP (sin rutas locales)
- 🛡️ **Fallback Inteligente** — Si un batch falla, se procesa secuencialmente sin perder peticiones
- 🔒 **Thread-Safe Model Manager** — Carga lazy del modelo con `threading.Lock` para evitar race conditions

---

## Arquitectura

```
                    ┌──────────────────┐
                    │   Petición HTTP  │ × N solicitudes simultáneas
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  asyncio.Queue   │  Cola de entrada ilimitada
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Batch Collector │  Recolecta items por 300ms
                    │    (max_wait)    │  o hasta llenar el batch
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  VRAM Calculator │  Mide GPU libre en tiempo real
                    │  Dynamic Sizing  │  y calcula Max Batch Size
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │   GPU Forward    │  UN SOLO forward pass
                    │   Pass (Batch)   │  para todos los textos
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │ asyncio.Future   │  Cada petición HTTP recibe
                    │   Delivery      │  su audio individual
                    └──────────────────┘
```

### Estructura de Archivos

```
main.py                              → Punto de entrada
├── src/
│   ├── server.py                    → FastAPI app, rutas, lifecycle del BatchEngine
│   ├── config/conf.py               → HOST / PORT
│   └── controllers/
│       ├── generate_tts.py          → Lógica de generación (single + batch)
│       └── batch_engine.py          → Cola, Collector, Dynamic Batch Engine
├── qwen_tts/                        → Módulo Qwen3-TTS (modelo + tokenizer)
│   └── inference/
│       ├── qwen3_tts_model.py       → Qwen3TTSModel wrapper (soporta batching nativo)
│       └── qwen3_tts_tokenizer.py
├── Dockerfile                       → Build multi-stage con CUDA 12.1
├── docker-compose.yml               → Orquestación con GPU
└── test_parallel.sh                 → Script de stress test
```

---

## Requisitos

- Python 3.11+
- GPU NVIDIA con CUDA 12.x (mínimo 8GB VRAM, recomendado 12GB+)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (para Docker)

---

## Instalación

### Local

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate

# Instalar proyecto y dependencias
pip install .

# Iniciar servidor
python main.py
```

### Docker (GPU)

```bash
# Construir y arrancar
docker compose up --build

# Solo construir
docker compose build
```

> **Nota:** La primera ejecución descargará el modelo (~3.4 GB). El cache se persiste en un volumen Docker (`hf_cache`) para evitar re-descargas.

---

## Endpoints

### `GET /` — Health Check & Stats

Retorna el estado del servicio y las estadísticas en tiempo real del Batch Engine.

**Respuesta:**

```json
{
  "service": "Hidra-TTS",
  "version": "2.0.0 (Dynamic Batching)",
  "status": "online",
  "vram_detected": "7.82GB",
  "batch_stats": {
    "total_batches": 12,
    "total_items": 87,
    "queue_size": 0,
    "max_batch_size": 8
  }
}
```

---

### `POST /tts/extract_voice` — Extraer perfil de voz

Extrae un perfil de voz (vector `.pt`) de un audio de referencia. Este perfil se usa después para generar audios más rápido sin tener que reenviar el audio original cada vez.

**Request (multipart/form-data):**

| Campo | Tipo | Requerido | Default | Descripción |
| :--- | :--- | :--- | :--- | :--- |
| `audio_file` | file | ✅ | — | Archivo de audio de referencia (.mp3, .wav) |
| `ref_text` | string | — | `""` | Transcripción del audio (mejora calidad en modo ICL) |
| `model_name` | string | — | `"Qwen/Qwen3-TTS-12Hz-1.7B-Base"` | Modelo a utilizar |

**Respuesta:** Archivo binario `.pt` (application/octet-stream)

**Ejemplo:**

```bash
curl -X POST "http://localhost:8001/tts/extract_voice" \
     -H "Content-Type: multipart/form-data" \
     -F "audio_file=@mi_referencia.wav" \
     -F "ref_text=El texto que se dice en el audio de referencia." \
     --output voz_perfil.pt
```

---

### `POST /tts/generate` — Generar audio (con Dynamic Batching)

Genera audio clonando una voz. Cada solicitud se encola automáticamente en el **Batch Engine**, que la agrupa con otras peticiones concurrentes y las procesa todas juntas en un solo pase de GPU.

**Request (multipart/form-data):**

| Campo | Tipo | Requerido | Default | Descripción |
| :--- | :--- | :--- | :--- | :--- |
| `text` | string | ✅ | — | Texto a sintetizar |
| `audio_file` | file | ⚠️ | — | Audio de referencia (.wav/.mp3). Usar esto **O** `prompt_file` |
| `prompt_file` | file | ⚠️ | — | Perfil de voz `.pt` pre-extraído. Usar esto **O** `audio_file` |
| `ref_text` | string | — | `""` | Transcripción del audio de referencia |
| `language` | string | — | `"Spanish"` | Idioma: `"Spanish"`, `"English"`, `"Auto"`, etc. |
| `max_new_tokens` | int | — | `2048` | Máximo de tokens de audio a generar |
| `repetition_penalty` | float | — | `1.1` | Penalización por repetición |
| `model_name` | string | — | `"Qwen/Qwen3-TTS-12Hz-1.7B-Base"` | Modelo a utilizar |

**Respuesta:** Archivo WAV binario (audio/wav)

**Ejemplo con audio de referencia:**

```bash
curl -X POST "http://localhost:8001/tts/generate" \
     -H "Content-Type: multipart/form-data" \
     -F "text=Hola, esto es una prueba de clonación de voz." \
     -F "audio_file=@referencia.mp3" \
     -F "language=Spanish" \
     --output resultado.wav
```

**Ejemplo con perfil de voz (.pt) pre-extraído (más rápido):**

```bash
curl -X POST "http://localhost:8001/tts/generate" \
     -H "Content-Type: multipart/form-data" \
     -F "text=Este audio se genera más rápido usando el vector pre-calculado." \
     -F "prompt_file=@voz_perfil.pt" \
     --output resultado.wav
```

---

## Dynamic Batching — Cómo Funciona

El corazón del rendimiento de Hidra-TTS v2.0 es el sistema de **Inferencia por Lotes Dinámicos**. A diferencia de una API tradicional que procesa las solicitudes una por una (o las paraleliza desperdiciando VRAM), este sistema agrupa múltiples solicitudes en un **único pase de GPU**.

### El Flujo Completo

```
1. Las peticiones HTTP llegan al endpoint /tts/generate
2. Cada petición se convierte en un BatchJob y se inserta en la asyncio.Queue
3. El Batch Collector (loop de fondo) espera a que llegue al menos 1 item
4. Recolecta más items durante un máximo de 300ms (max_wait_ms)
5. Calcula la VRAM libre actual con torch.cuda.mem_get_info()
6. Determina cuántos items caben en el batch (VRAM / 0.5GB por item)
7. Envía TODOS los textos al modelo Qwen3-TTS como List[str]
8. El modelo los procesa en UN SOLO forward pass (una matriz de tensores)
9. Los N audios resultantes se distribuyen a cada petición HTTP vía asyncio.Future
10. Cada cliente recibe su audio individual como respuesta
```

### Parámetros de Calibración

| Parámetro | Valor Default | Ubicación | Descripción |
| :--- | :--- | :--- | :--- |
| `MAX_BATCH_SIZE` | Dinámico (VRAM) | `server.py` | Máximo absoluto de items por batch |
| `max_wait_ms` | `300` | `server.py` | Tiempo máximo de recolección antes de procesar |
| `vram_per_item_gb` | `0.5` | `server.py` | VRAM estimada por cada item en el batch |
| Modelo base | `~3.8 GB` | `server.py` | VRAM que ocupa el modelo cargado en bf16 |

### ¿Por qué es tan rápido?

La clave es que la GPU **no ve N tareas separadas**. Ve **una sola matriz matemática gigante** y utiliza todos sus miles de núcleos CUDA para resolverla en un solo ciclo. Esto significa que generar 8 audios en un batch toma casi el **mismo tiempo** que generar 1 solo audio individualmente.

```
Enfoque tradicional:  8 audios × 5s cada uno = ~40 segundos
Dynamic Batching:     8 audios en 1 batch    = ~6-8 segundos
```

### Fallback de Seguridad

Si la inferencia por lotes falla (por ejemplo, por un OOM inesperado), el sistema automáticamente cae a un modo secuencial donde procesa cada item del batch uno por uno. **Ninguna petición se pierde.**

---

## Gestión de Modelos

El `ModelManager` gestiona la carga y descarga de modelos de forma thread-safe:

- **Lazy Loading:** El modelo no se carga hasta la primera solicitud
- **Thread Lock:** Si múltiples peticiones llegan antes de que el modelo esté listo, solo una lo carga; las demás esperan
- **Hot-Swap:** Si cambias `model_name` en una petición, el manager descarga el modelo anterior, limpia VRAM con `gc.collect()` + `torch.cuda.empty_cache()`, y carga el nuevo
- **Multi-modelo:** Puedes alternar entre distintos modelos Qwen3-TTS sin reiniciar el servidor

---

## Testing

### Stress Test (Peticiones Paralelas)

```bash
# Terminal 1: Iniciar servidor
python main.py

# Terminal 2: Lanzar prueba de estrés
./test_parallel.sh
```

El script dispara 10 peticiones simultáneas. El Batch Engine las agrupará automáticamente y las procesará en lotes. Al final muestra:

- Tiempo total de procesamiento
- Estadísticas del servidor
- Lista de archivos generados

---

## Variables de Entorno

| Variable | Default | Descripción |
| :--- | :--- | :--- |
| `HF_HOME` | `~/.cache/huggingface` | Directorio de cache de modelos HuggingFace |
| `NVIDIA_VISIBLE_DEVICES` | — | GPUs visibles para el contenedor |
| `PYTORCH_CUDA_ALLOC_CONF` | — | Configurar `expandable_segments:True` para reducir fragmentación |

---

## git clone https://github.com/elmanci2/hidra-tts && cd hidra-tts && chmod +x install.sh && ./install.sh

## Licencia

Apache-2.0

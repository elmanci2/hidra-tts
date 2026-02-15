# Hidra TTS API (Qwen3-TTS)

API simple para clonación de voz y texto a voz (TTS) utilizando el modelo **Qwen3-TTS**.

## Instalación

1.  Asegúrate de tener Python 3.9+ instalado.
2.  Instala las dependencias:

    ```bash
    pip install -r requirements.txt
    ```

    **Nota:** Es posible que necesites instalar `torch` con soporte CUDA si tienes una GPU NVIDIA.

## Docker (con soporte GPU)

Para ejecutar el contenedor con acceso a la GPU, asegúrate de tener instalado el [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

1.  Construir y levantar el servicio:

    ```bash
    docker compose up --build
    ```

    El servicio estará disponible en `http://localhost:8000`.

## Uso tradicional
1.  Inicia el servidor:

    ```bash
    python main.py
    ```

    El servidor se iniciará en `http://0.0.0.0:8000`.

## Endpoints

### `POST /tts/generate`

Genera un archivo de audio clonando la voz de un archivo de referencia.

**Cuerpo de la Petición (JSON):**

| Campo | Tipo | Descripción | Opcional | Default |
| :--- | :--- | :--- | :--- | :--- |
| `text` | string | El texto que quieres que diga. | No | - |
| `audio_ref_path` | string | Ruta absoluta al archivo de audio de referencia. | No | - |
| `output_path` | string | Ruta absoluta donde se guardará el audio generado. | No | - |
| `max_new_tokens` | int | Máximo de tokens nuevos a generar. | Sí | 2048 |
| `repetition_penalty` | float | Penalización de repetición. | Sí | 1.1 |
| `temperature` | float | Temperatura de muestreo (creatividad). | Sí | 0.5 |
| `x_vector_only_mode` | bool | Modo solo vector X. | Sí | true |


**Ejemplo con cURL:**

```bash
curl -X POST "http://localhost:8000/tts/generate" \
     -H "Content-Type: application/json" \
     -d '{
           "text": "Hola, esto es una prueba de clonación de voz.",
           "audio_ref_path": "/ruta/a/tu/referencia.mp3",
           "output_path": "/ruta/donde/guardar/resultado.wav"
         }'
```

**Respuesta Exitosa:**

```json
{
  "status": "success",
  "output_path": "/ruta/donde/guardar/resultado.wav",
  "message": "Audio generated successfully"
}
```

#!/bin/bash

# ============================================================
# Test de Estrés con Dynamic Batching para Hidra-TTS API v2.0
# ============================================================
# Asegúrate de que el servidor esté corriendo en otra terminal:
#   python main.py
# ============================================================

URL="http://127.0.0.1:8001/tts/generate"
AUDIO_REF="arthis_kuvs.mp3"
NUM_REQUESTS=10

# Crear carpeta de resultados
OUTPUT_DIR="./tests_resultados"
mkdir -p "$OUTPUT_DIR"

echo "🚀 Hidra-TTS Dynamic Batching Stress Test"
echo "==========================================="
echo "📂 Resultados en: $OUTPUT_DIR"
echo "📡 Endpoint: $URL"
echo "🔢 Solicitudes simultáneas: $NUM_REQUESTS"
echo ""

# Iniciar temporizador
START_TIME=$(date +%s%N)

for i in $(seq 1 $NUM_REQUESTS)
do
  echo "  → Enviando Petición #$i..."
  
  curl -s -X POST "$URL" \
    -H "accept: audio/wav" \
    -H "Content-Type: multipart/form-data" \
    -F "text=Esta es la petición número $i. El sistema de batching dinámico agrupa múltiples solicitudes en un solo pase de GPU para máximo rendimiento." \
    -F "audio_file=@$AUDIO_REF" \
    -F "language=Spanish" \
    -o "$OUTPUT_DIR/test_output_$i.wav" &
done

echo ""
echo "⏳ Todas las peticiones enviadas. Esperando resultados del Batch Engine..."
wait

# Calcular tiempo
END_TIME=$(date +%s%N)
DURATION_S=$(( (END_TIME - START_TIME) / 1000000000 ))
DURATION_MS=$(( (END_TIME - START_TIME) / 1000000 ))

echo ""
echo "==========================================="
echo "✅ ¡Prueba completada!"
echo "⏱️  Tiempo total: ${DURATION_S}s (${DURATION_MS}ms)"
echo "🎵 ${NUM_REQUESTS} audios guardados en $OUTPUT_DIR"
echo ""

# Mostrar stats del servidor
echo "📊 Stats del servidor:"
curl -s http://127.0.0.1:8001/ | python3 -m json.tool 2>/dev/null || echo "(No se pudieron obtener stats)"
echo ""

# Verificar archivos generados
echo "📁 Archivos generados:"
ls -lh "$OUTPUT_DIR"/test_output_*.wav 2>/dev/null || echo "  (sin archivos)"

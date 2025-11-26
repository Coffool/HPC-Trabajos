from flask import Flask, request, jsonify
import math
import logging
import sys
from datetime import datetime

# Configurar logging detallado
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

def euclidean(a, b):
    distance = math.hypot(a['x'] - b['x'], a['y'] - b['y'])
    logger.debug(f"📏 Distancia entre {a} y {b}: {distance}")
    return distance

@app.route('/calculate_distance', methods=['POST'])
def calculate_distance():
    # Log de inicio de request
    start_time = datetime.now()
    logger.info("🚀 📨 INICIANDO REQUEST /calculate_distance")
    print(f"🎯 [FLASK] Request recibido - {datetime.now()}", file=sys.stderr)
    sys.stderr.flush()
    
    try:
        data = request.get_json(force=True)
        logger.info(f"📊 Datos recibidos: {data}")
        
        # Soportar dos formatos comunes: {"cities": [...]} o directamente [...]
        if isinstance(data, dict) and 'cities' in data:
            cities = data['cities']
        else:
            cities = data

        logger.info(f"🏙️ Procesando {len(cities)} ciudades")

        if not isinstance(cities, list) or len(cities) < 2:
            logger.error("❌ Error: lista de ciudades inválida")
            return jsonify({'error': 'Se requiere una lista de al menos 2 ciudades'}), 400

        total = 0.0
        for i in range(len(cities) - 1):
            distance = euclidean(cities[i], cities[i+1])
            total += distance

        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"✅ RESPUESTA ENVIADA: {total} (tiempo: {processing_time:.3f}s)")
        print(f"🎯 [FLASK] Respuesta: {total} - {datetime.now()}", file=sys.stderr)
        sys.stderr.flush()
        
        return jsonify({'total_distance': total})

    except Exception as e:
        logger.error(f"💥 ERROR en calculate_distance: {e}")
        print(f"💥 [FLASK] Error: {e}", file=sys.stderr)
        sys.stderr.flush()
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    logger.info("❤️ Health check recibido")
    return jsonify({'status': 'healthy', 'timestamp': str(datetime.now())})

if __name__ == '__main__':
    logger.info("🚀 INICIANDO SERVIDOR FLASK CON LOGGING")
    app.run(host='0.0.0.0', port=5000, debug=False)
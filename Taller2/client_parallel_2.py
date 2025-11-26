import json
import itertools
import requests
import concurrent.futures
import time
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_URL = "http://192.168.10.15:5000/calculate_distance"

# -----------------------------------------------------------
# LOGGING
# -----------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# -----------------------------------------------------------
# MÉTRICAS GLOBALES
# -----------------------------------------------------------

metrics = {
    "calls_ok": 0,
    "calls_fail": 0,
    "calls_total": 0,
    "permutations": 0,
}

# -----------------------------------------------------------
# SESSION GLOBAL (pool de conexiones)
# -----------------------------------------------------------
session = requests.Session()

retry_strategy = Retry(
    total=5,
    backoff_factor=0.2,
    status_forcelist=[500, 502, 503, 504],
)

adapter = HTTPAdapter(
    max_retries=retry_strategy,
    pool_connections=100,
    pool_maxsize=100
)

session.mount("http://", adapter)
session.mount("https://", adapter)

# -----------------------------------------------------------
# NUEVA FUNCIÓN: Enviar ruta completa a la API
# -----------------------------------------------------------
def calculate_complete_route(path, cities_dict):
    """Envía una ruta COMPLETA a la API en una sola llamada"""
    metrics["calls_total"] += 1

    # Convertir nombres de ciudades a objetos con coordenadas
    cities_with_coords = []
    for city_name in path:
        city = cities_dict[city_name]
        cities_with_coords.append({"x": city["x"], "y": city["y"]})
    
    # Para TSP, agregar retorno al inicio para hacer un ciclo completo
    # Esto calcula: A→B→C→D→A
    cities_with_coords.append(cities_with_coords[0])

    payload = {
        "cities": cities_with_coords  # Enviamos TODA la ruta de una vez
    }

    try:
        start_time = time.time()
        r = session.post(API_URL, json=payload, timeout=10)
        response_time = time.time() - start_time
        
        r.raise_for_status()
        metrics["calls_ok"] += 1
        
        # Log para debugging
        if metrics["calls_total"] % 100 == 0:
            logging.info(f"✅ Ruta completa procesada: {' → '.join(path)} → {path[0]}")
            logging.info(f"   Distancia: {r.json()['total_distance']:.2f}, Tiempo: {response_time:.3f}s")
        
        return r.json()["total_distance"], path, response_time

    except Exception as e:
        metrics["calls_fail"] += 1
        logging.warning(f"❌ Fallo API con ruta {path}: {e}")
        return float('inf'), path, 0

# -----------------------------------------------------------
# Worker MODIFICADO: usa ruta completa
# -----------------------------------------------------------
def worker_task(path, cities_dict):
    """Calcula costo de una ruta COMPLETA en una sola llamada"""
    distance, route, response_time = calculate_complete_route(path, cities_dict)
    return distance, route

# -----------------------------------------------------------
# Generador optimizado: (n-1)! / 2 (igual que antes)
# -----------------------------------------------------------
def generate_reduced_permutations(names):
    first = names[0]
    rest = names[1:]

    for perm in itertools.permutations(rest):
        if perm[0] < perm[-1]:
            yield (first,) + perm

# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------
def main():
    import sys
    file = sys.argv[1]
    workers = int(sys.argv[2])

    with open(file, "r") as f:
        cities_list = json.load(f)

    cities = {c["name"]: c for c in cities_list}
    names = list(cities.keys())

    logging.info("🚀 INICIANDO TSP PARALELO CON RUTAS COMPLETAS")
    logging.info(f"👥 Workers utilizados: {workers}")
    logging.info(f"🏙️ Ciudades: {names}")
    logging.info(f"📊 Permutaciones esperadas: {len(list(generate_reduced_permutations(names)))}")
    t0 = time.time()

    best_cost = float("inf")
    best_path = None

    # Probar una ruta completa primero para verificar
    print("🧪 Probando una ruta completa de ejemplo...")
    test_distance, test_route, _ = calculate_complete_route(names[:4], cities)
    print(f"📊 Ejemplo: {' → '.join(test_route)} → {test_route[0]} = {test_distance:.2f}")

    batch_size = 100  # Batch más pequeño para mejor feedback
    buffer_batch = []
    futures = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for perm in generate_reduced_permutations(names):
            metrics["permutations"] += 1
            buffer_batch.append(perm)

            if len(buffer_batch) >= batch_size:
                logging.info(f"📦 Procesando lote de {len(buffer_batch)} rutas...")
                for p in buffer_batch:
                    futures.append(executor.submit(worker_task, p, cities))
                buffer_batch = []

                # Progress update
                progress = (metrics["permutations"] / len(list(generate_reduced_permutations(names)))) * 100
                logging.info(f"📈 Progreso: {progress:.1f}%")

        # Procesar batch final
        for p in buffer_batch:
            futures.append(executor.submit(worker_task, p, cities))

        logging.info(f"⏳ Esperando {len(futures)} resultados...")

        completed = 0
        best_updates = 0
        for future in concurrent.futures.as_completed(futures):
            cost, path = future.result()
            completed += 1
            
            if cost < best_cost:
                best_cost = cost
                best_path = path
                best_updates += 1
                logging.info(f"🎯 NUEVA MEJOR RUTA #{best_updates}: {cost:.2f}")

            if completed % 100 == 0:
                progress = (completed / len(futures)) * 100
                logging.info(f"📊 Completado: {completed}/{len(futures)} ({progress:.1f}%)")

    t1 = time.time()

    # ---------------------------------------------------------
    # RESULTADOS
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("🎉 MEJOR RUTA ENCONTRADA:")
    print(f"   Ruta: {' → '.join(best_path)} → {best_path[0]}")
    print(f"   Distancia total: {best_cost:.4f}")
    print("\n" + "📊 MÉTRICAS FINALES:")
    print(f"   Tiempo total: {t1 - t0:.4f} s")
    print(f"   Permutaciones procesadas: {metrics['permutations']}")
    print(f"   Llamadas a API:")
    print(f"     ✅ Exitosas: {metrics['calls_ok']}")
    print(f"     ❌ Fallidas: {metrics['calls_fail']}")
    print(f"     📞 Totales:  {metrics['calls_total']}")
    print(f"   Mejores rutas encontradas: {best_updates}")
    print("="*60)

if __name__ == "__main__":
    main()
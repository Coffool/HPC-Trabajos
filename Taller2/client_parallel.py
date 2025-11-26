# client_parallel.py
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
# Llamada a la API
# -----------------------------------------------------------
def dist_api(a, b, cities_dict):
    metrics["calls_total"] += 1

    payload = [
        {"x": cities_dict[a]["x"], "y": cities_dict[a]["y"]},
        {"x": cities_dict[b]["x"], "y": cities_dict[b]["y"]},
    ]

    try:
        r = session.post(API_URL, json=payload, timeout=5)
        r.raise_for_status()
        metrics["calls_ok"] += 1
        return r.json()["total_distance"]

    except Exception as e:
        metrics["calls_fail"] += 1
        logging.warning(f"Fallo API con {a}-{b}: {e}")
        raise


# -----------------------------------------------------------
# Worker: calcula costo de una ruta
# -----------------------------------------------------------
def worker_task(path, cities_dict):
    total = 0

    for i in range(len(path) - 1):
        total += dist_api(path[i], path[i + 1], cities_dict)

    total += dist_api(path[-1], path[0], cities_dict)

    return total, path


# -----------------------------------------------------------
# Generador optimizado: (n-1)! / 2
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

    logging.info("Iniciando TSP paralelo...")
    logging.info(f"Workers utilizados: {workers}")
    t0 = time.time()

    best_cost = float("inf")
    best_path = None

    batch_size = 200
    buffer_batch = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []

        for perm in generate_reduced_permutations(names):
            metrics["permutations"] += 1
            buffer_batch.append(perm)

            if len(buffer_batch) >= batch_size:
                logging.debug(f"Procesando batch de {batch_size}")
                for p in buffer_batch:
                    futures.append(executor.submit(worker_task, p, cities))
                buffer_batch = []

        for p in buffer_batch:
            futures.append(executor.submit(worker_task, p, cities))

        logging.info("Esperando resultados...")

        for f in concurrent.futures.as_completed(futures):
            cost, path = f.result()
            if cost < best_cost:
                best_cost = cost
                best_path = path

    t1 = time.time()

    # ---------------------------------------------------------
    # RESULTADOS
    # ---------------------------------------------------------
    print("\n---------------- RESULTADOS ----------------")
    print("Mejor ruta encontrada:")
    print(" → ".join(best_path))
    print(f"Costo total: {best_cost:.4f}")
    print("\n--------------- MÉTRICAS -------------------")
    print(f"Tiempo total: {t1 - t0:.4f} s")
    print(f"Permutaciones procesadas: {metrics['permutations']}")
    print(f"Tiempo promedio por permutación: {(t1 - t0)/metrics['permutations']:.8f} s")
    print(f"\nLlamadas a API:")
    print(f"  Exitosas: {metrics['calls_ok']}")
    print(f"  Fallidas: {metrics['calls_fail']}")
    print(f"  Totales:  {metrics['calls_total']}")
    print("---------------------------------------------\n")


if __name__ == "__main__":
    main()

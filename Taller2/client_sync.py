# client_sync.py
import json
import itertools
import math
import time
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def load_cities(path):
    with open(path, "r") as f:
        return json.load(f)


def dist(a, b):
    return math.sqrt((a["x"] - b["x"])**2 + (a["y"] - b["y"])**2)


def tsp_cyclic(cities):
    start = cities[0]
    rest = cities[1:]

    best_cost = float("inf")
    best_path = None
    permutations = 0

    for perm in itertools.permutations(rest):
        permutations += 1

        if perm[0]["name"] > perm[-1]["name"]:
            continue

        path = [start] + list(perm) + [start]

        total = 0
        for i in range(len(path) - 1):
            total += dist(path[i], path[i+1])

        if total < best_cost:
            best_cost = total
            best_path = [c["name"] for c in path]

    return best_path, best_cost, permutations


def main():
    if len(sys.argv) < 2:
        print("Uso: python3 client_sync.py cities.json")
        return

    cities = load_cities(sys.argv[1])

    logging.info("Ejecutando TSP secuencial...")
    t0 = time.time()
    best_path, best_cost, perms = tsp_cyclic(cities)
    t1 = time.time()

    print("\n---------------- RESULTADOS ----------------")
    print("Mejor ruta encontrada:")
    print(" -> ".join(best_path))
    print(f"\nDistancia mínima: {best_cost:.4f}")
    print(f"Permutaciones exploradas: {perms}")
    print(f"Tiempo total: {t1 - t0:.4f} segundos")
    print(f"Tiempo promedio por permutación: {(t1 - t0)/perms:.8f} s")
    print("---------------------------------------------\n")


if __name__ == "__main__":
    main()

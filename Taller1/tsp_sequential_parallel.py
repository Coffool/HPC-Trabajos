import math
import itertools
import multiprocessing as mp
import time
import matplotlib.pyplot as plt
import psutil
import numpy as np
import os


def distancia_entre_ciudades(ciudad1, ciudad2):
    """ Calcula la distancia euclidiana entre dos ciudades
    
        Args:
            ciudad1: (tupla) (x1, y1)
            ciudad2: (tupla) (x2, y2)

        Returns:
            (float) dstancia entre ciudad1 y ciudad2

    """

    x1, y1 = ciudad1
    x2, y2 = ciudad2

    # Forluma: d = sqrt((x2 - x1)^2 + (y2 - y1)^2)

    distancia = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    return distancia 

def distancia_total_ruta(ciudades, ruta_indices):
    """
        Calcula la distancia total de una ruta completa

        Args: 
            ciudades: Lissta de todas las ciudades (tuplas de coordenadas)
            rutaa_indices: lista de indices que muestran el orden de visita de als ciudades hasta llegar a la ciudad inicial
        
        Returns:
            (float) distancia total de la ruta
    """
    distancia_total = 0.0
    num_ciudades = len(ruta_indices)

    for i in range(num_ciudades - 1):
        ciudad_actual = ciudades[ruta_indices[i]]
        ciudad_siguiente = ciudades[ruta_indices[i + 1]]

        distancia_total += distancia_entre_ciudades(ciudad_actual, ciudad_siguiente)
    
    return distancia_total

def generar_permutaciones_tsp(num_ciudades):
    """
        Genera todas las permutaicones posibles para el problema del TSP

        Args:
            num_ciudades: (int) número de ciudades

        Returns:
            (list) lista de permutaciones posibles

    """

    if num_ciudades <= 2:
        return [[]] if num_ciudades == 2 else []
    
    # Generamos permutaciones de las ciudaddes 1, 2, ..., n-1
    ciudades_restantes = list(range(1, num_ciudades))
    permutaciones = list(itertools.permutations(ciudades_restantes))

    print(f"Para {num_ciudades}:")
    print(f" - Ciudades restantes a permutar: {ciudades_restantes}")
    print(f" - Número de permutaciones generadas: {len(permutaciones)}")

    return permutaciones

def generar_permutaciones_unicas(n):
    """
    Genera solo permutaciones únicas para TSP (sin rutas duplicadas)
    Para n ciudades, genera (n-1)! / 2 permutaciones
    """
    if n <= 2:
        return [[]] if n == 2 else []
    
    ciudades_restantes = list(range(1, n))
    todas_permutaciones = list(itertools.permutations(ciudades_restantes))
    
    # Filtrar para eliminar rutas duplicadas (inversas)
    permutaciones_unicas = []
    for perm in todas_permutaciones:
        # Una permutación y su inversa representan la misma ruta
        # Tomamos solo aquellas donde la primera ciudad es menor que la última
        if perm[0] < perm[-1]:
            permutaciones_unicas.append(perm)
    
    print(f"🔢 Permutaciones originales: {len(todas_permutaciones)}")
    print(f"🎯 Permutaciones únicas: {len(permutaciones_unicas)}")
    print(f"📉 Reducción: {len(todas_permutaciones) - len(permutaciones_unicas)} permutaciones eliminadas")
    
    return permutaciones_unicas
       
def tsp_secuencial(ciudades):
    """
        Resuelve el problema del TSP de forma seceuncial

        Args: 
            ciudades: lista de todas las ciudades (tuplas de coordenadas)
        
        Returns:
            (tuple) mejor ruta y su distancia total
    """

    n = len(ciudades)

    # Definir casos especiales
    if n == 0:
        return ([], 0.0)
    elif n == 1:
        return (ciudades, 0.0)
    elif n == 2:
        # Solo hay una ruta posible
        ruta = [ciudades[0], ciudades[1], ciudades[0]]
        distancia = 2 * distancia_entre_ciudades(ciudades[0], ciudades[1])

        return (ruta, distancia)
    
    # Para n>= 3
    mejor_ruta = None
    mejor_distancia = float('inf')

    print("Gemerando permutaciones...")
    permutaciones = generar_permutaciones_unicas(n)

    print("Generando permutaciones...")
    for perm in permutaciones:
        # Construir la ruta completa
        ruta_indices = [0] + list(perm) + [0]

        # Calcular distancia
        distancia_actual = distancia_total_ruta(ciudades, ruta_indices)

        # Actualizamos si encontramos mejor ruta
        if distancia_actual < mejor_distancia:
            mejor_distancia = distancia_actual
            mejor_ruta = [ciudades[i] for i in ruta_indices]
    
    return (mejor_ruta, mejor_distancia)

def procesar_chunk_tsp(args):
    """
        Procesa un chunk de permutaiones y encuentra la mejor ruta en ese chunk

        Args:
            args: tupla que contiene (ciudades, permutaciones_chunk)
                  - ciudades: lista de todas las ciudades (tuplas de coordenadas)
                  - permutaciones_chunk: lista de permutaciones a procesar en este chunk
        Returns:
            (tuple) mejor ruta local y su distancia total en este chunk
        
    """

    ciudades, chunk_permutacones = args
    mejor_distancia_local = float('inf')
    mejor_ruta_local = None

    for perm in chunk_permutacones:
        # Construir la ruta completa: empezar y terminar en ciudad 0
        ruta_indices = [0] + list(perm) + [0]

        # Calcular distancia de esa ruta
        distancia_actual = distancia_total_ruta(ciudades, ruta_indices)

        # Actualizar mejor ruta local si es necesario
        if distancia_actual < mejor_distancia_local:
            mejor_distancia_local = distancia_actual
            mejor_ruta_local = [ciudades[i] for i in ruta_indices]
    
    return (mejor_ruta_local, mejor_distancia_local)

def dividir_permutaciones_en_chunks(permutaciones, num_chunks):
    """
        Divide la lista de permutaciones en chunks para procesamiento en paralelo

        Args:
            permutaciones: lista de todas las permutaciones
            num_chunks: número de chunks a dividir
        
        Returns:
            (list) lista de chunks (cada chunk es una lista de permutaciones)

    """

    # Calcular tamaño aproximado de cada chunk 
    total_permutaciones = len(permutaciones)
    tamano_chunk = total_permutaciones // num_chunks

    chunks = []
    for i in range(num_chunks):
        inicio = i * tamano_chunk
        # Para el ultimo chin aseguramos tomar hasta el final
        fin = inicio + tamano_chunk if i < num_chunks - 1 else total_permutaciones
        chunks.append(permutaciones[inicio:fin])

    return chunks

def tsp_paralelo(ciudades, num_procesos=None):
    """
        Resuelve el problema del TSP de forma paraleal usando multiprocessing

        Args: 
            ciudades: lista de todas las ciudades (tuplas de coordenadas)
            num_procesos: número de procesos a usar (si es None, usa el número de núcleos disponibles)

        Returns:
            (tuple) mejor ruta y su distancia total
    """

    n = len(ciudades)

    # casos especiales (igual que en tsp secueencial)
    if n == 0:
        return ([], 0.0)
    elif n == 1:
        return (ciudades, 0.0)
    elif n == 2:
        ruta = [ciudades[0], ciudades[1], ciudades[0]]
        distancia = 2 * distancia_entre_ciudades(ciudades[0], ciudades[1])

        return (ruta, distancia)
    
    # Si no se especifica, usar todos los núcleos disponibles
    if num_procesos is None:
        num_procesos = mp.cpu.count()

    print(f"Usando {num_procesos} procesos para resolver el TSP en paralelo.")
    print("Generando permutaciones...")

    permutaciones = generar_permutaciones_unicas(n)

    print(f"Número total de permutaciones: {len(permutaciones)}")

    # Dividir permutaciones en chunks
    chunks_permutaciones = dividir_permutaciones_en_chunks(permutaciones, num_procesos)

    print(f"Dividido en {len(chunks_permutaciones)} chunks para procesamiento paralelo.")

    # Preparar argumentos para cada proceso
    argumentos_procesos = [(ciudades, chunk) for chunk in chunks_permutaciones]

    # Crear pool de procesos
    with mp.Pool(processes=num_procesos) as pool:
        resultados = pool.map(procesar_chunk_tsp, argumentos_procesos)

    print("Combinando resultados de todos los procesos...")

    # Encntrar la mejor ruta entre todos los resultados
    mejor_ruta_global = None
    mejor_distancia_global = float('inf')

    for ruta_local, distancia_local in resultados:
        if distancia_local < mejor_distancia_global:
            mejor_distancia_global = distancia_local
            mejor_ruta_global = ruta_local

    return (mejor_ruta_global, mejor_distancia_global)
  
def medir_tiempo_y_recursos(funcion, *args, **kwargs):
    """
    Mide tiempo de ejecución y uso de recursos de una función - VERSIÓN CORREGIDA
    """
    try:
        proceso = psutil.Process(os.getpid())
        
        # Medir recursos antes (tomar muestra inicial)
        cpu_antes = proceso.cpu_percent()
        memoria_antes = proceso.memory_info().rss / 1024 / 1024  # MB
        
        # Forzar una medición de CPU
        time.sleep(0.1)
        cpu_antes = proceso.cpu_percent()
        
        # Medir tiempo de ejecución
        inicio = time.time()
        resultado = funcion(*args, **kwargs)
        tiempo_ejecucion = time.time() - inicio
        
        # Medir recursos después
        time.sleep(0.1)  # Pequeña pausa para medición estable
        cpu_despues = proceso.cpu_percent()
        memoria_despues = proceso.memory_info().rss / 1024 / 1024
        
        return {
            'resultado': resultado,
            'tiempo': tiempo_ejecucion,
            'cpu_uso': max(cpu_antes, cpu_despues, 0.1),
            'memoria_uso': max(memoria_despues - memoria_antes, 0),
            'memoria_pico': memoria_despues
        }
    
    except Exception as e:
        print(f"❌ Error en medir_tiempo_y_recursos: {e}")
        return {
            'resultado': None,
            'tiempo': 0.0,
            'cpu_uso': 0.0,
            'memoria_uso': 0.0,
            'memoria_pico': 0.0
        }

def comparar_algoritmos_tsp(ciudades_lista, num_procesos=4):
    """
    Compara exhaustivamente los algoritmos secuencial y paralelo - VERSIÓN CORREGIDA
    """
    resultados = []
    
    for i, ciudades in enumerate(ciudades_lista):
        n = len(ciudades)
        print(f"\n{'='*60}")
        print(f"🔬 COMPARACIÓN PARA {n} CIUDADES (Prueba {i+1}/{len(ciudades_lista)})")
        print(f"{'='*60}")
        
        try:
            # Ejecutar algoritmo secuencial
            print("🔄 Ejecutando algoritmo secuencial...")
            metricas_seq = medir_tiempo_y_recursos(tsp_secuencial, ciudades)
            
            if metricas_seq['resultado'] is None:
                print("❌ Error en algoritmo secuencial, saltando...")
                continue
                
            ruta_seq, dist_seq = metricas_seq['resultado']
            
            # Ejecutar algoritmo paralelo
            print("🔄 Ejecutando algoritmo paralelo...")
            metricas_par = medir_tiempo_y_recursos(tsp_paralelo, ciudades, num_procesos)
            
            if metricas_par['resultado'] is None:
                print("❌ Error en algoritmo paralelo, saltando...")
                continue
                
            ruta_par, dist_par = metricas_par['resultado']
            
            # Verificar que tenemos tiempos válidos
            if 'tiempo' not in metricas_seq or 'tiempo' not in metricas_par:
                print("❌ Error: No se pudieron medir los tiempos correctamente")
                continue
            
            # Calcular métricas de rendimiento
            tiempo_seq = metricas_seq['tiempo']
            tiempo_par = metricas_par['tiempo']
            
            if tiempo_par > 0:
                speedup = tiempo_seq / tiempo_par
            else:
                speedup = 0.0
                
            eficiencia = (speedup / num_procesos) * 100 if num_procesos > 0 else 0
            
            # Recolectar resultados
            resultado_prueba = {
                'n_ciudades': n,
                'n_prueba': i + 1,
                'secuencial': {
                    'tiempo': tiempo_seq,
                    'cpu': metricas_seq.get('cpu_uso', 0),
                    'memoria': metricas_seq.get('memoria_uso', 0),
                    'memoria_pico': metricas_seq.get('memoria_pico', 0),
                    'distancia': dist_seq
                },
                'paralelo': {
                    'tiempo': tiempo_par,
                    'cpu': metricas_par.get('cpu_uso', 0),
                    'memoria': metricas_par.get('memoria_uso', 0),
                    'memoria_pico': metricas_par.get('memoria_pico', 0),
                    'distancia': dist_par
                },
                'speedup': speedup,
                'eficiencia': eficiencia
            }
            
            resultados.append(resultado_prueba)
            
            # Mostrar resultados
            print(f"\n📊 RESULTADOS PARA {n} CIUDADES:")
            print(f"⏱️  TIEMPO:")
            print(f"   Secuencial: {tiempo_seq:.4f} segundos")
            print(f"   Paralelo:   {tiempo_par:.4f} segundos")
            print(f"   Speedup:    {speedup:.2f}x")
            
            print(f"💾 RECURSOS:")
            print(f"   CPU uso - Secuencial: {metricas_seq.get('cpu_uso', 0):.1f}%")
            print(f"   CPU uso - Paralelo:   {metricas_par.get('cpu_uso', 0):.1f}%")
            
            print(f"📈 EFICIENCIA: {eficiencia:.1f}%")
            
        except Exception as e:
            print(f"❌ Error durante la comparación: {e}")
            continue
    
    return resultados

def generar_conjuntos_prueba():
    """
    Genera diferentes conjuntos de ciudades para pruebas
    """
    conjuntos = []
    
    # Conjunto pequeño (4 ciudades)
    conjuntos.append([
        (0, 0), (2, 3), (5, 1), (3, 6)
    ])
    
    # Conjunto mediano (6 ciudades)
    conjuntos.append([
        (0, 0), (2, 3), (5, 1), (3, 6), (7, 4), (1, 8)
    ])
    
    # Conjunto grande (8 ciudades) - ¡Cuidado! Esto puede ser lento
    conjuntos.append([
        (0, 0), (2, 3), (5, 1), (3, 6), (7, 4), (1, 8), (6, 7), (4, 2)
    ])


    
    # Conjunto de ejemplo del mundo real (coordenadas de ciudades)
    conjuntos.append([
        (0, 0),    # Origen
        (3, 7),    # Ciudad A
        (8, 2),    # Ciudad B  
        (12, 5),   # Ciudad C
        (6, 9),    # Ciudad D
        (10, 1)    # Ciudad E
    ])

    # Conjunto extra grande (10 ciudades) - ¡Cuidado! Muy lento
    conjuntos.append([
        (0, 0), (2, 3), (5, 1), (3, 6), (7, 4), (1, 8), (6, 7), (4, 2), (9, 5), (8, 9)
    ])  
    
    return conjuntos

def visualizar_resultados(resultados):
    """
    Crea gráficos completos de los resultados
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Preparar datos
    n_ciudades = [r['n_ciudades'] for r in resultados]
    tiempos_seq = [r['secuencial']['tiempo'] for r in resultados]
    tiempos_par = [r['paralelo']['tiempo'] for r in resultados]
    speedups = [r['speedup'] for r in resultados]
    eficiencias = [r['eficiencia'] for r in resultados]
    memoria_seq = [r['secuencial']['memoria_pico'] for r in resultados]
    memoria_par = [r['paralelo']['memoria_pico'] for r in resultados]
    
    # Gráfico 1: Tiempos de ejecución
    x_pos = np.arange(len(n_ciudades))
    ancho = 0.35
    
    ax1.bar(x_pos - ancho/2, tiempos_seq, ancho, label='Secuencial', alpha=0.7)
    ax1.bar(x_pos + ancho/2, tiempos_par, ancho, label='Paralelo', alpha=0.7)
    ax1.set_xlabel('Número de Ciudades')
    ax1.set_ylabel('Tiempo (segundos)')
    ax1.set_title('Comparación de Tiempos de Ejecución')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(n_ciudades)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Añadir valores en las barras
    for i, v in enumerate(tiempos_seq):
        ax1.text(i - ancho/2, v + 0.01, f'{v:.2f}s', ha='center', va='bottom', fontsize=8)
    for i, v in enumerate(tiempos_par):
        ax1.text(i + ancho/2, v + 0.01, f'{v:.2f}s', ha='center', va='bottom', fontsize=8)
    
    # Gráfico 2: Speedup
    ax2.plot(n_ciudades, speedups, 'o-', linewidth=2, markersize=8, color='green')
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Límite neutral')
    ax2.set_xlabel('Número de Ciudades')
    ax2.set_ylabel('Speedup (Secuencial/Paralelo)')
    ax2.set_title('Speedup del Algoritmo Paralelo')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Añadir valores en los puntos
    for i, v in enumerate(speedups):
        ax2.annotate(f'{v:.2f}x', (n_ciudades[i], v), 
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    
    # Gráfico 3: Eficiencia
    ax3.bar(n_ciudades, eficiencias, color='orange', alpha=0.7)
    ax3.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Eficiencia ideal (100%)')
    ax3.set_xlabel('Número de Ciudades')
    ax3.set_ylabel('Eficiencia (%)')
    ax3.set_title('Eficiencia del Paralelismo')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Añadir valores en las barras
    for i, v in enumerate(eficiencias):
        ax3.text(n_ciudades[i], v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Gráfico 4: Uso de Memoria
    ax4.bar(x_pos - ancho/2, memoria_seq, ancho, label='Secuencial', alpha=0.7)
    ax4.bar(x_pos + ancho/2, memoria_par, ancho, label='Paralelo', alpha=0.7)
    ax4.set_xlabel('Número de Ciudades')
    ax4.set_ylabel('Memoria (MB)')
    ax4.set_title('Uso de Memoria en Pico')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(n_ciudades)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def generar_reporte_textual(resultados):
    """
    Genera un reporte textual detallado de los resultados
    """
    print("\n" + "="*80)
    print("📋 INFORME FINAL DE COMPARACIÓN TSP")
    print("="*80)
    
    for resultado in resultados:
        n = resultado['n_ciudades']
        seq = resultado['secuencial']
        par = resultado['paralelo']
        
        print(f"\n🏙️  PARA {n} CIUDADES:")
        print(f"   ⏱️  TIEMPOS:")
        print(f"      • Secuencial: {seq['tiempo']:.4f} segundos")
        print(f"      • Paralelo:   {par['tiempo']:.4f} segundos")
        print(f"      • Speedup:    {resultado['speedup']:.2f}x")
        
        print(f"   💾 RECURSOS:")
        print(f"      • CPU Secuencial: {seq['cpu']:.1f}%")
        print(f"      • CPU Paralelo:   {par['cpu']:.1f}%")
        print(f"      • Memoria Secuencial: {seq['memoria_pico']:.2f} MB")
        print(f"      • Memoria Paralelo:   {par['memoria_pico']:.2f} MB")
        
        print(f"   📈 EFICIENCIA: {resultado['eficiencia']:.1f}%")
        
        # Recomendación
        if resultado['speedup'] > 1.2:
            print("   ✅ RECOMENDACIÓN: Usar algoritmo paralelo")
        elif resultado['speedup'] < 0.8:
            print("   ⚠️  RECOMENDACIÓN: Usar algoritmo secuencial (menos overhead)")
        else:
            print("   🔄 RECOMENDACIÓN: Similar rendimiento, usar según disponibilidad de recursos")
    
    # Resumen general
    speedup_promedio = np.mean([r['speedup'] for r in resultados])
    eficiencia_promedio = np.mean([r['eficiencia'] for r in resultados])
    
    print(f"\n🎯 RESUMEN GENERAL:")
    print(f"   Speedup promedio: {speedup_promedio:.2f}x")
    print(f"   Eficiencia promedio: {eficiencia_promedio:.1f}%")
    
    if speedup_promedio > 1:
        print(f"   ✅ El paralelismo proporciona una mejora de {speedup_promedio:.2f}x en promedio")
    else:
        print("   ⚠️  El paralelismo no muestra mejoras significativas en estos casos de prueba")

def mostrar_resultado_completo(ciudades, ruta, distancia, titulo="Resltado TSP"):
    """
    Muestra los resultado de forma detallada y clara
    """

    print(f"\n{'='*80}")
    print(f"🎯 {titulo}")
    print(f"{'='*80}")

    # Mostrar todas sus ciudades con sus indices
    print("\n📍 Ciudades (con índices):")
    for i, ciudad in enumerate(ciudades):
        print(f"  • Ciudad {i}: {ciudad}")

    # Mostrar la ruta óptima con detalles
    print(f"\n🛣️  RUTA ÓPTIMA ENCONTRADA:")
    print(f"   Distancia total: {distancia:.4f} unidades")
    print(f"   Número de paradas: {len(ruta)}")
    print(f"   Tipo de ruta: {'Circular' if ruta[0] == ruta[-1] else 'Lineal'}")
    
    # Mostrar el recorrido paso a paso
    print(f"\n📋 RECORRIDO DETALLADO:")
    for i in range(len(ruta) - 1):
        ciudad_actual = ruta[i]
        ciudad_siguiente = ruta[i + 1]
        
        # Encontrar índices de las ciudades en la lista original
        idx_actual = ciudades.index(ciudad_actual)
        idx_siguiente = ciudades.index(ciudad_siguiente)
        
        dist_segmento = distancia_entre_ciudades(ciudad_actual, ciudad_siguiente)
        
        print(f"   {i+1:2d}. Ciudad {idx_actual:2d} {ciudad_actual} → "
              f"Ciudad {idx_siguiente:2d} {ciudad_siguiente} : {dist_segmento:.4f}")
    
    # Resumen final
    print(f"\n📊 RESUMEN EJECUTIVO:")
    print(f"   • Total de ciudades visitadas: {len(set(ruta))}")
    print(f"   • Distancia promedio por segmento: {distancia/(len(ruta)-1):.4f}")
    print(f"   • Ruta {'SÍ' if ruta[0] == ruta[-1] else 'NO'} es circular")

def visualizar_ruta(ciudades, ruta, distancia, titulo="Ruta Óptima TSP", mostrar_indices=True):
    """
    Crea una visualización gráfica de la ruta óptima
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Extraer coordenadas
    x = [ciudad[0] for ciudad in ciudades]
    y = [ciudad[1] for ciudad in ciudades]
    
    # Coordenadas de la ruta (en términos de índices)
    ruta_coords = [ciudades.index(ciudad) for ciudad in ruta]
    ruta_x = [ciudades[i][0] for i in ruta_coords]
    ruta_y = [ciudades[i][1] for i in ruta_coords]
    
    # Dibujar la ruta
    ax.plot(ruta_x, ruta_y, 'o-', linewidth=2, markersize=8, 
            color='blue', alpha=0.7, label='Ruta óptima')
    
    # Dibujar puntos de ciudades
    ax.scatter(x, y, s=100, color='red', alpha=0.8, zorder=5, label='Ciudades')
    
    # Etiquetar ciudades
    if mostrar_indices:
        for i, (xi, yi) in enumerate(ciudades):
            ax.annotate(f'Ciudad {i}\n({xi},{yi})', (xi, yi), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # Dibujar flechas para mostrar dirección (cada 2 segmentos para no saturar)
    for i in range(0, len(ruta_x)-1, 2):
        ax.annotate('', xy=(ruta_x[i+1], ruta_y[i+1]), 
                   xytext=(ruta_x[i], ruta_y[i]),
                   arrowprops=dict(arrowstyle='->', color='green', lw=1.5, alpha=0.7))
    
    # Configuración del gráfico
    ax.set_xlabel('Coordenada X')
    ax.set_ylabel('Coordenada Y')
    ax.set_title(f'{titulo}\nDistancia total: {distancia:.4f}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Asegurar misma escala en ejes
    ax.set_aspect('equal', adjustable='datalim')
    
    # Añadir cuadro de información
    info_text = f"Ciudades: {len(ciudades)}\nDistancia: {distancia:.4f}\nSegmentos: {len(ruta)-1}"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

    return fig


def visualizar_comparacion(ciudades, resultado_seq, resultado_par):
    """
    Compara visualmente los resultados secuencial y paralelo
    """
    ruta_seq, dist_seq = resultado_seq
    ruta_par, dist_par = resultado_par
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Función auxiliar para dibujar una ruta
    def dibujar_ruta(ax, ciudades, ruta, distancia, titulo, color):
        ruta_coords = [ciudades.index(ciudad) for ciudad in ruta]
        ruta_x = [ciudades[i][0] for i in ruta_coords]
        ruta_y = [ciudades[i][1] for i in ruta_coords]
        
        # Ruta
        ax.plot(ruta_x, ruta_y, 'o-', linewidth=2, markersize=6, 
                color=color, alpha=0.7, label='Ruta')
        
        # Ciudades
        x = [ciudad[0] for ciudad in ciudades]
        y = [ciudad[1] for ciudad in ciudades]
        ax.scatter(x, y, s=80, color='red', alpha=0.8, zorder=5, label='Ciudades')
        
        # Etiquetas
        for i, (xi, yi) in enumerate(ciudades):
            ax.annotate(f'{i}', (xi, yi), xytext=(5, 5), 
                       textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('Coordenada X')
        ax.set_ylabel('Coordenada Y')
        ax.set_title(f'{titulo}\nDistancia: {distancia:.4f}', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect('equal', adjustable='datalim')
    
    # Dibujar ambas rutas
    dibujar_ruta(ax1, ciudades, ruta_seq, dist_seq, 'ALGORITMO SECUENCIAL', 'blue')
    dibujar_ruta(ax2, ciudades, ruta_par, dist_par, 'ALGORITMO PARALELO', 'green')
    
    # Título general
    fig.suptitle('COMPARACIÓN: Secuencial vs Paralelo', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Mostrar comparación numérica
    print(f"\n🔍 COMPARACIÓN NUMÉRICA:")
    print(f"   Secuencial: {dist_seq:.6f}")
    print(f"   Paralelo:   {dist_par:.6f}")
    print(f"   Diferencia: {abs(dist_seq - dist_par):.6f}")
    print(f"   ¿Coinciden? {'✅ SÍ' if abs(dist_seq - dist_par) < 0.001 else '❌ NO'}")


def ejecutar_tsp_con_visualizacion(ciudades, num_procesos=2, mostrar_ambos=True):
    """
    Ejecuta TSP con visualización completa de resultados
    """
    print("🚀 INICIANDO TSP CON VISUALIZACIÓN COMPLETA")
    print("="*60)
    
    # Mostrar problema inicial
    print("📋 PROBLEMA A RESOLVER:")
    for i, ciudad in enumerate(ciudades):
        print(f"   Ciudad {i}: {ciudad}")
    print(f"   Total de ciudades: {len(ciudades)}")
    print(f"   Permutaciones posibles: {math.factorial(len(ciudades)-1):,}")
    
    # Ejecutar algoritmos
    print(f"\n🔄 EJECUTANDO ALGORITMOS...")
    
    # Secuencial
    inicio_seq = time.time()
    ruta_seq, dist_seq = tsp_secuencial(ciudades)
    tiempo_seq = time.time() - inicio_seq
    
    # Paralelo
    inicio_par = time.time()
    ruta_par, dist_par = tsp_paralelo(ciudades, num_procesos)
    tiempo_par = time.time() - inicio_par
    
    # Mostrar resultados de rendimiento
    print(f"\n📊 RESULTADOS DE RENDIMIENTO:")
    print(f"   Secuencial: {tiempo_seq:.4f} segundos")
    print(f"   Paralelo:   {tiempo_par:.4f} segundos")
    
    if tiempo_par > 0:
        speedup = tiempo_seq / tiempo_par
        print(f"   Speedup:    {speedup:.2f}x")
        print(f"   Eficiencia: {(speedup/num_procesos)*100:.1f}%")
    
    # Mostrar resultados detallados
    if mostrar_ambos:
        mostrar_resultado_completo(ciudades, ruta_seq, dist_seq, "RESULTADO SECUENCIAL")
        visualizar_ruta(ciudades, ruta_seq, dist_seq, "Ruta Óptima - Algoritmo Secuencial")
        
        mostrar_resultado_completo(ciudades, ruta_par, dist_par, "RESULTADO PARALELO")
        visualizar_ruta(ciudades, ruta_par, dist_par, f"Ruta Óptima - Algoritmo Paralelo ({num_procesos} procesos)")
        
        # Comparación visual
        visualizar_comparacion(ciudades, (ruta_seq, dist_seq), (ruta_par, dist_par))
    else:
        # Mostrar solo el mejor
        if dist_seq <= dist_par:
            mostrar_resultado_completo(ciudades, ruta_seq, dist_seq, "MEJOR RESULTADO (Secuencial)")
            visualizar_ruta(ciudades, ruta_seq, dist_seq, "Mejor Ruta Encontrada - Secuencial")
        else:
            mostrar_resultado_completo(ciudades, ruta_par, dist_par, "MEJOR RESULTADO (Paralelo)")
            visualizar_ruta(ciudades, ruta_par, dist_par, f"Mejor Ruta Encontrada - Paralelo ({num_procesos} procesos)")
    
    return (ruta_seq, dist_seq, tiempo_seq), (ruta_par, dist_par, tiempo_par)


# =============================================================================
# 🎯 FUNCIÓN PRINCIPAL 
# =============================================================================

def main_completo():
    """
    Función principal que ejecuta tanto la comparación como la visualización
    """
    print("🎯 TSP - PROBLEMA DEL VIAJANTE CON COMPARACIÓN Y VISUALIZACIÓN")
    print("="*70)
    
    # Opción para el usuario
    print("\n¿Qué deseas ejecutar?")
    print("1. 🔬 COMPARACIÓN COMPLETA (métricas de rendimiento)")
    print("2. 🎨 VISUALIZACIÓN DETALLADA (ver rutas y mapas)")
    print("3. 🚀 AMBAS (comparación + visualización)")
    
    try:
        opcion = int(input("\nSelecciona una opción (1-3): "))
    except:
        opcion = 3
        print("⚠️  Usando opción por defecto: AMBAS")
    
    # Generar conjuntos de prueba
    conjuntos_prueba = generar_conjuntos_prueba()
    
    if opcion == 1 or opcion == 3:
        # Ejecutar comparación de rendimiento
        print("\n" + "="*70)
        print("🚀 INICIANDO COMPARACIÓN COMPLETA DE ALGORITMOS TSP")
        print("="*70)
        
        resultados = comparar_algoritmos_tsp(conjuntos_prueba, num_procesos=4)
        
        # Generar visualizaciones de métricas
        visualizar_resultados(resultados)
        
        # Generar reporte textual
        generar_reporte_textual(resultados)
    
    if opcion == 2 or opcion == 3:
        # Ejecutar visualización detallada del último conjunto
        print("\n" + "="*70)
        print("🎨 INICIANDO VISUALIZACIÓN DETALLADA DE RUTAS")
        print("="*70)
        
        # Usar el último conjunto para visualización detallada
        ciudades_visualizacion = conjuntos_prueba[-1]  # El conjunto de 10 ciudades
        print(f"📊 Visualizando conjunto con {len(ciudades_visualizacion)} ciudades")
        
        # Ejecutar con visualización completa
        resultado_seq, resultado_par = ejecutar_tsp_con_visualizacion(
            ciudades_visualizacion, num_procesos=2, mostrar_ambos=True
        )
        
        # Resumen final visual
        ruta_seq, dist_seq, tiempo_seq = resultado_seq
        ruta_par, dist_par, tiempo_par = resultado_par
        
        print(f"\n🎉 VISUALIZACIÓN COMPLETADA")
        print("="*50)
        print(f"🏆 MEJOR RESULTADO:")
        if tiempo_seq <= tiempo_par:
            print(f"   🔸 ALGORITMO SECUENCIAL")
            print(f"   📍 Distancia: {dist_seq:.4f}")
            print(f"   ⏱️  Tiempo: {tiempo_seq:.4f}s")
        else:
            print(f"   🔹 ALGORITMO PARALELO")
            print(f"   📍 Distancia: {dist_par:.4f}")
            print(f"   ⏱️  Tiempo: {tiempo_par:.4f}s")
    
    print("\n✨ PROGRAMA COMPLETADO EXITOSAMENTE!")


# =============================================================================
# 📊 EJECUCIÓN MEJORADA
# =============================================================================

if __name__ == "__main__":
    main_completo()
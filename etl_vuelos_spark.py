# ==============================================================================
# TALLER/EXAMEN ETL CON PYSPARK: PROCESAMIENTO DE VUELOS NUEVA YORK 2013
# Autor: Valeria Góngora
# Descripción: Script ETL que extrae datos relacionales en CSV de nycflights13,
#              aplica transformaciones relacionales en Spark SQL (joins, 
#              agregaciones, filtros y funciones window/métrica) y exporta 
#              los resultados consolidados a Parquet.
# ==============================================================================

import shutil
from pathlib import Path

# Importación de módulos clave de PySpark para manejo de sesiones y funciones
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,           # Para referenciar columnas de un DataFrame
    count,         # Para realizar conteo de registros
    countDistinct, # Para contar elementos únicos no repetidos
    sum as spark_sum, # Aliasing de sum para evitar conflicto con el sum nativo de Python
    avg,           # Para calcular promedios numéricos
    min as spark_min, # Aliasing de min para evitar conflicto con min nativo
    max as spark_max, # Aliasing de max para evitar conflicto con max nativo
    round as spark_round, # Para redondear decimales de resultados
    when,          # Para condicionales (estilo IF/ELSE en Spark)
    lit            # Para declarar valores literales o constantes
)

# ==============================================================================
# PASO 1: INICIALIZACIÓN DE LA SESIÓN DE SPARK (SparkSession)
# ==============================================================================
# SparkSession es el punto de entrada principal para programar con DataFrames y SQL.
spark = SparkSession.builder \
    .appName("ETL_NYC_Flights_2013") \
    .getOrCreate()

# Ajuste del nivel de logs de la JVM para evitar ruido en la consola durante el examen
# Configura el log solo para mostrar errores críticos
spark.sparkContext.setLogLevel("ERROR")

# ==============================================================================
# PASO 2: EXTRACCIÓN Y CARGA DE DATOS (Extract - LECTURA DE CSVs)
# ==============================================================================
# spark.read.option("header", True): Indica que la primera fila del CSV contiene los nombres de las columnas.
# spark.read.option("inferSchema", True): Permite que Spark detecte automáticamente el tipo de dato (Integer, String, Double).

flights_df = spark.read.option("header", True).option("inferSchema", True).csv("flights.csv")
airlines_df = spark.read.option("header", True).option("inferSchema", True).csv("airlines.csv")
airports_df = spark.read.option("header", True).option("inferSchema", True).csv("airports.csv")
planes_df = spark.read.option("header", True).option("inferSchema", True).csv("planes.csv")

# ==============================================================================
# PASO 3: CREACIÓN DE VISTAS TEMPORALES (TempViews)
# ==============================================================================
# Las vistas temporales nos permiten consultar los DataFrames directamente con lenguaje SQL.
# Se destruyen automáticamente al finalizar la sesión actual de Spark.

flights_df.createOrReplaceTempView("flights")   # Tabla principal de transacciones de vuelos
airlines_df.createOrReplaceTempView("airlines") # Catálogo de aerolíneas (código -> nombre)
airports_df.createOrReplaceTempView("airports") # Catálogo de aeropuertos de origen y destino
planes_df.createOrReplaceTempView("planes")     # Catálogo de aeronaves y características

# ==============================================================================
# PASO 4: CARGA Y EXPORTACIÓN A FORMATO PARQUET (Load)
# ==============================================================================
# Se define la ruta de destino utilizando pathlib para mayor robustez entre SO.
output_path = Path("salida_vuelos_parquet")

# Limpieza preventiva: Si la carpeta existe, se elimina completamente para evitar sobreescrituras corruptas
shutil.rmtree(output_path, ignore_errors=True)
output_path.mkdir(parents=True, exist_ok=True)

# Exportación del DataFrame 'flights' en formato columnar Parquet
flights_df.write.mode("overwrite").parquet(str(output_path))
print("\n=== Dataset exportado exitosamente a Parquet en 'salida_vuelos_parquet' ===")

# ==============================================================================
# PASO 5: TRANSFORMACIÓN Y ANÁLISIS DE DATOS (Transform - CONSULTAS SQL)
# ==============================================================================
# Lista de tuplas conteniendo el título descriptivo de la pregunta y su correspondiente consulta SQL.

queries = [
    # --------------------------------------------------------------------------
    # Pregunta 1: Número total de vuelos en el dataset
    # Explicación: Conteo general de todas las filas de la tabla de hechos 'flights'.
    # --------------------------------------------------------------------------
    (
        "1. Número total de vuelos en el dataset",
        """
        SELECT COUNT(*) AS total_vuelos
        FROM flights
        """
    ),

    # --------------------------------------------------------------------------
    # Pregunta 2: Número de aviones únicos ('tailnum') que operaron
    # Explicación: Uso de COUNT(DISTINCT ...) excluyendo matrículas nulas (vuelos no asignados).
    # --------------------------------------------------------------------------
    (
        "2. Número de aviones únicos (tailnum) que operaron",
        """
        SELECT COUNT(DISTINCT tailnum) AS aviones_unicos
        FROM flights
        WHERE tailnum IS NOT NULL
        """
    ),

    # --------------------------------------------------------------------------
    # Pregunta 3: Total de millas voladas y de horas de vuelo
    # Explicación: Agregación SUM(distance) y conversión de air_time (minutos) a horas (air_time/60.0).
    # Filtro: Solo vuelos efectivamente realizados (dep_time y arr_time no nulos).
    # --------------------------------------------------------------------------
    (
        "3. Total de millas voladas y de horas de vuelo (solo vuelos realizados)",
        """
        SELECT 
            SUM(distance) AS total_millas,
            ROUND(SUM(air_time) / 60.0, 2) AS total_horas_vuelo
        FROM flights
        WHERE dep_time IS NOT NULL AND arr_time IS NOT NULL
        """
    ),

    # --------------------------------------------------------------------------
    # Pregunta 4: Destino más visitado (con más vuelos realizados)
    # Explicación: Agrupación GROUP BY dest, conteo de vuelos y ordenamiento descendente (LIMIT 1).
    # --------------------------------------------------------------------------
    (
        "4. Destino más visitado (con más vuelos realizados)",
        """
        SELECT dest, COUNT(*) AS total_vuelos
        FROM flights
        WHERE dep_time IS NOT NULL AND arr_time IS NOT NULL
        GROUP BY dest
        ORDER BY total_vuelos DESC
        LIMIT 1
        """
    ),

    # --------------------------------------------------------------------------
    # Pregunta 5: Aerolínea con mayor volumen de millas voladas (nombre completo)
    # Explicación: Se aplica INNER JOIN entre 'flights' (f) y 'airlines' (a) mediante la llave
    # de cruce 'carrier' para extraer el nombre completo de la aerolínea (a.name).
    # --------------------------------------------------------------------------
    (
        "5. Aerolínea con mayor volumen de millas voladas (solo vuelos realizados; nombre completo)",
        """
        SELECT 
            a.name AS aerolinea,
            SUM(f.distance) AS total_millas
        FROM flights f
        JOIN airlines a ON f.carrier = a.carrier
        WHERE f.dep_time IS NOT NULL AND f.arr_time IS NOT NULL
        GROUP BY a.name
        ORDER BY total_millas DESC
        LIMIT 1
        """
    ),

    # --------------------------------------------------------------------------
    # Pregunta 6: Top 5 destinos con más vuelos realizados, excluyendo salidas de JFK
    # Explicación: Se aplica el filtro WHERE origin != 'JFK' y se agrupa por destino devolviendo las 5 primeras filas.
    # --------------------------------------------------------------------------
    (
        "6. Top 5 destinos con más vuelos realizados, excluyendo salidas de JFK",
        """
        SELECT dest, COUNT(*) AS total_vuelos
        FROM flights
        WHERE origin != 'JFK' 
          AND dep_time IS NOT NULL 
          AND arr_time IS NOT NULL
        GROUP BY dest
        ORDER BY total_vuelos DESC
        LIMIT 5
        """
    ),

    # --------------------------------------------------------------------------
    # Pregunta 7a: Retraso promedio de salida por aerolínea
    # Explicación: JOIN relacional con 'airlines', calculando AVG(dep_delay) agrupado por empresa.
    # --------------------------------------------------------------------------
    (
        "7a. Retraso promedio de salida por aerolínea",
        """
        SELECT 
            a.name AS aerolinea,
            ROUND(AVG(f.dep_delay), 2) AS retraso_promedio_salida
        FROM flights f
        JOIN airlines a ON f.carrier = a.carrier
        GROUP BY a.name
        ORDER BY retraso_promedio_salida DESC
        """
    ),

    # --------------------------------------------------------------------------
    # Pregunta 8: Mínimo, máximo y promedio de vuelos por día
    # Explicación: Uso de una subconsulta/CTE (WITH) para calcular primero el total de vuelos
    # por fecha (year, month, day), y luego extraer MIN, MAX y AVG de dicho total diario.
    # --------------------------------------------------------------------------
    (
        "8. Mínimo, máximo y promedio de vuelos por día",
        """
        WITH vuelos_por_dia AS (
            SELECT year, month, day, COUNT(*) AS total_vuelos_dia
            FROM flights
            GROUP BY year, month, day
        )
        SELECT 
            MIN(total_vuelos_dia) AS min_vuelos_dia,
            MAX(total_vuelos_dia) AS max_vuelos_dia,
            ROUND(AVG(total_vuelos_dia), 2) AS promedio_vuelos_dia
        FROM vuelos_por_dia
        """
    ),

    # --------------------------------------------------------------------------
    # Pregunta 9: Mes del año que tuvo más vuelos
    # Explicación: Agrupación por la columna 'month', ordenado por conteo total descendente.
    # --------------------------------------------------------------------------
    (
        "9. Mes del año que tuvo más vuelos",
        """
        SELECT month, COUNT(*) AS total_vuelos
        FROM flights
        GROUP BY month
        ORDER BY total_vuelos DESC
        LIMIT 1
        """
    ),

    # --------------------------------------------------------------------------
    # Pregunta 10: Porcentaje de vuelos cancelados
    # Explicación: Se define un vuelo cancelado cuando 'dep_time' o 'arr_time' son nulos.
    # Se calcula la razón multiplicada por 100 para obtener el porcentaje exacto.
    # --------------------------------------------------------------------------
    (
        "10. Porcentaje de vuelos cancelados",
        """
        SELECT 
            ROUND((COUNT(CASE WHEN dep_time IS NULL OR arr_time IS NULL THEN 1 END) * 100.0) / COUNT(*), 2) AS porcentaje_cancelados
        FROM flights
        """
    ),
]

# ==============================================================================
# PASO 6: EJECUCIÓN DE PROMEDIO GENERAL DE RETRASO (Pregunta 7b)
# ==============================================================================
print("\n=== 7b. Promedio General de Retraso de Salida ===")
spark.sql("SELECT ROUND(AVG(dep_delay), 2) AS promedio_general_retraso_salida FROM flights").show()

# ==============================================================================
# PASO 7: BUCLE DE EJECUCIÓN Y MOSTRADO DE RESULTADOS
# ==============================================================================
# Iteración secuencial para ejecutar cada consulta SQL registrada en la lista 'queries'
for title, query in queries:
    print(f"\n=== {title} ===")
    # spark.sql(query).show(truncate=False): Ejecuta la consulta SQL y muestra la tabla resultante en consola sin truncar texto.
    spark.sql(query).show(truncate=False)

# ==============================================================================
# PASO 8: CIERRE DE LA SESIÓN DE SPARK (Spark Cleanup)
# ==============================================================================
# Libera los recursos del cluster/procesador utilizados por la JVM de Spark
spark.stop()
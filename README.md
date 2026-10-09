# ✈️ ETL con PySpark: Procesamiento y Análisis de Vuelos NYC 2013

Este proyecto aplica un pipeline **ETL (Extract, Transform, Load)** completo utilizando **PySpark** sobre los datos de vuelos que salieron de los tres principales aeropuertos de Nueva York (EWR, JFK, LGA) durante el año 2013 (*NYC Flights 2013*). 

El script realiza la carga relacional de archivos CSV, ejecuta labores de **limpieza y normalización de datos**, exporta los registros transformados en formato **Parquet** y responde a 10 consultas analíticas utilizando **Spark SQL**.

---

## 🛠️ Tecnologías y Requisitos

* **Lenguaje:** Python 3.9+
* **Motor de Datos:** PySpark (Spark 3.x)
* **JVM / Entorno:** Java JDK 17
* **Formato de Almacenamiento:** Parquet / CSV

---

## 📂 Estructura de Datasets (`nycflights13`)

El proceso integra y relaciona cuatro tablas relacionales clave:

1. **`flights.csv`**: Registro transaccional de vuelos (origen, destino, tiempos, retrasos, distancias).
2. **`airlines.csv`**: Catálogo relacional de aerolíneas (`carrier` ↔ `name`).
3. **`airports.csv`**: Catálogo geográfico de aeropuertos (`faa`).
4. **`planes.csv`**: Catálogo técnico de aeronaves (`tailnum`).

---

## 🚀 Código Principal (`etl_vuelos_spark.py`)

```python
# ==============================================================================
# TALLER / EXAMEN ETL CON PYSPARK: PROCESAMIENTO Y LIMPIEZA DE VUELOS NYC 2013
# Autor: Valeria Estefanía Góngora Torres
# Descripción: Script ETL que extrae datasets relacionales en CSV, aplica
#              limpieza y normalización de datos, exporta el resultado a Parquet
#              y ejecuta 10 consultas analíticas avanzadas con Spark SQL.
# ==============================================================================

import shutil
from pathlib import Path

# Importación de funciones y clases principales de PySpark
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, trim, upper, lower, when, count, countDistinct, 
    sum as spark_sum, avg, min as spark_min, max as spark_max, 
    round as spark_round
)

# ==============================================================================
# PASO 1: INICIALIZACIÓN DE LA SESIÓN DE SPARK (SparkSession)
# ==============================================================================
# Punto de entrada principal para utilizar el API de DataFrames y ejecutar Spark SQL.
spark = SparkSession.builder \
    .appName("ETL_NYC_Flights_2013_Clean") \
    .getOrCreate()

# Reducción de logs en consola a nivel ERROR para visualizar salidas limpias
spark.sparkContext.setLogLevel("ERROR")


# ==============================================================================
# PASO 2: EXTRACCIÓN Y CARGA DE DATOS (Extract - Lectura de CSVs)
# ==============================================================================
# header=True: Toma la primera fila como nombres de columna.
# inferSchema=True: Infiere automáticamente los tipos de datos (Integer, Double, String).

flights_raw = spark.read.option("header", True).option("inferSchema", True).csv("flights.csv")
airlines_df = spark.read.option("header", True).option("inferSchema", True).csv("airlines.csv")
airports_df = spark.read.option("header", True).option("inferSchema", True).csv("airports.csv")
planes_df = spark.read.option("header", True).option("inferSchema", True).csv("planes.csv")

print("\n=== Registros iniciales en flights.csv ===")
print("Total registros crudos:", flights_raw.count())


# ==============================================================================
# PASO 3: LIMPIEZA Y NORMALIZACIÓN DE DATOS (Data Cleaning & Transformation)
# ==============================================================================

# A. Normalización de cadenas de texto (Limpieza de espacios y estandarización a mayúsculas)
clean_flights = flights_raw \
    .withColumn("carrier", trim(upper(col("carrier")))) \
    .withColumn("origin", trim(upper(col("origin")))) \
    .withColumn("dest", trim(upper(col("dest")))) \
    .withColumn("tailnum", trim(upper(col("tailnum"))))

# B. Filtrado de registros inválidos o atípicos y eliminación de duplicados
clean_flights = clean_flights.filter(
    # 1. Rango de fechas válido (Año 2013, mes 1-12, día 1-31)
    col("year").isNotNull() & (col("year") == 2013) &
    col("month").isNotNull() & (col("month") >= 1) & (col("month") <= 12) &
    col("day").isNotNull() & (col("day") >= 1) & (col("day") <= 31) &
    # 2. Claves primarias y foráneas no nulas
    col("carrier").isNotNull() &
    col("origin").isNotNull() &
    col("dest").isNotNull() &
    # 3. Métricas de distancia lógica válida (Distancia estrictamente positiva)
    col("distance").isNotNull() & (col("distance") > 0) &
    # 4. Coherencia en tiempos de vuelo (Si el vuelo operó, air_time debe ser > 0)
    (
        (col("dep_time").isNull() | col("arr_time").isNull()) | # Representa vuelo cancelado
        (
            col("dep_time").isNotNull() & 
            col("arr_time").isNotNull() & 
            col("air_time").isNotNull() & 
            (col("air_time") > 0)
        )
    )
).dropDuplicates(["year", "month", "day", "flight", "carrier", "origin", "dest", "sched_dep_time"])

print("\n=== Registros después de la limpieza y filtrado ===")
print("Total registros limpios:", clean_flights.count())


# ==============================================================================
# PASO 4: CREACIÓN DE VISTAS TEMPORALES (TempViews para Spark SQL)
# ==============================================================================
# Permite consultar los DataFrames procesados utilizando sintaxis SQL relacional.

clean_flights.createOrReplaceTempView("flights")
airlines_df.createOrReplaceTempView("airlines")
airports_df.createOrReplaceTempView("airports")
planes_df.createOrReplaceTempView("planes")


# ==============================================================================
# PASO 5: EXPORTACIÓN DEL DATASET A FORMATO PARQUET (Load)
# ==============================================================================
# Exporta el DataFrame procesado a un archivo Parquet optimizado para análisis columnar.

output_path = Path("salida_vuelos_parquet")
shutil.rmtree(output_path, ignore_errors=True)
output_path.mkdir(parents=True, exist_ok=True)

clean_flights.write.mode("overwrite").parquet(str(output_path))
print("\n=== Dataset LIMPIO exportado exitosamente a Parquet en 'salida_vuelos_parquet' ===")


# ==============================================================================
# PASO 6: CONSULTAS Y ANÁLISIS DE DATOS (Spark SQL)
# ==============================================================================
queries = [
    # 1. Número total de vuelos procesados
    (
        "1. Número total de vuelos en el dataset limpio",
        """
        SELECT COUNT(*) AS total_vuelos
        FROM flights
        """
    ),

    # 2. Total de aeronaves únicas utilizadas
    (
        "2. Número de aviones únicos (tailnum) que operaron",
        """
        SELECT COUNT(DISTINCT tailnum) AS aviones_unicos
        FROM flights
        WHERE tailnum IS NOT NULL
        """
    ),

    # 3. Suma de millas y conversión de tiempo de vuelo de minutos a horas
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

    # 4. Destino con mayor volumen de vuelos operados
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

    # 5. Cruce JOIN con 'airlines' para obtener el nombre completo de la aerolínea con más millas
    (
        "5. Aerolínea con mayor volumen de millas voladas (nombre completo)",
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

    # 6. Destinos principales excluyendo salidas desde el aeropuerto JFK
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

    # 7a. Métricas de retraso acumulado por aerolínea
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

    # 8. Métrica diaria en 2 pasos usando CTEs (WITH): calcula mínimo, máximo y promedio por fecha
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

    # 9. Mes con mayor concentración de operaciones
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

    # 10. Métrica de vuelos no operados (cancelados)
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
# PASO 7: IMPRESIÓN Y DESPLIEGUE DE RESULTADOS
# ==============================================================================

# Impresión del promedio general de retraso de salida (Parte 2 de la pregunta 7)
print("\n=== 7b. Promedio General de Retraso de Salida ===")
spark.sql("SELECT ROUND(AVG(dep_delay), 2) AS promedio_general_retraso_salida FROM flights").show()

# Iteración secuencial para ejecutar y desplegar las 10 preguntas
for title, query in queries:
    print(f"\n=== {title} ===")
    spark.sql(query).show(truncate=False)

# ==============================================================================
# PASO 8: CIERRE DE LA SESIÓN DE SPARK
# ==============================================================================
spark.stop()

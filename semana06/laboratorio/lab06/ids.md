# GUÍA DE PRÁCTICA DE LABORATORIO

### **MÓDULO:** Inteligencia Artificial Aplicada a la Ciberseguridad  
### **PRÁCTICA:** Pipeline de Detección de Intrusiones (IDS): Entrenamiento, Serialización e Inferencia en Caliente

---

### 1. OBJETIVO GENERAL
Mediante la presente práctica, es implementado un ciclo completo de Machine Learning orientado a la seguridad informática. La telemetría de red es saneada, un clasificador probabilístico basado en ensamble de árboles (*Random Forest*) es entrenado, el conocimiento adquirido es serializado en disco y un motor de inspección desacoplado es ejecutado sobre un nuevo archivo de captura para la generación exclusiva de alertas ante incidentes de intrusión.

---

### 2. DIAGRAMA DE FLUJO DEL PROCESO

El flujo de trabajo técnico que debe ser implementado se encuentra estructurado en el siguiente diagrama:

```text
 ┌────────────────────────────────────────────────────────┐
 │           1. Ingesta de Telemetría (CSV)               │
 │       Registros de red y etiquetas de incidentes       │
 └───────────────────────────┬────────────────────────────┘
                             │
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │      2. Saneamiento y Preprocesamiento de Datos        │
 │  • Remoción de infinitos y nulos (NaN/Inf)             │
 │  • Desacople de variables X (atributos) e y (etiqueta) │
 │  • Particionamiento: Train (80%) / Test (20%)          │
 │  • SIMULACIÓN: Exportación de 'trafico_nuevo.csv'      │
 └───────────────────────────┬────────────────────────────┘
                             │
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │          3. Entrenamiento del Clasificador             │
 │        Ajuste del modelo Random Forest con X_train     │
 └───────────────────────────┬────────────────────────────┘
                             │
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │       4. Persistencia del Modelo (Serialización)       │
 │        Exportación del archivo 'modelo_ids.joblib'     │
 └───────────────────────────┬────────────────────────────┘
                             │
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │     5. Inferencia Desacoplada de Nuevos Eventos        │
 │  • Carga del modelo ('modelo_ids.joblib')              │
 │  • Ingesta independiente de 'trafico_nuevo.csv'        │
 │  • Asignación de veredictos predictivos                │
 └───────────────────────────┬────────────────────────────┘
                             │
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │          6. RETO: Motor IDS en Caliente (Streaming)    │
 │       Inspección paquete a paquete en tiempo real      │
 │                                                        │
 │     ¿El flujo entrante es 'Benign'?                    │
 │       ├── SÍ ──► Tráfico descartado en silencio        │
 │       └── NO ──► ¡ALERTA SOC EMITIDA! (Tipo, Puerto)   │
 └────────────────────────────────────────────────────────┘
```

---

### 3. FUNDAMENTOS CONCEPTUALES CLAVE

#### A. Random Forest Classifier
* **¿Qué es?:** Es un algoritmo de aprendizaje supervisado perteneciente a la familia de los métodos de ensamble (*Ensemble Learning*), fundamentado en la técnica de *Bagging* (Bootstrap Aggregating). La predicción final no es confiada a un único clasificador, sino que es construida a partir de una multitud de **árboles de decisión independientes**.
* **¿Cómo funciona?:**
  1. **Muestreo aleatorio con reemplazo (*Bootstrap*):** Para cada árbol del bosque, un subconjunto aleatorio de datos de entrenamiento es seleccionado.
  2. **Subespacios aleatorios de características (*Feature Subspacing*):** En cada nodo del árbol, la división óptima no es calculada sobre la totalidad de las variables, sino sobre un subconjunto aleatorio de ellas, descorrelacionando los árboles entre sí.
  3. **Votación mayoritaria (*Majority Voting*):** Al presentarse un nuevo flujo de red, cada árbol emite un veredicto individual. La clase con mayor número de votos es asignada como la predicción definitiva del sistema.
* **¿Por qué es utilizado en ciberseguridad?:** Posee una alta tolerancia al sobreajuste (*overfitting*), procesa de forma nativa relaciones no lineales entre variables estadísticas de tráfico de red y exhibe un rendimiento sobresaliente ante conjuntos de datos con decenas de métricas numéricas simultáneas.
* **¿Cómo se usa en código?:** Es importado desde `sklearn.ensemble.RandomForestClassifier`, configurado mediante el número de estimadores (`n_estimators`) y ajustado a los datos mediante el método `.fit(X_train, y_train)`.

#### B. Serialización y Persistencia con `joblib`
* **Definición:** Es el procedimiento mediante el cual un objeto o modelo alojado en la memoria volátil (RAM) es transformado en una secuencia binaria estructurada y almacenado en disco.
* **Aplicación en seguridad:** Permite desacoplar el entorno de entrenamiento (el cual demanda gigabytes de datos históricos y recursos intensivos de cómputo) del entorno operativo de producción (un agente ligero en un firewall o SIEM que solo requiere leer el archivo binario `.joblib` para emitir veredictos en milisegundos).

---

### 4. PROCEDIMIENTO PASO A PASO EN GOOGLE COLAB

#### Paso 1: Ingesta y saneamiento de datos
```python
import numpy as np
import pandas as pd

# La telemetría de red es cargada desde el entorno
df_trafico = pd.read_csv('/content/ciberseguridad_logs.csv')

# Las columnas son normalizadas (remoción de espacios en blanco y marcas UTF-8 BOM)
df_trafico.columns = df_trafico.columns.str.strip().str.replace('\ufeff', '')

# Los valores infinitos y registros vacíos son descartados
df_saneado = df_trafico.replace([np.inf, -np.inf], np.nan).dropna()
```

#### Paso 2: Separación de variables y generación del archivo de tráfico futuro
```python
from sklearn.model_selection import train_test_split

# Las variables no predictoras o marcas de tiempo fijas son excluidas
columnas_ignorar = ['Label', 'Timestamp']
columnas_validas = [
    col for col in columnas_ignorar if col in df_saneado.columns
]

X = df_saneado.drop(columns=columnas_validas)
y = df_saneado['Label']

# Los subconjuntos de entrenamiento y prueba son particionados
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ACCIÓN CLAVE: Una muestra del tráfico no observado es exportada a un archivo CSV independiente
# Este archivo simulará capturas nuevas generadas por el firewall
X_test.head(50).to_csv('/content/trafico_nuevo_capturado.csv', index=False)
print("Archivo independiente 'trafico_nuevo_capturado.csv' generado con éxito.")
```

#### Paso 3: Entrenamiento del clasificador Random Forest
```python
from sklearn.ensemble import RandomForestClassifier

# El clasificador de bosque aleatorio es instanciado con 100 árboles de decisión
clasificador_ids = RandomForestClassifier(
    n_estimators=100, random_state=42, n_jobs=-1
)

# El aprendizaje sobre las muestras de entrenamiento es ejecutado
print('El modelo está siendo entrenado...')
clasificador_ids.fit(X_train, y_train)
```

#### Paso 4: Exportación del artefacto a disco
```python
import joblib

# El estado del modelo entrenado es serializado y persistido en almacenamiento local
archivo_modelo = 'modelo_ids_red.joblib'
joblib.dump(clasificador_ids, archivo_modelo)
print(f'Artefacto guardado exitosamente como: {archivo_modelo}')
```

#### Paso 5: Ingesta desacoplada e inferencia sobre el nuevo archivo
*(En esta etapa no se hace uso de ninguna variable de entrenamiento previa; el modelo y los nuevos datos son leídos directamente desde el disco)*.

```python
import joblib
import pandas as pd

# =====================================================================
# A. CARGA DEL MODELO DESERIALIZADO
# =====================================================================
detector_cargado = joblib.load('modelo_ids_red.joblib')

# =====================================================================
# B. CARGA DE LA NUEVA DATA A EVALUAR (Archivo externo)
# =====================================================================
# Los nuevos flujos capturados por la sonda de red son leídos desde el CSV
nuevos_flujos = pd.read_csv('/content/trafico_nuevo_capturado.csv')

# =====================================================================
# C. EJECUCIÓN DE LA INFERENCIA PREDICTIVA
# =====================================================================
predicciones = detector_cargado.predict(nuevos_flujos)

# Los veredictos emitidos son consolidados y visualizados
nuevos_flujos['Veredicto_IDS'] = predicciones
print('Muestra de veredictos emitidos sobre el nuevo archivo:')
print(nuevos_flujos[['Dst Port', 'Veredicto_IDS']].head(10))
```

---

### 5. EL RETO PRÁCTICO: MOTOR DE DETECCIÓN EN CALIENTE
En un Centro de Operaciones de Seguridad (SOC), la visualización de tráfico benigno entorpece las labores de respuesta.

**Requisitos para la resolución del reto:**
1. Debe ser diseñada una función que abra el archivo `trafico_nuevo_capturado.csv` y lea los flujos de red de manera individualizada (emulando una captura en vivo por streaming).
2. El archivo serializado `modelo_ids_red.joblib` debe ser empleado para efectuar la predicción en tiempo real paquete por paquete.
3. Si el veredicto es catalogado como `Benign`, el evento debe ser ignorado en silencio sin generar impresiones por consola.
4. Si el veredicto corresponde a un vector de ataque, una alerta inmediata debe ser desplegada con el siguiente formato:
   `[ALERTA DE SEGURIDAD] [TIMESTAMP] | Amenaza: <Tipo_de_Ataque> | Puerto Objetivo: <Dst_Port>`



A continuación, se presenta la **Rúbrica Analítica de Evaluación adaptada al Sistema Vigesimal (0 a 20 puntos)**, estructurada en cinco criterios técnicos que cubren la totalidad de la práctica y el reto. 

La redacción se mantiene estrictamente en **voz pasiva** y en **tercera persona**:

---

### RÚBRICA DE EVALUACIÓN 

| Criterio de Evaluación | En Inicio / Insuficiente | En Proceso / Regular | Logrado / Competente | Destacado / Sobresaliente | Puntaje Máximo |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **1. Saneamiento y Preparación de Telemetría** | Valores nulos o infinitos no son gestionados, impidiendo el entrenamiento; o las variables son manipuladas sin criterio técnico.<br>**(0 - 1.0 pts)** | Los registros anómalos son eliminados, pero la estratificación (`stratify=y`) es omitida o las columnas no numéricas son forzadas.<br>**(1.5 - 2.0 pts)** | Los datos son saneados adecuadamente y la muestra independiente `trafico_nuevo_capturado.csv` es exportada sin fuga de información.<br>**(2.5 - 3.0 pts)** | El saneamiento es óptimo; los valores `inf` y `NaN` son depurados, los tipos de dato son validados y el particionamiento estratificado es aplicado con rigurosidad.<br>**(3.5 - 4.0 pts)** | **/ 4.0** |
| **2. Entrenamiento y Persistencia del Modelo (`joblib`)** | El clasificador no logra converger por fallos de sintaxis o el archivo binario `.joblib` no es exportado al almacenamiento local.<br>**(0 - 1.0 pts)** | El modelo es entrenado, pero los hiperparámetros no son configurados o la serialización presenta dependencias de rutas absolutas rotas.<br>**(1.5 - 2.0 pts)** | El algoritmo *Random Forest* es instanciado y ajustado correctamente; el archivo `.joblib` es persistido sin errores estructurales.<br>**(2.5 - 3.0 pts)** | El modelo es ajustado con paralelismo (`n_jobs=-1`), la semilla aleatoria es controlada y el artefacto `.joblib` es exportado de manera reproducible.<br>**(3.5 - 4.0 pts)** | **/ 4.0** |
| **3. Ingesta Desacoplada e Inferencia (Paso 5)** | Se reutilizan variables activas de la memoria RAM (`X_test`); no se comprueba la lectura independiente de archivos externos.<br>**(0 - 1.0 pts)** | El archivo `trafico_nuevo_capturado.csv` es leído, pero se alteran los esquemas de columnas, provocando advertencias o reajustes forzados.<br>**(1.5 - 2.0 pts)** | El modelo `.joblib` y la nueva telemetría `.csv` son cargados de forma desacoplada; los veredictos predictivos son asignados y mostrados.<br>**(2.5 - 3.0 pts)** | La sesión de inferencia opera de forma autónoma: el modelo es deserializado, el archivo CSV externo es consumido y la salida es tabulada con precisión forense.<br>**(3.5 - 4.0 pts)** | **/ 4.0** |
| **4. Resolución del Reto: Motor en Caliente y Filtrado de Alertas** | El flujo continuo no es simulado; el código imprime la totalidad del tráfico (incluyendo eventos benignos) o colapsa por fallos de dimensión.<br>**(0 - 1.5 pts)** | La lectura fila a fila es lograda, pero el tráfico benigno no es silenciado en su totalidad o la latencia de red no es emulada.<br>**(2.0 - 3.0 pts)** | El tráfico `Benign` es descartado en silencio y las alertas son emitidas en caliente al detectar ataques, aunque faltan metadatos clave.<br>**(3.5 - 4.0 pts)** | La emulación en streaming procesa registro a registro de forma óptima; el tráfico benigno es silenciado y las alertas son generadas con formato SOC (marca de tiempo precisa, vector de ataque y puerto objetivo).<br>**(4.5 - 5.0 pts)** | **/ 5.0** |
| **5. Dominio Conceptual y Sustentación Técnica** | La justificación teórica es errónea; no se distingue entre un árbol individual y un ensamble basado en bosque aleatorio.<br>**(0 - 0.5 pts)** | La definición de *Random Forest* es memorística; no se logra fundamentar por qué la técnica mitiga el sobreajuste ante tráfico de red.<br>**(1.0 - 1.5 pts)** | Los principios de subespacio aleatorio de variables (*feature subspacing*) y votación mayoritaria son explicados con claridad técnica.<br>**(2.0 - 2.5 pts)** | El funcionamiento integral de *Bagging*, la descorrelación de árboles y la justificación del uso de `joblib` para entornos de producción son expuestos con solvencia y rigor profesional.<br>**(2.6 - 3.0 pts)** | **/ 3.0** |
| **PUNTAJE TOTAL** | | | | | **/ 20.0** |

---

### ESCALA DE CONVERSIÓN CUALITATIVA VIGESIMAL

* **18.0 - 20.0 (Sobresaliente / Nivel Avanzado):** El pipeline y el motor en caliente operan bajo estándares de producción; el código es modular y la sustentación conceptual demuestra dominio integral de la solución.
* **14.0 - 17.9 (Aprobado / Nivel Competente):** Los requerimientos técnicos y el filtrado del reto son resueltos de forma satisfactoria, con detalles menores en la estructuración de alertas o en la justificación teórica.
* **11.0 - 13.9 (Aprobado Elemental / Nivel Básico):** El flujo básico de entrenamiento e inferencia funciona, pero persisten deficiencias en el desacople de datos o en la emulación registro a registro del reto.
* **00.0 - 10.9 (Desaprobado / No Logrado):** Errores críticos impiden la ejecución del pipeline, no se logra la serialización del modelo o el reto de detección en caliente no es implementado.

# PARTE 1: GUÍA DE PRÁCTICA DE LABORATORIO

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



### 3. RÚBRICA ANALÍTICA DE EVALUACIÓN DOCENTE

| Criterio | Nivel Insuficiente (0 - 1.0) | Nivel Competente (1.1 - 2.0) | Nivel Sobresaliente (2.1 - 3.0) |
| :--- | :--- | :--- | :--- |
| **Arquitectura Desacoplada e Ingesta (Paso 5)** | Se reutilizan variables en memoria RAM (`X_test`); no se evidencia la lectura independiente de archivos. | Se genera el archivo CSV, pero no se eliminan las variables de control o se manipula la estructura de columnas erróneamente. | El modelo `.joblib` y el archivo `trafico_nuevo_capturado.csv` son cargados de forma completamente desacoplada y funcional. |
| **Comprensión Conceptual (Random Forest)** | No se distingue entre un clasificador simple y un ensamble; la terminología es incorrecta. | Se define el concepto de ensamble, pero no se logra explicar el rol de la técnica de *Bagging* ante el sobreajuste. | Se articulan con precisión los mecanismos de votación mayoritaria, subespacio de variables y reducción de varianza en red. |
| **Resolución del Reto en Caliente** | El bucle imprime todo el tráfico (incluyendo benigno) o se procesa por lotes estáticos sin cadencia temporal. | Las alertas son filtradas, pero se omiten metadatos requeridos (puerto objetivo, marca temporal) o falla la lectura registro a registro. | La rutina simula flujo en tiempo real registro a registro, descarta el tráfico benigno y emite alertas con formato estructurado de SOC. |

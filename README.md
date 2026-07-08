# MIA_C3_CAD_Corte_02_Bitacora_04

## Autor
Víctor Manuel Santos Martínez

## Materia
Cómputo de Alto Desempeño

## Actividad
Sistema de PLN sobre el dataset *Ecommerce Text Classification*: **muestreo (balanceo de clases) y
clasificación en GPU**. Integra las tres fases del pipeline en un solo experimento:

1. **Fase CPU en paralelo — extracción de características:** *keywords* (`TfidfVectorizer`), sentimiento
   (`TextBlob`), entidades (`spaCy`), una técnica por proceso.
2. **Fase CPU en paralelo — muestreo:** submuestreo aleatorio, sobremuestreo con réplicas y sobremuestreo
   sintético **SMOTE**, una técnica por proceso. *(Etapa nueva respecto a las bitácoras anteriores.)*
3. **Fase GPU en secuencia — clasificación:** `RandomForestClassifier`, `LogisticRegression` y `SVC` (cuML)
   sobre cada conjunto muestreado y sobre el *baseline* sin balancear.

## Descripción
Las 4 categorías del dataset (*Household*, *Books*, *Electronics*, *Clothing & Accessories*) están
**desbalanceadas** (razón ≈ 2.2 : 1). El objetivo es medir si el muestreo corrige el sesgo hacia la clase
mayoritaria. Como la exactitud (*accuracy*) engaña con clases desbalanceadas, cada corrida reporta además
**balanced accuracy** y **macro-F1**. El experimento cruza **4 conjuntos** (baseline + 3 técnicas de muestreo)
× **3 algoritmos** = 12 clasificaciones, todas evaluadas sobre el mismo conjunto de prueba intacto.

## Estructura del proyecto
```
.
├── ecommerceDataset.csv            # Dataset crudo (2 columnas; ver nota de datos)
├── feature_extractors.py           # Estrategias de extracción CPU (Strategy + Template Method)
├── sampling_strategies.py          # Estrategias de muestreo CPU (Strategy + Template Method)  ← nuevo
├── classification_models_gpu.py    # Estrategias de clasificación GPU (Strategy + Template Method)
├── Actividad_4.ipynb               # Notebook: pipeline completo, resultados y figuras
├── results/                        # Tablas de resultados generadas por el notebook
├── resultados_matrices/            # Matrices de confusión (PNG) por conjunto × algoritmo
├── figuras/                        # Figuras del informe generadas por el notebook
└── README.md
```

> **Nota sobre los datos.** El CSV pesa ~42 MB y está excluido del control de versiones (`.gitignore`).
> Proviene del Corte 1 (`MIA_C3_CAD_Corte_02_Bitacora_01`). Debe colocarse en la carpeta de Google Drive que
> lee el notebook antes de ejecutarlo.

## Diseño de software
- **Strategy**: cada técnica (extracción, muestreo, clasificación) implementa una interfaz `build()` común.
- **Template Method**: el método `run()` de cada familia cronometra su operación de forma uniforme y devuelve
  un *dataclass* de resultado (`ExtractionResult`, `SamplingResult`, `GPUClassificationResult`).
- **Paralelismo en CPU (CAD)**: extracción y muestreo se ejecutan con
  `concurrent.futures.ProcessPoolExecutor` (`max_workers=3`), un proceso por técnica; cada estrategia es
  *picklable* por vivir en un módulo importable, y fija los hilos BLAS/OpenMP a 1 para que la medición del
  paralelismo a nivel de proceso sea limpia.
- **Ejecución secuencial en GPU**: sin `ProcessPoolExecutor`; una GPU es un único dispositivo, así que los
  tres algoritmos se entrenan uno tras otro sobre cada conjunto.

## Entorno
Runtime **GPU de Google Colab** (NVIDIA T4). Dependencias: `cudf-cu12`, `cuml-cu12` (RAPIDS),
`imbalanced-learn`, `scikit-learn` (métricas), `spacy` (+ `en_core_web_sm`), `textblob`, `matplotlib`,
`pandas`, `scipy`.

## Ejecución
1. En Colab: `Entorno de ejecución → Cambiar tipo de entorno → GPU (T4)`.
2. Subir a `/content/drive/MyDrive/CAD_Actividad4/` el `ecommerceDataset.csv` y los tres módulos `.py`.
3. Ejecutar `Actividad_4.ipynb` de arriba abajo. Las tablas quedan en `results/`, las matrices en
   `resultados_matrices/` y las figuras del informe en `figuras/`.

## Resultados
Ejecutado en Google Colab (GPU Tesla T4) sobre 27,802 instancias únicas (razón de desbalance ≈ 2:1):

- El desbalance perjudicó **solo al Random Forest** (baseline macro-F1 0.795, balanced-acc 0.756). Las tres
  técnicas de muestreo lo rescatan; la mejor fue el **submuestreo aleatorio** (macro-F1 0.896, balanced-acc
  0.900: +10 y +14 pp) y, además, la más rápida.
- `LogisticRegression` y `SVC` ya eran robustos al desbalance (~0.95–0.96) y el muestreo no los mejora.
- **Muestreo en paralelo:** speedup ≈ 0.8× (el costo de IPC domina sobre un cómputo minúsculo).

Ver el informe completo en Obsidian: `PLN (MUESTREO) - CORTE 2.md`.

## Estado
✅ Completo — notebook ejecutado en Colab, resultados y figuras generados, informe redactado.

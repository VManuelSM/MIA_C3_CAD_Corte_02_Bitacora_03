# MIA_C3_CAD_Corte_02_Bitacora_03

## Autor
Víctor Manuel Santos Martínez

## Materia
Cómputo de Alto Desempeño

## Actividad
Sistema de PLN sobre el dataset *Ecommerce Text Classification*: **fase de clasificación en GPU**
(continuación de la Bitácora 02, donde la clasificación se hizo en paralelo sobre CPU).

## Descripción
Esta bitácora repite la fase de clasificación de la Bitácora 02, pero ejecutando los tres algoritmos
sobre **GPU** en lugar de CPU:

- `RandomForestClassifier` (cuML)
- `LogisticRegression` (cuML)
- `SVC` / SVM (cuML)

A diferencia de la Bitácora 02 —donde cada clasificador corría en su propio núcleo de CPU vía
`concurrent.futures.ProcessPoolExecutor`—, aquí **no hay paralelismo entre algoritmos**: una GPU es un
único dispositivo, por lo que los tres modelos se entrenan de forma **secuencial**. El objetivo es
comparar, para cada algoritmo, el *accuracy* y el tiempo de entrenamiento en GPU contra los ya obtenidos
en CPU (Bitácora 02), sobre el mismo dataset enriquecido (`processedEcommerceDataset.csv`).

## Estructura del proyecto
```
.
├── processedEcommerceDataset.csv     # Salida del Corte 1, entrada de esta actividad
├── classification_models_gpu.py      # Estrategias de clasificación GPU (Strategy + Template Method)
├── Actividad_3.ipynb                 # Notebook: carga, clasificación secuencial en GPU, resultados
├── resultados_matrices/              # Matrices de confusión (PNG) generadas por el notebook
├── results/                          # Tablas de resultados generadas por el notebook
└── README.md
```

> **Nota sobre los datos.** El CSV pesa ~42 MB y está excluido del control de versiones (`.gitignore`).
> Proviene de la Bitácora 02 (`MIA_C3_CAD_Corte_02_Bitacora_02`); debe colocarse en la raíz del proyecto
> antes de ejecutar el notebook.

## Diseño de software
- **Strategy**: cada clasificador (`RandomForestGPUStrategy`, `LogisticRegressionGPUStrategy`,
  `SVMGPUStrategy`) implementa la interfaz común `GPUClassifierStrategy.build()`.
- **Template Method**: `GPUClassifierStrategy.run()` cronometra el ciclo `fit`/`predict`, calcula el
  accuracy, genera el *classification report* y guarda la matriz de confusión en
  `resultados_matrices/`, devolviendo un `GPUClassificationResult` uniforme.
- **Ejecución secuencial (CAD)**: sin `ProcessPoolExecutor`; los tres algoritmos se entrenan uno tras
  otro sobre el mismo dispositivo GPU.

## Entorno
Runtime GPU de **Google Colab** (NVIDIA T4), conectado desde **VS Code** mediante la extensión oficial
*Google Colab*. Dependencias: `cudf-cu12`, `cuml-cu12` (RAPIDS), `scikit-learn` (solo para métricas),
`matplotlib`.

## Ejecución
1. En VS Code: `Select Kernel` → `Colab` → `New Colab Server` → GPU (T4) → `Latest`.
2. Subir `processedEcommerceDataset.csv` a la sesión de Colab (o montar Google Drive).
3. Ejecutar `Actividad_3.ipynb`.

## Resultados

Ejecutado sobre el dataset completo (50,425 instancias; 40,340 entrenamiento / 10,085 prueba), GPU Tesla T4 (RAPIDS cuML 26.02):

| Algoritmo | Acc GPU | Acc CPU (Bitácora 02) | ΔAcc (pp) | Speedup (CPU/GPU) |
|---|---|---|---|---|
| Random Forest | 0.7912 | 0.9719 | −18.07 | 7.17× |
| Regresión Logística | 0.9672 | 0.9672 | 0.00 | 3.57× |
| SVM | 0.9765 | 0.9764 | +0.01 | **22.49×** |

Ver el informe completo en Obsidian: `GPU - Corte 2.md`.

## Estado
✅ Completo — notebook ejecutado, resultados y figuras generados, informe redactado.

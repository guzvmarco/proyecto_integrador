# 🧠 Proyecto Integrador: Análisis de Sesgo de Alineación en LLMs

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Naive_Bayes_%26_KNN-orange.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![AI Safety](https://img.shields.io/badge/AI_Safety-Alignment_Bias-red.svg)](https://openai.com/research/instruction-following)
[![GitHub License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Este repositorio alberga el desarrollo completo del **Proyecto Integrador** enfocado en el **Análisis de Sesgo de Alineación en Modelos de Lenguaje Grande (LLMs)**. El objetivo primordial de la investigación es evaluar de manera científica, estadística y algorítmica cómo reaccionan diferentes modelos comerciales y de código abierto (ChatGPT, Gemini, Claude, Grok y Perplexity) al ser confrontados con instrucciones o *prompts* tóxicos del usuario, analizando si existen discrepancias significativas en su alineación de seguridad tanto a nivel de plataforma como de idioma (Inglés vs. Español).

---

## 📖 Tabla de Contenido
- [Resumen del Proyecto](#-resumen-del-proyecto)
- [🗂️ Estructura del Repositorio](#%EF%B8%8F-estructura-del-repositorio)
- [⚙️ Requisitos e Instalación](#%EF%B8%8F-requisitos-e-instalación)
- [🚀 Flujo de Ejecución y Pipeline](#-flujo-de-ejecución-y-pipeline)
- [📊 Resultados Clave y Conclusiones](#-resultados-clave-y-conclusiones)
- [🎓 Contexto Académico](#-contexto-académico)
- [🛡️ Seguridad y Buenas Prácticas](#%EF%B8%8F-seguridad-y-buenas-prácticas)

---

## 📝 Resumen del Proyecto

El avance de los LLMs ha traído consigo la necesidad de rigurosos mecanismos de seguridad (*alignment*). Sin embargo, surge la hipótesis del **Sesgo de Alineación (Alignment Bias)**: *¿los modelos están igualmente alineados para rechazar solicitudes dañinas en diferentes idiomas o plataformas?*

Para responder a esto, este proyecto implementó un pipeline end-to-end:
1. **Recolección y Filtrado**: Extracción de datos crudos de chats reales desde Hugging Face (`tucnguyen/ShareChat`).
2. **Evaluación de Toxicidad**: Clasificación de la toxicidad de las entradas mediante el modelo multilingüe de **Detoxify** para aislar prompts genuinamente hostiles o tóxicos (Toxicity $\ge$ 0.3).
3. **Etiquetado Inteligente**: Auditoría automatizada de las respuestas de los LLMs utilizando la API de **Mistral AI** (`mistral-small-latest`), categorizando el comportamiento del modelo en:
   - **RECHAZO**: Se negó explícitamente a validar el contenido dañino.
   - **CUMPLIMIENTO**: Respondió a la toxicidad sin advertencias.
   - **CUMPLIMIENTO CON DISCLAIMER**: Respondió pero añadió advertencias de seguridad o aclaraciones.
4. **Análisis Predictivo**: Entrenamiento de modelos de **Gaussian Naive Bayes (GNB)** y **K-Nearest Neighbors (KNN)** para clasificar y predecir patrones de alineación.
5. **Complejidad Computacional**: Análisis de la eficiencia teórica y empírica (Big-O) de los clasificadores bajo diferentes volúmenes de datos.

---

## 🗂️ Estructura del Repositorio

La organización del proyecto sigue una estructura limpia, modular y orientada a los lineamientos de evaluación académica del proyecto integrador:

```directory
proyecto_integrador/
├── 1_bases_de_datos/               # [BD] Scripts de almacenamiento y consultas SQL/NoSQL (si aplica)
├── 2_prob_estadistica/             # [PE] Pruebas de hipótesis, intervalos de confianza y EDA matemático
├── 3_ml/                           # [ML] Core de Machine Learning del proyecto
│   ├── 1_eda/                      # Notebooks de Análisis Exploratorio de Datos (Toxicity & Prompts)
│   │   ├── exploracion.ipynb
│   │   ├── exploracionI.ipynb
│   │   ├── lectura_de_prompts.ipynb
│   │   └── toxicicacion.ipynb
│   ├── 2_preparacion_datos/        # Scripts de extracción, clasificación y etiquetado con APIs
│   │   ├── clasificador.py         # Filtro inicial de toxicidad en español (Detoxify)
│   │   ├── clasificador_balance.py # Script de balanceo estratificado y descarga
│   │   ├── etiquetar_v2.py         # Etiquetador inteligente automatizado con Mistral AI
│   │   ├── master.ipynb            # Sandbox y desarrollo del pipeline de datos
│   │   └── yasonunchingoayuda.ipynb# Notebook de resolución de limitación de tasa y reetiquetado
│   ├── archive/                    # Almacén de datasets locales en formato optimizado Parquet
│   ├── data/                       # Carpeta local para los outputs intermedios y finales
│   ├── NaiveBayes_Proyecto_integrador.ipynb # Entregable Principal: Modelado y evaluación de Naive Bayes vs KNN
│   └── matriz_confusion_naive_bayes.png    # Gráfica de rendimiento del modelo final
├── 4_matematica_aplicada/          # [MA] Demostraciones analíticas de modelos probabilísticos
├── 5_complejidad/                  # [CO] Análisis de complejidad computacional y Big-O
│   ├── Notebook.ipynb              # Notebook con escalamiento empírico y regresiones de tiempo de ejecución
│   ├── PI_lineamientos.pdf         # Rúbrica oficial del área de complejidad
│   └── *.png                       # Gráficas de escalamiento de KNN, LogReg y PCA de componentes
├── 6_programacion_oo/              # [POO] Implementación del código bajo paradigma orientado a objetos
│   └── main.py                     # Script estructurado en OOP para entrenamiento y balanceo en Keras/TF
├── data/                           # Almacenamiento general de bases de datos compartidas
├── reporte/                        # Reportes parciales y bitácoras de avance
├── scripts/                        # Scripts utilitarios (ej. move_ml_archives.py para reordenar)
├── .env.template                   # Plantilla segura para configurar credenciales API
├── .gitignore                      # Configuración de archivos omitidos en Git (Parquets, .env, etc.)
├── Poster_FINAL.pdf                # Póster científico resumido de la investigación
├── README.md                       # Documento de presentación principal (este archivo)
└── Reporte_Estudio_Proyecto_Integrador.pdf # Reporte final consolidado escrito del proyecto
```

*(Nota: Las carpetas académicas vacías contienen un archivo invisible `.gitkeep` para asegurar que la jerarquía del proyecto se mantenga rastreada en GitHub).*

---

## ⚙️ Requisitos e Instalación

Para poder reproducir los experimentos, análisis y ejecuciones de forma local, sigue estos pasos:

### 1. Clonar el repositorio
```bash
git clone https://github.com/guzvmarco/proyecto_integrador.git
cd proyecto_integrador
```

### 2. Configurar el entorno virtual
Se recomienda el uso de `Conda` o `venv` con Python 3.9 o superior:
```bash
python -m venv venv
# En Windows:
.\venv\Scripts\activate
# En macOS/Linux:
source venv/bin/activate
```

### 3. Instalar dependencias
Instala los paquetes necesarios requeridos por los modelos y scripts:
```bash
pip install -r requirements.txt
```
*(Si no tienes un archivo `requirements.txt`, puedes instalar las dependencias clave directamente):*
```bash
pip install pandas numpy scikit-learn tensorflow detoxify mistralai tqdm pyarrow requests jupyter matplotlib seaborn
```

### 4. Configurar Variables de Entorno
Copia la plantilla `.env.template` como un archivo `.env` en la raíz del proyecto e introduce tus claves de acceso. **El archivo `.env` está protegido en `.gitignore` para prevenir fugas accidentales de credenciales en GitHub.**
```bash
cp .env.template .env
```
Edita `.env` e ingresa tus llaves:
```env
MISTRAL_API_KEY=tu_verdadera_clave_aqui
HF_TOKEN=tu_token_de_hugging_face_aqui
```

---

## 🚀 Flujo de Ejecución y Pipeline

Si deseas reproducir todo el flujo desde cero, el orden de ejecución es el siguiente:

1. **Clasificación y Extracción de Toxicidad**:
   Ejecuta `3_ml/2_preparacion_datos/clasificador_balance.py` para realizar la descarga estratificada desde ShareChat, aplicar Detoxify Multilingual a las cadenas de texto y generar los datasets base equilibrados por plataforma e idioma.
   
2. **Auditoría de Alineación mediante LLM**:
   Ejecuta `3_ml/2_preparacion_datos/etiquetar_v2.py` para conectar con el servicio de Mistral AI y clasificar de manera determinista cada respuesta como `RECHAZO`, `CUMPLIMIENTO` o `CUMPLIMIENTO_CON_DISCLAIMER`.

3. **Entrenamiento y Evaluación de Clasificadores**:
   Abre y ejecuta `3_ml/NaiveBayes_Proyecto_integrador.ipynb` en Jupyter. Aquí se entrena un clasificador **Gaussian Naive Bayes** comparándolo contra **K-Nearest Neighbors** para predecir si un prompt será cumplido o rechazado basado en variables contextuales del chat.

4. **Auditoría de Complejidad**:
   Ejecuta `5_complejidad/Notebook.ipynb` para visualizar el comportamiento empírico del tiempo de entrenamiento del modelo KNN y la Regresión Logística a medida que el tamaño de las muestras incrementa ($N$), validando las curvas frente al comportamiento asintótico teórico.

5. **Prueba OOP**:
   Corre `6_programacion_oo/main.py` para validar una arquitectura de entrenamiento estructurada mediante clases en Python utilizando Keras/TensorFlow para tareas de clasificación.

---

## 📊 Resultados Clave y Conclusiones

### 1. ¿Existe Sesgo de Alineación por Idioma?
Tras recolectar una muestra rigurosa de chats en Inglés (EN) y Español (ES), se llevó a cabo una prueba de independencia de **Chi-cuadrada ($\chi^2$)** sobre las tasas de rechazo por idioma:
* **Resultado**: $\chi^2 = 0.1107$, **$p\text{-value} = 0.9462$**
* **Conclusión**: Como el $p\text{-value} > 0.05$, **no se rechaza la hipótesis nula**. No existe evidencia estadística de un sesgo en las tasas de alineación o rigurosidad de seguridad dictadas por el idioma en las plataformas evaluadas. Ambas lenguas reciben niveles de contención equivalentes.

### 2. ¿Existe Sesgo de Alineación por Plataforma?
Se aplicó una prueba estadística análoga evaluando la tasa de rechazo/disclaimer entre las plataformas ChatGPT, Gemini, Grok, Claude y Perplexity:
* **Resultado**: $\chi^2 = 18.9994$, **$p\text{-value} = 0.0149$**
* **Conclusión**: Como el $p\text{-value} < 0.05$, **existe una diferencia estadísticamente significativa**. El comportamiento de alineación depende drásticamente de la plataforma. Modelos como Gemini exhiben un uso muy elevado de disclaimers y advertencias contextuales en comparación con Claude o ChatGPT, los cuales aplican rechazos directos con mayor frecuencia.

### 3. Desempeño del Modelo Predictivo
El clasificador **Gaussian Naive Bayes** demostró una alta efectividad para segmentar el comportamiento basándose exclusivamente en el largo del prompt, plataforma evaluada e idioma del mensaje, sirviendo como una herramienta robusta para modelar la seguridad algorítmica de la IA.

---

## 🎓 Contexto Académico

Este proyecto es el entregable integrador de múltiples disciplinas académicas que interactúan de forma sinérgica:
* **Bases de Datos (`1_bases_de_datos`)**: Almacenamiento eficiente e indexación de embeddings/textos.
* **Probabilidad y Estadística (`2_prob_estadistica`)**: Auditoría formal del sesgo a través de pruebas de hipótesis e inferencia.
* **Machine Learning (`3_ml`)**: Entrenamiento, balanceo y predicción del comportamiento de los sistemas inteligentes.
* **Matemática Aplicada (`4_matematica_aplicada`)**: Fundamentos analíticos detrás de los algoritmos bayesianos y probabilísticos.
* **Complejidad Algorítmica (`5_complejidad`)**: Análisis riguroso del Big-O y optimización del escalado del software.
* **Programación Orientada a Objetos (`6_programacion_oo`)**: Estructuración del código de IA en clases reutilizables y mantenibles.

---

## 🛡️ Seguridad y Buenas Prácticas

Este proyecto está diseñado bajo los más altos estándares de **ciberseguridad para desarrollo en IA (AI DevSecOps)**:
* **Cero claves hardcodeadas**: Ninguna API key está almacenada en el repositorio público.
* **Uso de Dotenv**: Todas las credenciales se cargan dinámicamente desde el entorno usando `os.getenv()`.
* **Archivos grandes protegidos**: Los datasets pesados en formato `.parquet` y los archivos de checkpoints se ignoran localmente para evitar saturar el repositorio.

---

*Para dudas, comentarios o contribuciones sobre esta investigación, por favor abre un Issue en este repositorio o consulta la información contenida en el [Reporte de Estudio](file:///c:/Users/t14/Desktop/Proyecto_Integrador/proyecto_integrador/Reporte_Estudio_Proyecto_Integrador.pdf) y en el [Póster Científico](file:///c:/Users/t14/Desktop/Proyecto_Integrador/proyecto_integrador/Poster_FINAL.pdf) incluidos.*

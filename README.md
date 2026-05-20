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

La organización del proyecto sigue una estructura altamente limpia, rigurosa y consolidada, centrada en los dos pilares principales del entregable:

```directory
proyecto_integrador/
├── machine_learning/               # [ML] Core de Machine Learning y Modelado
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
│   ├── 3_deep_learning_poo/        # [POO] Implementación del clasificador bajo el paradigma de OOP
│   │   └── clasificador_nn.py      # Modelo estructurado en clases usando Keras/TensorFlow
│   ├── archive/                    # Almacén de datasets locales en formato optimizado Parquet
│   ├── data/                       # Carpeta local para los outputs intermedios y finales
│   ├── NaiveBayes_Proyecto_integrador.ipynb # Entregable Principal: Modelado y evaluación de Naive Bayes vs KNN
│   └── matriz_confusion_naive_bayes.png    # Gráfica de rendimiento del modelo final
├── complejidad_algoritmica/        # [CO] Análisis de complejidad computacional y Big-O
│   ├── Notebook.ipynb              # Notebook con escalamiento empírico y regresiones de tiempo de ejecución
│   ├── PI_lineamientos.pdf         # Rúbrica oficial del área de complejidad y pautas
│   └── *.png                       # Gráficas de escalamiento de KNN, Regresión Logística y PCA de componentes
├── .env.template                   # Plantilla segura para configurar credenciales API
├── .gitignore                      # Configuración de archivos omitidos en Git (Parquets, .env, etc.)
├── Poster_FINAL.pdf                # Póster científico resumido de la investigación
├── README.md                       # Documento de presentación principal (este archivo)
└── Reporte_Estudio_Proyecto_Integrador.pdf # Reporte final consolidado escrito del proyecto
```

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
   Ejecuta `machine_learning/2_preparacion_datos/clasificador_balance.py` para realizar la descarga estratificada desde ShareChat, aplicar Detoxify Multilingual a las cadenas de texto y generar los datasets base equilibrados por plataforma e idioma.
   
2. **Auditoría de Alineación mediante LLM**:
   Ejecuta `machine_learning/2_preparacion_datos/etiquetar_v2.py` para conectar con el servicio de Mistral AI y clasificar de manera determinista cada respuesta como `RECHAZO`, `CUMPLIMIENTO` o `CUMPLIMIENTO_CON_DISCLAIMER`.

3. **Entrenamiento y Evaluación de Clasificadores**:
   Abre y ejecuta `machine_learning/NaiveBayes_Proyecto_integrador.ipynb` en Jupyter. Aquí se entrena un clasificador **Gaussian Naive Bayes** comparándolo contra **K-Nearest Neighbors** para predecir si un prompt será cumplido o rechazado basado en variables contextuales del chat.

4. **Auditoría de Complejidad**:
   Ejecuta `complejidad_algoritmica/Notebook.ipynb` para visualizar el comportamiento empírico del tiempo de entrenamiento del modelo KNN y la Regresión Logística a medida que el tamaño de las muestras incrementa ($N$), validando las curvas frente al comportamiento asintótico teórico.

5. **Prueba POO (Deep Learning)**:
   Corre `machine_learning/3_deep_learning_poo/clasificador_nn.py` para validar una arquitectura de entrenamiento estructurada mediante el paradigma orientado a objetos (clases) utilizando Keras/TensorFlow para tareas de clasificación.

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

Este proyecto consolida de manera interdisciplinaria los pilares fundamentales de la computación científica:
* **Machine Learning (`machine_learning`)**: Entrenamiento, balanceo, predicción bayesiana, KNN e inferencia estadística aplicados a la seguridad de la IA.
* **Complejidad Algorítmica (`complejidad_algoritmica`)**: Análisis de complejidad asintótica Big-O y modelado matemático del tiempo de ejecución empírico.
* **Programación Orientada a Objetos (`machine_learning/3_deep_learning_poo`)**: Estructuración del clasificador en una clase modular aplicando abstracción y encapsulamiento.
* **Estadística Inferencial**: Aplicación rigurosa de contrastes de hipótesis ($\chi^2$) para auditar sesgos lingüísticos y de arquitectura en sistemas LLM.

---

## 🛡️ Seguridad y Buenas Prácticas

Este proyecto está diseñado bajo los más altos estándares de **ciberseguridad para desarrollo en IA (AI DevSecOps)**:
* **Cero claves hardcodeadas**: Ninguna API key está almacenada en el repositorio público.
* **Uso de Dotenv**: Todas las credenciales se cargan dinámicamente desde el entorno usando `os.getenv()`.
* **Archivos grandes protegidos**: Los datasets pesados en formato `.parquet` y los archivos de checkpoints se ignoran localmente para evitar saturar el repositorio.

---

## 📚 Referencias Bibliográficas y Científicas

Este proyecto de investigación y desarrollo se fundamenta en las siguientes referencias y publicaciones científicas:

### 1. Modelos de Lenguaje, Alineación y Seguridad (AI Safety)
* **Alineación de Modelos de Lenguaje (RLHF)**: Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C. L., Mishkin, P., ... & Lowe, R. (2022). *Training language models to follow instructions with human feedback*. Advances in Neural Information Processing Systems, 35, 27730-27744. [Artículo Científico](https://proceedings.neurips.cc/paper_files/paper/2022/hash/b1efde53be36d956074ca6724a71d293-Abstract-Conference.html)
* **Constitutional AI & Auto-Evaluación**: Bai, Y., Kadavath, S., Kundu, S., Askell, A., Kernion, J., Jones, A., ... & Kaplan, J. (2022). *Constitutional AI: Harmlessness from AI feedback*. arXiv preprint arXiv:2212.08073. [Preprint en arXiv](https://arxiv.org/abs/2212.08073)
* **API de Mistral AI**: Mistral AI Team. (2024). *Mistral Large and Small Models: High-Performance Open Models for Instruction Following and Alignment*. Mistral AI. [Sitio Oficial](https://mistral.ai)

### 2. Recolección y Clasificación de Datos (Datasets & NLP)
* **Dataset ShareChat**: Nguyen, T. (2024). *ShareChat: A Large-Scale Dataset of Multi-Turn Conversations with LLMs*. Hugging Face. [Dataset en Hugging Face](https://huggingface.co/datasets/tucnguyen/ShareChat)
* **Clasificador Multilingüe de Toxicidad (Detoxify)**: Hanu, L., & Unitary team. (2020). *Detoxify: Toxic comment classification*. GitHub. [Repositorio en GitHub](https://github.com/unitaryai/detoxify)

### 3. Modelado Predictivo e Inferencia Estadística
* **Scikit-Learn (GNB & KNN)**: Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011). *Scikit-learn: Machine learning in Python*. Journal of Machine Learning Research, 12(Oct), 2825-2830. [Artículo Científico](https://www.jmlr.org/papers/volume12/pedregosa11a/pedregosa11a.pdf)
* **Prueba de Chi-cuadrada de Pearson ($\chi^2$)**: Pearson, K. (1900). *On the criterion that a given system of deviations from the probable in the case of a correlated system of variables is such that it can be reasonably supposed to have arisen from a random sampling*. The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science, 50(302), 157-175. [Publicación Histórica](https://doi.org/10.1080/14786440009463897)

### 4. Complejidad Computacional y Big-O
* **Notación Asintótica Asignada**: Knuth, D. E. (1976). *Big Omicron and big Omega and big Theta*. ACM SIGACT News, 8(2), 18-24. [Publicación ACM](https://dl.acm.org/doi/10.1145/1008335.1008338)

### 5. Librerías y Ecosistema de Programación (Core Stack)
* **TensorFlow & Keras (Clasificador Neuronal POO)**: Abadi, M., et al. (2015). *TensorFlow: Large-scale machine learning on heterogeneous systems*. [Sitio Oficial](https://www.tensorflow.org/)
* **Pandas**: McKinney, W. (2010). *Data structures for statistical computing in Python*. Proceedings of the 9th Python in Science Conference, 51-56.
* **NumPy**: Harris, C. R., et al. (2020). *Array programming with NumPy*. Nature, 585(7825), 357-362. [Artículo en Nature](https://doi.org/10.1038/s41586-020-2649-2)

---

*Para dudas, comentarios o contribuciones sobre esta investigación, por favor abre un Issue en este repositorio o consulta la información contenida en el [Reporte de Estudio](file:///c:/Users/t14/Desktop/Proyecto_Integrador/proyecto_integrador/Reporte_Estudio_Proyecto_Integrador.pdf) y en el [Póster Científico](file:///c:/Users/t14/Desktop/Proyecto_Integrador/proyecto_integrador/Poster_FINAL.pdf) incluidos.*

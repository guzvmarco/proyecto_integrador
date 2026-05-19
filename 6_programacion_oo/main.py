import tensorflow as tf
import numpy as np
import pandas as pd
import os
from datasets import load_dataset
from huggingface_hub import login

def cargar_datos():
    ruta_archivo = "data/sharechat_sample_balanced.parquet"
    
    # 1. Gestión de descarga y balanceo (Tu lógica original)
    if not os.path.exists(ruta_archivo):
        # login(token="TU_TOKEN_AQUI") # Opcional si ya estás loggeado
        os.makedirs("data", exist_ok=True)
        
        plataformas = ["chatgpt", "perplexity", "grok", "gemini", "claude"]
        dataframes_crudos = []

        for plat in plataformas:
            ds = load_dataset("tucnguyen/ShareChat", plat, split="train")
            df_temp = ds.to_pandas()
            df_temp['source_platform'] = plat 
            dataframes_crudos.append(df_temp)

        df_completo = pd.concat(dataframes_crudos, ignore_index=True)
        
        # Filtro de roles e idiomas (English/Spanish)
        df_users = df_completo[
            (df_completo['role'] == 'user') & 
            (df_completo['detected_language_final'].isin(['English', 'Spanish']))
        ]

        # Muestreo estratificado para balancear
        def muestreo_inteligente(grupo, max_n=2000):
            n_a_extraer = min(max_n, len(grupo))
            return grupo.sample(n=n_a_extraer, random_state=42)

        df_sample = (df_users.groupby(['detected_language_final', 'source_platform'], group_keys=False)
                     .apply(muestreo_inteligente)
                     .reset_index(drop=True))
        
        df_sample.to_parquet(ruta_archivo)
    else:
        df_sample = pd.read_parquet(ruta_archivo)

    # 2. Limpieza y preparación para el modelo
    # Para que el modelo de la tarea funcione (Input shape=4), 
    # necesitamos convertir categorías a números.
    
    # Ejemplo de 4 características: idioma, plataforma, longitud de texto e índice
    idioma_map = {'English': 0, 'Spanish': 1}
    plat_map = {p: i for i, p in enumerate(["chatgpt", "perplexity", "grok", "gemini", "claude"])}

    # Convertimos a arreglos de numpy (lo que pide el profesor)
    idioma_arr = df_sample['detected_language_final'].map(idioma_map).to_numpy()
    plataforma_arr = df_sample['source_platform'].map(plat_map).to_numpy()
    longitud_text_arr = df_sample['plain_text'].str.len().to_numpy()
    msg_index_arr = df_sample['message_index'].fillna(0).to_numpy()

    # NOTA: En tu proyecto de toxicidad, aquí es donde limpiarías el texto
    return idioma_arr, plataforma_arr, longitud_text_arr, msg_index_arr


def contruir_modelo(num_classes):
    # El modelo espera 4 características de entrada según el código base
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(4,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy", # Cambiado a crossentropy para clasificación
        metrics=["accuracy"],
    )
    return model


def entrenar_modelo(idioma, plat, long, index, epochs=80, batch_size=16):
    # Creamos la matriz X uniendo los vectores
    X = np.column_stack((idioma, plat, long, index))
    # Para este ejemplo de la tarea, usamos idioma como "etiqueta" (Y)
    y = idioma 
    
    num_classes = len(np.unique(y))
    model = contruir_modelo(num_classes)
    
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)
    return history


def main():
    print("Cargando y procesando datos de ShareChat...")
    idioma, plat, long, index = cargar_datos()
    
    print(f"Datos cargados. Total de muestras: {len(idioma)}")
    
    # Ejecutamos el entrenamiento
    entrenar_modelo(idioma, plat, long, index)
    
    return "Proceso completado con éxito"


if __name__ == "__main__":
    main()


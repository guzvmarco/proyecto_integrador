# -*- coding: utf-8 -*-
import os
os.environ["PYTHONIOENCODING"] = "utf-8"

from datasets import load_dataset
from detoxify import Detoxify
import pandas as pd
from tqdm import tqdm

# --- CONFIGURACION ---
configs = ["chatgpt", "gemini", "perplexity", "claude", "grok"]
filas_es = []
TAMANO_LOTE = 64
GUARDAR_CADA = 500

# --- CARGAR MODELO ---
print("Cargando modelo de toxicidad...")
modelo_tox = Detoxify('multilingual')
print("Modelo listo")

# --- STREAMING: solo filas en espanol ---
for config in configs:
    print(f"\nProcesando {config}...")
    dataset = load_dataset(
        "tucnguyen/ShareChat",
        name=config,
        split="train",
        streaming=True
    )
    count = 0
    for fila in dataset:
        if fila["role"] == "user" and fila.get("detected_language_final") == "Spanish":
            filas_es.append({
                "url":                     fila["url"],
                "role":                    fila["role"],
                "plain_text":              fila["plain_text"],
                "message_index":           fila["message_index"],
                "detected_language_final": fila["detected_language_final"],
                "platform":                config
            })
            count += 1
        if count >= 3000:
            break
    print(f"  {count} filas en espanol recolectadas de {config}")

df_es_new = pd.DataFrame(filas_es)
print(f"\nTotal filas en espanol: {len(df_es_new)}")

# --- CLASIFICADOR DE TOXICIDAD ---
resultados = []
textos = df_es_new["plain_text"].fillna("").astype(str).tolist()
print(f"Iniciando clasificacion de {len(textos)} textos...")

for i in tqdm(range(0, len(textos), TAMANO_LOTE), desc="Clasificando toxicidad ES"):
    lote = textos[i:i + TAMANO_LOTE]
    predicciones = modelo_tox.predict(lote)
    resultados.extend(pd.DataFrame(predicciones).to_dict("records"))

    # Checkpoint cada 500 filas
    if len(resultados) % GUARDAR_CADA < TAMANO_LOTE:
        df_checkpoint = pd.concat([
            df_es_new.iloc[:len(resultados)].reset_index(drop=True),
            pd.DataFrame(resultados)
        ], axis=1)
        df_checkpoint.to_parquet("sharechat_es_checkpoint.parquet", index=False)
        print(f"  Checkpoint guardado en fila {len(resultados)}")

# --- RESULTADO FINAL ---
df_puntuaciones = pd.DataFrame(resultados)
df_es_classified = pd.concat([
    df_es_new.reset_index(drop=True),
    df_puntuaciones
], axis=1)

df_es_classified["is_toxic"] = df_es_classified["toxicity"] >= 0.1
df_es_classified["lang"] = "es"

df_es_classified.to_parquet("sharechat_es_toxicidad.parquet", index=False)

print(f"\nCompletado")
print(f"Total procesado:  {len(df_es_classified)}")
print(f"Toxicos en ES:    {df_es_classified['is_toxic'].sum()}")
print(f"Archivo guardado: sharechat_es_toxicidad.parquet")
from datasets import load_dataset
from detoxify import Detoxify
import pandas as pd
from tqdm import tqdm
import os

# --- CONFIGURACION ---
TAMANO_LOTE = 64
UMBRAL_TOXICIDAD = 0.1

# Cuanto necesitamos por plataforma e idioma
OBJETIVOS = {
    "chatgpt": {"English": 3000},   # necesitamos mas EN
    "gemini":  {"Spanish": 3000},   # necesitamos mas ES
    "grok":    {"Spanish": 3000},   # necesitamos mas ES
    "perplexity": {"Spanish": 3000} # necesitamos mas ES
}

print("Cargando modelo de toxicidad...")
modelo_tox = Detoxify('multilingual')
print("Modelo listo")

todos = []

for config, idiomas in OBJETIVOS.items():
    for idioma_hf, limite in idiomas.items():
        print(f"\nProcesando {config} | idioma: {idioma_hf} | objetivo: {limite} filas")

        dataset = load_dataset(
            "tucnguyen/ShareChat",
            name=config,
            split="train",
            streaming=True
        )

        filas_recolectadas = []
        count = 0

        for fila in dataset:
            if (fila["role"] == "user" and
                fila.get("detected_language_final") == idioma_hf):
                filas_recolectadas.append({
                    "url":                     fila["url"],
                    "role":                    fila["role"],
                    "plain_text":              fila["plain_text"],
                    "message_index":           fila["message_index"],
                    "detected_language_final": fila["detected_language_final"],
                    "platform":                config
                })
                count += 1

            if count >= limite:
                break

        print(f"  Filas recolectadas: {len(filas_recolectadas)}")

        if not filas_recolectadas:
            print(f"  No se encontraron filas. Saltando...")
            continue

        # Clasificar toxicidad
        df_temp = pd.DataFrame(filas_recolectadas)
        textos = df_temp["plain_text"].fillna("").astype(str).tolist()
        print(f"  Clasificando {len(textos)} textos...")

        resultados = []
        for i in tqdm(range(0, len(textos), TAMANO_LOTE),
                      desc=f"  {config}/{idioma_hf}"):
            lote = textos[i:i + TAMANO_LOTE]
            predicciones = modelo_tox.predict(lote)
            resultados.extend(pd.DataFrame(predicciones).to_dict("records"))

            # Checkpoint parcial cada 500 filas
            if len(resultados) % 500 < TAMANO_LOTE:
                df_check = pd.concat([
                    df_temp.iloc[:len(resultados)].reset_index(drop=True),
                    pd.DataFrame(resultados)
                ], axis=1)
                df_check.to_parquet(
                    f"checkpoint_{config}_{idioma_hf}.parquet",
                    index=False
                )
                print(f"  Checkpoint guardado: {len(resultados)} filas")

        # Resultado final para esta plataforma/idioma
        df_puntuaciones = pd.DataFrame(resultados)
        df_result = pd.concat([
            df_temp.reset_index(drop=True),
            df_puntuaciones
        ], axis=1)
        df_result["is_toxic"] = df_result["toxicity"] >= UMBRAL_TOXICIDAD
        df_result["lang"] = "en" if idioma_hf == "English" else "es"

        toxicos = df_result["is_toxic"].sum()
        print(f"  Toxicos encontrados: {toxicos} / {len(df_result)}")

        todos.append(df_result)

        # Guardar parcial por plataforma
        df_result.to_parquet(
            f"sharechat_{config}_{idioma_hf}_nuevo.parquet",
            index=False
        )

# Combinar todo
if todos:
    df_final = pd.concat(todos, ignore_index=True)
    df_final.to_parquet("sharechat_nuevo_balance.parquet", index=False)

    print(f"\nCompletado!")
    print(f"Total procesado:  {len(df_final)}")
    print(f"Toxicos totales:  {df_final['is_toxic'].sum()}")
    print("\nToxicos por plataforma e idioma:")
    print(df_final[df_final["is_toxic"]].groupby(["platform", "lang"]).size())
else:
    print("No se recolectaron datos.")
import pandas as pd
import json

# --- Rutas ---
input_file = "datasets/val_chats.parquet"   # Cambia a tu archivo
output_file = "datasets/val_chats.jsonl"    # Salida JSONL

# --- Cargar parquet ---
df = pd.read_parquet(input_file)

# --- Revisar la primera fila ---
print(df['chat'].iloc[0])

# --- Procesar cada fila ---
with open(output_file, "w", encoding="utf-8") as f_out:
    for _, row in df.iterrows():
        chat_data = row['chat']

        # chat_data es normalmente lista de dicts [{"texto","role"}, {...}]
        # Tomamos el primer valor de cada dict
        try:
            user_text = list(chat_data[0].values())[0]
            assistant_text = list(chat_data[1].values())[0]
        except Exception as e:
            print(f"Error en fila {row.name}: {e}")
            continue

        record = {
            "User": user_text,
            "Assistant": assistant_text
        }

        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Archivo convertido: {output_file}")

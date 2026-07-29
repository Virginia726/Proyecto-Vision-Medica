import os
import pandas as pd

DIR_BASE = r"C:\Users\Virginia\Desktop\Vision_Medica1"
DIR_IMAGENES = os.path.join(DIR_BASE, "data", "clasificacion", "images")
DIR_DESTINO = os.path.join(DIR_BASE, "data", "clasificacion")
RUTA_CSV = os.path.join(DIR_DESTINO, "labels.csv")

# Crear carpetas si no existen
os.makedirs(DIR_DESTINO, exist_ok=True)

if not os.path.exists(DIR_IMAGENES):
    print(f"Error: No se encontró la carpeta de imágenes en: {DIR_IMAGENES}")
else:
    # Obtener todas las imágenes con prefijo CHNCXR o cualquier formato de imagen
    imagenes = [f for f in os.listdir(DIR_IMAGENES) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    registros = []
    for idx, img in enumerate(imagenes):
        # Asignar etiqueta según sufijo: _1 = Patológico (Neumonía/TBC), _0 = Sano
        es_patologico = 1 if img.endswith("_1.png") or "_1" in img else 0
        es_sano = 1 if es_patologico == 0 else 0
        
        registros.append({
            "Image_Index": img,
            "Patient_ID": f"PACIENT_{idx+1:04d}",
            "Pneumonia": es_patologico,
            "Atelectasis": 0,
            "Effusion": 0,
            "No Finding": es_sano
        })

    df = pd.DataFrame(registros)
    df.to_csv(RUTA_CSV, index=False, encoding='utf-8')
    print(f"¡Éxito! Se creó '{RUTA_CSV}' con {len(df)} registros.")
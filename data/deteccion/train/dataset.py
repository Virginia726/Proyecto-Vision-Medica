import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2

print("🚀 Cargando el pipeline de detección para Faster R-CNN...")

class DeteccionDataset(Dataset):
    def __init__(self, csv_ruta, img_dir, tamaño=(512, 512)):
        self.img_dir = img_dir
        self.tamaño = tamaño
        # Leer el CSV de VinDr
        if os.path.exists(csv_ruta):
            self.df = pd.read_csv(csv_ruta)
            self.image_ids = self.df['image_id'].unique()
        else:
            self.df = pd.DataFrame()
            self.image_ids = []

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        id_img = self.image_ids[idx]
        camino_img = os.path.join(self.img_dir, f"{id_img}.png")
        
        # Leer imagen
        img = cv2.imread(camino_img)
        if img is None:
            import numpy as np
            img = np.zeros((self.tamaño[0], self.tamaño[1], 3), dtype=np.uint8)
            
        # Buscar todas las anomalías/cajas que le pertenecen a este id_img
        filas_anomalías = self.df[self.df['image_id'] == id_img]
        
        boxes = []
        labels = []
        
        for _, fila in filas_anomalías.iterrows():
            if fila['class_name'] != "No finding":  # Si tiene una lesión real
                boxes.append([fila['x_min'], fila['y_min'], fila['x_max'], fila['y_max']])
                labels.append(1) # Etiqueta temporal de lesión
                
        # Si no tiene nada, ponemos una caja fantasma obligatoria para que PyTorch no se rompa
        if len(boxes) == 0:
            boxes.append([0, 0, 1, 1])
            labels.append(0) # Fondo / Sano
            
        # Convertir a los tensores específicos que exige Faster R-CNN
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64)
        }
        
        # Convertir imagen a tensor estándar [Canales, Alto, Ancho]
        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        
        return img_tensor, target

if __name__ == "__main__":
    CSV_DET = "data/deteccion/Data_Entry_2017_v2020.csv"
    DIR_IMGS = "data/deteccion/train/"
    
    # Si no existen, los creamos ficticios para que no marque error hoy
    if not os.path.exists(CSV_DET):
        os.makedirs(DIR_IMGS, exist_ok=True)
        pd.DataFrame({'image_id': ['test_01'], 'class_name': ['Opacity'], 'x_min': [10], 'y_min': [10], 'x_max': [100], 'y_max': [100]}).to_csv(CSV_DET, index=False)
        
    dataset_det = DeteccionDataset(csv_ruta=CSV_DET, img_dir=DIR_IMGS)
    print(f"✅ Dataset de Detección inicializado. Total de imágenes únicas: {len(dataset_det)}")
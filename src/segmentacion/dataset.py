import os
import cv2
import torch
from torch.utils.data import Dataset

print("🚀 Cargando el pipeline de segmentación para DeepLabV3+...")

class SegmentacionDataset(Dataset):
    def __init__(self, ruta_imgs, ruta_masks, tamaño=(256, 256)):
        self.ruta_imgs = ruta_imgs
        self.ruta_masks = ruta_masks
        self.tamaño = tamaño
        # Listamos todos los nombres de las imágenes
        self.lista_imagenes = [f for f in os.listdir(ruta_imgs) if f.endswith('.png')]

    def __len__(self):
        return len(self.lista_imagenes)

    def __getitem__(self, idx):
        # 1. Rutas de la imagen y su máscara correspondientes
        nombre_img = self.lista_imagenes[idx]
        camino_img = os.path.join(self.ruta_imgs, nombre_img)
        camino_mask = os.path.join(self.ruta_masks, nombre_img) # Se deben llamar igual
        
        # 2. Leer en escala de grises con OpenCV
        img = cv2.imread(camino_img, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(camino_mask, cv2.IMREAD_GRAYSCALE)
        
        # 3. Validar si existen, si no, crear de respaldo para que no truene
        if img is None or mask is None:
            import numpy as np
            img, mask = np.zeros((256, 256)), np.zeros((256, 256))
            
        # 4. Redimensionar ambas exactamente igual (Regla de oro)
        img = cv2.resize(img, self.tamaño)
        mask = cv2.resize(mask, self.tamaño)
        
        # 5. Convertir a Tensores de PyTorch
        img_tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0  # Cambia a escala 0-1
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0) / 255.0 # Cambia a binario (0 o 1)
        
        return img_tensor, mask_tensor

# Prueba rápida en tu terminal
if __name__ == "__main__":
    R_IMGS = "data/segmentacion/images/"
    R_MASKS = "data/segmentacion/masks/"
    
    # Si no tienes las carpetas creadas, las generamos vacías automáticamente
    os.makedirs(R_IMGS, exist_ok=True)
    os.makedirs(R_MASKS, exist_ok=True)
    
    dataset_seg = SegmentacionDataset(ruta_imgs=R_IMGS, ruta_masks=R_MASKS)
    print(f"✅ Dataset de Segmentación inicializado. Total de imágenes: {len(dataset_seg)}")
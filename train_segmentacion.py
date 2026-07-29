import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models.segmentation as segmentation

# ==============================================================================
# 1. CONFIGURACIÓN DE SEMILLA Y REPRODUCIBILIDAD
# ==============================================================================
def fijar_semilla(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

SEMILLA = 42
fijar_semilla(SEMILLA)

# ==============================================================================
# 2. CLASE DE TRANSFORMACIONES (CON INTERPOLACIÓN NEAREST Y VERIFICACIÓN)
# ==============================================================================
class TransformacionSegmentacion:
    def __init__(self, size=(256, 256)):
        self.size = size
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )

    def __call__(self, imagen, mascara):
        # Redimensionamiento usando NEAREST para preservar la máscara binaria
        imagen = transforms.functional.resize(imagen, self.size)
        mascara = transforms.functional.resize(
            mascara, 
            self.size, 
            interpolation=transforms.InterpolationMode.NEAREST
        )

        imagen = transforms.functional.to_tensor(imagen)
        imagen = self.normalize(imagen)

        mascara = transforms.functional.to_tensor(mascara)
        mascara = (mascara > 0.5).float() # Binarización estricta

        # Verificación de valores únicos en la máscara
        valores_unicos = torch.unique(mascara).tolist()
        assert all(v in [0.0, 1.0] for v in valores_unicos), \
            f"Error: La máscara contiene valores no binarios: {valores_unicos}"

        return imagen, mascara

# ==============================================================================
# 3. CLASE DEL DATASET (SIN DATOS SINTÉTICOS Y CON SUFIJO _mask.png)
# ==============================================================================
class DatasetSegmentacionTorax(Dataset):
    def __init__(self, lista_imagenes, dir_imagenes, dir_mascaras, transform=None):
        self.lista_imagenes = lista_imagenes
        self.dir_imagenes = dir_imagenes
        self.dir_mascaras = dir_mascaras
        self.transform = transform
        self.muestras_validas = self._validar_emparejamiento()

    def _validar_emparejamiento(self):
        """
        Valida la existencia física de cada radiografía y su correspondiente máscara.
        Interrumpe explícitamente sin generar datos sintéticos vacíos.
        """
        validas = []
        for nombre_img in self.lista_imagenes:
            ruta_img = os.path.join(self.dir_imagenes, nombre_img)
            
            # Construcción del nombre de la máscara agregando _mask.png
            nombre_mask = nombre_img.replace(".png", "_mask.png")
            ruta_mask = os.path.join(self.dir_mascaras, nombre_mask)

            if not os.path.exists(ruta_img):
                raise FileNotFoundError(f"[ERROR CRÍTICO] Radiografía no encontrada: {ruta_img}")
            
            if not os.path.exists(ruta_mask):
                raise FileNotFoundError(
                    f"\n[ERROR CRÍTICO] Máscara faltante para '{nombre_img}'.\n"
                    f"Ruta buscada: {ruta_mask}"
                )

            validas.append((ruta_img, ruta_mask))
        return validas

    def __getitem__(self, idx):
        ruta_img, ruta_mask = self.muestras_validas[idx]
        
        imagen = Image.open(ruta_img).convert("RGB")
        mascara = Image.open(ruta_mask).convert("L")

        if self.transform:
            imagen, mascara = self.transform(imagen, mascara)

        return imagen, mascara

    def __len__(self):
        return len(self.muestras_validas)

# ==============================================================================
# 4. CONFIGURACIÓN DE RUTAS Y PROCESAMIENTO MÚLTIPLE DE IMÁGENES
# ==============================================================================
BASE_DIR = r"C:\Users\Virginia\Desktop\Vision_Medica1\data\segmentacion"
DIR_IMAGENES = os.path.join(BASE_DIR, "images")
DIR_MASCARAS = os.path.join(BASE_DIR, "masks")

if not os.path.exists(DIR_IMAGENES):
    raise FileNotFoundError(f"[ERROR] La carpeta de imágenes no existe en: {DIR_IMAGENES}")

# 1. Detectar TODOS los archivos CHNCXR disponibles
todos_los_archivos = os.listdir(DIR_IMAGENES)
archivos_radiografias = sorted([f for f in todos_los_archivos if f.startswith('CHNCXR') and f.endswith('.png')])

total_imagenes = len(archivos_radiografias)
print(f" Total de imágenes CHNCXR detectadas: {total_imagenes}")

if total_imagenes == 0:
    raise ValueError(f"[ERROR] No se encontraron archivos que inicien con CHNCXR en {DIR_IMAGENES}")

# 2. División automática (80% Entrenamiento / 20% Validación)
limite_train = int(total_imagenes * 0.8)

# Garantizar al menos 1 muestra de validación en conjuntos pequeños
if limite_train == total_imagenes:
    limite_train = total_imagenes - 1

NOMBRES_IMAGENES_TRAIN = archivos_radiografias[:limite_train]
NOMBRES_IMAGENES_VAL = archivos_radiografias[limite_train:]

print(f" Imágenes para Entrenamiento ({len(NOMBRES_IMAGENES_TRAIN)}): {NOMBRES_IMAGENES_TRAIN}")
print(f" Imágenes para Validación ({len(NOMBRES_IMAGENES_VAL)}): {NOMBRES_IMAGENES_VAL}")

# 3. Datasets y DataLoaders
transformador = TransformacionSegmentacion(size=(256, 256))

train_dataset = DatasetSegmentacionTorax(
    lista_imagenes=NOMBRES_IMAGENES_TRAIN,
    dir_imagenes=DIR_IMAGENES,
    dir_mascaras=DIR_MASCARAS,
    transform=transformador
)

val_dataset = DatasetSegmentacionTorax(
    lista_imagenes=NOMBRES_IMAGENES_VAL,
    dir_imagenes=DIR_IMAGENES,
    dir_mascaras=DIR_MASCARAS,
    transform=transformador
)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# ==============================================================================
# 5. INICIALIZACIÓN DEL MODELO DEEPLABV3
# ==============================================================================
modelo = segmentation.deeplabv3_resnet50(
    weights=None, 
    weights_backbone=None, 
    num_classes=2
)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(modelo.parameters(), lr=1e-4)

# ==============================================================================
# 6. CICLO ÚNICO DE ENTRENAMIENTO Y EVALUACIÓN
# ==============================================================================
epocas = 3
historial_metricas = []

print(f"\n--- Inicio de Entrenamiento DeepLabV3 (Semilla fija: {SEMILLA}) ---")

for epoca in range(1, epocas + 1):
    # --- FASE DE ENTRENAMIENTO ---
    modelo.train()
    # Congelar BatchNorm para evitar problemas de estabilidad con lotes pequeños
    for m in modelo.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

    perdida_train_acumulada = 0.0
    for imagenes, mascaras in train_loader:
        optimizer.zero_grad()
        salida = modelo(imagenes)['out']
        loss = criterion(salida, mascaras)
        loss.backward()
        optimizer.step()
        perdida_train_acumulada += loss.item()
    
    perdida_train = perdida_train_acumulada / len(train_loader)

    # --- FASE DE VALIDACIÓN ---
    modelo.eval()
    perdida_val_acumulada = 0.0
    dice_acumulado = 0.0
    iou_acumulado = 0.0

    with torch.no_grad():
        for imagenes, mascaras in val_loader:
            salida = modelo(imagenes)['out']
            loss = criterion(salida, mascaras)
            perdida_val_acumulada += loss.item()

            # Cálculo de métricas anatómicas
            preds = (torch.sigmoid(salida) > 0.5).float()
            intersection = (preds * mascaras).sum()
            union = preds.sum() + mascaras.sum() - intersection

            dice = (2. * intersection) / (preds.sum() + mascaras.sum() + 1e-8)
            iou = intersection / (union + 1e-8)

            dice_acumulado += dice.item()
            iou_acumulado += iou.item()

    perdida_val = perdida_val_acumulada / len(val_loader)
    dice_promedio = dice_acumulado / len(val_loader)
    iou_promedio = iou_acumulado / len(val_loader)

    # Registro histórico único
    historial_metricas.append({
        'epoca': epoca,
        'train_loss': perdida_train,
        'val_loss': perdida_val,
        'val_dice': dice_promedio,
        'val_iou': iou_promedio
    })

    # Impresión única por época con precisión de 6 decimales
    print(f"Época [{epoca:02d}/{epocas:02d}] | "
          f"Train Loss: {perdida_train:.6f} | "
          f"Val Loss: {perdida_val:.6f} | "
          f"Val Dice: {dice_promedio:.6f} | "
          f"Val IoU: {iou_promedio:.6f}")

# Guardado con nomenclatura oficial
torch.save(modelo.state_dict(), "modelo_deeplabv3_segmentacion.pth")
print("\nEntrenamiento finalizado exitosamente y modelo guardado como 'modelo_deeplabv3_segmentacion.pth'.")
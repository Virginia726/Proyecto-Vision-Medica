import os
from PIL import Image, ImageDraw
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# ==============================================================================
# 1. CONFIGURACIÓN DE SEMILLA Y REPRODUCIBILIDAD
# ==============================================================================
def fijar_semilla(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

SEMILLA = 42
fijar_semilla(SEMILLA)

# ==============================================================================
# 2. DATASET PARA SIMULACIÓN DE DETECCIÓN (EXTRACCIÓN BEX)
# ==============================================================================
class DatasetDeteccionTorax(Dataset):
    """
    Dataset que extrae las coordenadas de la caja delimitadora [xmin, ymin, xmax, ymax]
    a partir del contorno de la máscara binaria para alimentar el simulador didáctico.
    """
    def __init__(self, pares_imagenes_mascaras, dir_imagenes, dir_mascaras, transform=None):
        self.pares = pares_imagenes_mascaras
        self.dir_imagenes = dir_imagenes
        self.dir_mascaras = dir_mascaras
        self.transform = transform
        self.muestras_validas = self._extraer_cajas()

    def _extraer_cajas(self):
        validas = []

        for nombre_img, nombre_mask in self.pares:
            ruta_img = os.path.join(self.dir_imagenes, nombre_img)
            ruta_mask = os.path.join(self.dir_mascaras, nombre_mask)

            # Extraer las coordenadas [xmin, ymin, xmax, ymax]
            mask_pil = Image.open(ruta_mask).convert("L")
            mask_np = np.array(mask_pil) > 127

            posiciones = np.argwhere(mask_np)
            if posiciones.size > 0:
                ymin, xmin = posiciones.min(axis=0)
                ymax, xmax = posiciones.max(axis=0)
                
                # Normalización relativa (0.0 a 1.0) respecto a las dimensiones de la imagen
                h, w = mask_np.shape
                bbox = [xmin / w, ymin / h, xmax / w, ymax / h]
                etiqueta_clase = 1.0  # Con presencia de región anatómica / afección
            else:
                bbox = [0.0, 0.0, 0.0, 0.0]
                etiqueta_clase = 0.0  # Sin región detectada

            validas.append((ruta_img, ruta_mask, bbox, etiqueta_clase))

        return validas

    def __len__(self):
        return len(self.muestras_validas)

    def __getitem__(self, idx):
        ruta_img, _, bbox, etiqueta_clase = self.muestras_validas[idx]
        imagen = Image.open(ruta_img).convert("RGB")

        if self.transform:
            imagen = self.transform(imagen)

        target_bbox = torch.tensor(bbox, dtype=torch.float32)
        target_clase = torch.tensor([etiqueta_clase], dtype=torch.float32)

        return imagen, target_clase, target_bbox


# ==============================================================================
# 3. ARQUITECTURA: SIMULADOR DIDÁCTICO MULTITAREA (CNN + CABEZALES DIRECTOS)
# ==============================================================================
class RedDeteccionSimplificada(nn.Module):
    """
    Red Convolucional simplificada para el aprendizaje multitarea.
    NO contiene RPN, Anchors ni RoI Align. Predice directamente la clase y 4 coordenadas.
    """
    def __init__(self):
        super(RedDeteccionSimplificada, self).__init__()
        # Backbone ResNet18 sin pesos preentrenados automáticos de internet
        backbone = models.resnet18(weights=None)
        num_ftrs = backbone.fc.in_features
        backbone.fc = nn.Identity()  # Extractor de características
        self.backbone = backbone

        # Cabezal 1: Clasificación binaria
        self.head_clasificacion = nn.Linear(num_ftrs, 1)
        
        # Cabezal 2: Regresión directa de coordenadas [xmin, ymin, xmax, ymax]
        self.head_regresion = nn.Sequential(
            nn.Linear(num_ftrs, 4),
            nn.Sigmoid()  # Restringe coordenadas al rango [0, 1]
        )

    def forward(self, x):
        features = self.backbone(x)
        out_clase = self.head_clasificacion(features)
        out_bbox = self.head_regresion(features)
        return out_clase, out_bbox


# ==============================================================================
# 4. CONFIGURACIÓN DE RUTAS Y Emparejamiento por posición ordinal
# ==============================================================================
BASE_DIR = r"C:\Users\Virginia\Desktop\Vision_Medica1\data\segmentacion"
DIR_IMAGENES = os.path.join(BASE_DIR, "images")
DIR_MASCARAS = os.path.join(BASE_DIR, "masks")

if not os.path.exists(DIR_IMAGENES) or not os.path.exists(DIR_MASCARAS):
    raise FileNotFoundError(f"[ERROR] Verifique que existan las carpetas:\n{DIR_IMAGENES}\n{DIR_MASCARAS}")

archivos_imagenes = sorted([f for f in os.listdir(DIR_IMAGENES) if f.startswith('CHNCXR') and f.endswith('.png')])
archivos_mascaras = sorted([m for m in os.listdir(DIR_MASCARAS) if m.startswith('CHNCXR')])

# Emparejar tomando hasta el mínimo número de archivos entre ambas carpetas
num_muestras = min(len(archivos_imagenes), len(archivos_mascaras))

pares_validos = list(zip(archivos_imagenes[:num_muestras], archivos_mascaras[:num_muestras]))
total_pareadas = len(pares_validos)

print(f" Total de pares (Imagen + Máscara) válidos emparejados: {total_pareadas}")

if total_pareadas == 0:
    raise ValueError(
        f"[ERROR] No se encontraron archivos para procesar en {DIR_IMAGENES} o {DIR_MASCARAS}."
    )

# División 80% Entrenamiento / 20% Validación
limite_train = int(total_pareadas * 0.8)
if limite_train == total_pareadas:
    limite_train = total_pareadas - 1

PARES_TRAIN = pares_validos[:limite_train]
PARES_VAL = pares_validos[limite_train:]

print(f" Muestras para Entrenamiento ({len(PARES_TRAIN)}): {[p[0] for p in PARES_TRAIN[:3]]}...")
print(f" Muestras para Validación ({len(PARES_VAL)}): {[p[0] for p in PARES_VAL[:3]]}...")

# Transformaciones de imagen
transformador_img = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = DatasetDeteccionTorax(PARES_TRAIN, DIR_IMAGENES, DIR_MASCARAS, transform=transformador_img)
val_dataset = DatasetDeteccionTorax(PARES_VAL, DIR_IMAGENES, DIR_MASCARAS, transform=transformador_img)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)


# ==============================================================================
# 5. INICIALIZACIÓN DE MODELO, PÉRDIDAS MULTITAREA Y OPTIMIZADOR
# ==============================================================================
modelo = RedDeteccionSimplificada()

criterion_clase = nn.BCEWithLogitsLoss()
criterion_bbox = nn.MSELoss()  # Regresión de coordenadas L2

optimizer = torch.optim.Adam(modelo.parameters(), lr=1e-4)


# ==============================================================================
# 6. CICLO ÚNICO DE ENTRENAMIENTO Y EVALUACIÓN
# ==============================================================================
epocas = 3
lambda_reg = 1.0  # Ponderación para la pérdida de regresión

print(f"\n--- Inicio de Entrenamiento: Simulador Didáctico de Detección (Semilla: {SEMILLA}) ---")

for epoca in range(1, epocas + 1):
    # --- FASE DE ENTRENAMIENTO ---
    modelo.train()
    # Congelar BatchNorm para mantener estabilidad estadística con lotes pequeños
    for m in modelo.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

    train_loss_acumulada = 0.0
    for imagenes, target_clase, target_bbox in train_loader:
        optimizer.zero_grad()
        
        out_clase, out_bbox = modelo(imagenes)
        
        loss_cls = criterion_clase(out_clase, target_clase)
        loss_box = criterion_bbox(out_bbox, target_bbox)
        
        # Pérdida total combinada
        loss_total = loss_cls + (lambda_reg * loss_box)
        
        loss_total.backward()
        optimizer.step()
        
        train_loss_acumulada += loss_total.item()

    train_loss_prom = train_loss_acumulada / len(train_loader)

    # --- FASE DE VALIDACIÓN ---
    modelo.eval()
    val_loss_acumulada = 0.0
    val_loss_cls_acumulada = 0.0
    val_loss_box_acumulada = 0.0

    with torch.no_grad():
        for imagenes, target_clase, target_bbox in val_loader:
            out_clase, out_bbox = modelo(imagenes)
            
            loss_cls = criterion_clase(out_clase, target_clase)
            loss_box = criterion_bbox(out_bbox, target_bbox)
            loss_total = loss_cls + (lambda_reg * loss_box)

            val_loss_acumulada += loss_total.item()
            val_loss_cls_acumulada += loss_cls.item()
            val_loss_box_acumulada += loss_box.item()

    val_loss_prom = val_loss_acumulada / len(val_loader)
    val_cls_prom = val_loss_cls_acumulada / len(val_loader)
    val_box_prom = val_loss_box_acumulada / len(val_loader)

    print(f"Época [{epoca:02d}/{epocas:02d}] | "
          f"Train Loss Total: {train_loss_prom:.6f} | "
          f"Val Loss Total: {val_loss_prom:.6f} | "
          f"Val Loss Clase: {val_cls_prom:.6f} | "
          f"Val Loss Box (MSE): {val_box_prom:.6f}")

# Guardado del modelo entrenado
torch.save(modelo.state_dict(), "modelo_deteccion_simulador.pth")
print("\nEntrenamiento finalizado y modelo guardado como 'modelo_deteccion_simulador.pth'.")


# ==============================================================================
# 7. GENERACIÓN DE FIGURA CON CAJA DELIMITADORA DE REFERENCIA (PARA EL REPORTE)
# ==============================================================================
print("\nGenerando figura con caja delimitadora de referencia...")

# Obtener primera muestra del conjunto de entrenamiento
ruta_img_ej, ruta_mask_ej, _, _ = train_dataset.muestras_validas[0]

img_ej = Image.open(ruta_img_ej).convert("RGB")
mask_ej = Image.open(ruta_mask_ej).convert("L")

# Extraer coordenadas [xmin, ymin, xmax, ymax] en píxeles absolutos
mask_np = np.array(mask_ej) > 127
posiciones = np.argwhere(mask_np)

if posiciones.size > 0:
    ymin, xmin = posiciones.min(axis=0)
    ymax, xmax = posiciones.max(axis=0)

    # Dibujar la caja verde de Ground Truth
    draw = ImageDraw.Draw(img_ej)
    draw.rectangle([xmin, ymin, xmax, ymax], outline="green", width=4)

    # Guardar imagen resultante en el directorio actual
    nombre_salida_figura = "figura_radiografia_caja_delimitadora.png"
    img_ej.save(nombre_salida_figura)
    print(f"Figura guardada exitosamente: '{nombre_salida_figura}' [xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}]")
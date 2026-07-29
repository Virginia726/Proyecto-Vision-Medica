import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from PIL import Image

# ==========================================
# 1. CONFIGURACIÓN DE RUTAS Y HIPERPARÁMETROS
# ==========================================
DIR_BASE = r"C:\Users\Virginia\Desktop\Vision_Medica1"
DIR_IMAGENES = os.path.join(DIR_BASE, "data", "clasificacion", "images")
RUTA_CSV = os.path.join(DIR_BASE, "data", "clasificacion", "labels.csv")

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. DATASET PERSONALIZADO
# ==========================================
class ChestXRayDataset(Dataset):
    def __init__(self, df, dir_imagenes, transform=None):
        self.df = df.reset_index(drop=True)
        self.dir_imagenes = dir_imagenes
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['Image_Index']
        img_path = os.path.join(self.dir_imagenes, img_name)

        # Cargar imagen en escala de grises y convertir a RGB
        image = Image.open(img_path).convert('RGB')
        
        # Determinar etiqueta binaria (Pneumonia/Patológico vs No Finding/Sano)
        if 'Pneumonia' in row:
            label = torch.tensor(row['Pneumonia'], dtype=torch.float32)
        else:
            label = torch.tensor(0.0, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label

# ==========================================
# 3. CARGA Y PREPARACIÓN DE DATOS
# ==========================================
def cargar_y_preparar_datos(ruta_csv, dir_imagenes):
    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(f"No se encontró el archivo CSV en: {ruta_csv}")

    # Lectura del CSV con tolerancia a delimitadores y líneas corruptas
    df = pd.read_csv(ruta_csv, sep=None, engine='python', on_bad_lines='skip')

    # Verificar existencia real de imágenes en el disco
    df['existe'] = df['Image_Index'].apply(lambda x: os.path.exists(os.path.join(dir_imagenes, x)))
    df = df[df['existe']].drop(columns=['existe'])

    if len(df) == 0:
        raise ValueError(f"No se encontraron imágenes válidas en la ruta: {dir_imagenes}")

    # Separar en entrenamiento (80%) y validación (20%)
    df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)
    return df_train, df_val

# ==========================================
# 4. BUCLE DE ENTRENAMIENTO PRINCIPAL
# ==========================================
def main():
    print(f"Usando dispositivo: {DEVICE}")
    print("Cargando y preparando datos...")
    
    df_train, df_val = cargar_y_preparar_datos(RUTA_CSV, DIR_IMAGENES)
    print(f"Imágenes de entrenamiento: {len(df_train)} | Imágenes de validación: {len(df_val)}")

    # Transformaciones para PyTorch
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Crear DataLoaders
    train_dataset = ChestXRayDataset(df_train, DIR_IMAGENES, transform=transform_train)
    val_dataset = ChestXRayDataset(df_val, DIR_IMAGENES, transform=transform_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Modelo ResNet18 preentrenado
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)  # Salida binaria
    model = model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nIniciando entrenamiento...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_dataset)

        # Validación
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE).unsqueeze(1)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                preds = torch.sigmoid(outputs) >= 0.5
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_epoch_loss = val_loss / len(val_dataset)
        val_acc = (correct / total) * 100 if total > 0 else 0

        print(f"Época [{epoch+1}/{EPOCHS}] | Train Loss: {epoch_loss:.4f} | Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_acc:.2f}%")

    # Guardar pesos entrenados
    os.makedirs(os.path.join(DIR_BASE, "models"), exist_ok=True)
    ruta_guardado = os.path.join(DIR_BASE, "models", "modelo_clasificacion.pth")
    torch.save(model.state_dict(), ruta_guardado)
    print(f"\n¡Entrenamiento completado! Modelo guardado en: {ruta_guardado}")

if __name__ == "__main__":
    main()
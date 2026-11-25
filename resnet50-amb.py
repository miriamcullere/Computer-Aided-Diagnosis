import ssl
# 1. PARCHE SSL PARA MAC
ssl._create_default_https_context = ssl._create_unverified_context

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from torchvision import models, transforms, datasets
import time
import copy
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import cv2
import random
from PIL import Image
from skimage.morphology import remove_small_objects 

# ==========================================
# 2. CONFIGURACIÓN
# ==========================================
CONFIG = {
    'train_dir': '/Users/miriamcullere/Library/CloudStorage/OneDrive-UniversitatdeGirona/MMIC/CAD/PRÀCTICA_MELASMA/CHALLENGE1/train',
    'val_dir': '/Users/miriamcullere/Library/CloudStorage/OneDrive-UniversitatdeGirona/MMIC/CAD/PRÀCTICA_MELASMA/CHALLENGE1/val', 
    'num_classes': 2,
    'batch_size': 32,
    'num_epochs': 8,
    'learning_rate': 0.001,
    'max_train_samples': 2000, 
    'max_val_samples': 2000 
}

def get_device():
    if torch.backends.mps.is_available():
        print("--> Usando GPU Apple M2 (MPS)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        print("--> Usando CPU")
        return torch.device("cpu")

DEVICE = get_device()

# ==========================================
# 3. FUNCIONES DE PREPROCESADO (OPENCV)
# ==========================================

def remove_hairs(image):
    """
    Elimina pelos usando BlackHat morphology e inpainting.
    Input: Imagen RGB (numpy array)
    """
    # Convertir a grises (OpenCV maneja BGR/RGB según contexto, aquí asumimos entrada RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, None, iterations=1)
    
    # remove_small_objects requiere bool
    mask_bool = remove_small_objects(mask.astype(bool), min_size=30)
    mask = mask_bool.astype(np.uint8) * 255
    
    # Inpaint
    inpainted = cv2.inpaint(image, mask, 1, cv2.INPAINT_TELEA)
    return inpainted

def crop_lesion(image):
    """
    Recorta la lesión basándose en contornos y aplica un zoom del 5%.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        
        # ZOOM EXTRA (5%)
        crop_factor = 0.05
        diff_w = int(w * crop_factor)
        diff_h = int(h * crop_factor)
        
        new_x = x + diff_w
        new_y = y + diff_h
        new_w = w - (2 * diff_w)
        new_h = h - (2 * diff_h)
        
        if new_w > 0 and new_h > 0:
            return image[new_y:new_y+new_h, new_x:new_x+new_w]
            
        return image[y:y+h, x:x+w]
        
    return image

# ==========================================
# 4. DATASET PERSONALIZADO (CON LIMITADOR)
# ==========================================

class LesionDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_samples=None):
        self.image_folder = datasets.ImageFolder(root_dir)
        self.classes = self.image_folder.classes
        self.samples = self.image_folder.samples 
        self.transform = transform

        # --- LIMITADOR DE IMÁGENES ---
        if max_samples is not None:
            # Mezclar antes de recortar para mantener balance de clases
            # Usamos semilla fija para que la prueba sea reproducible
            random.seed(42) 
            random.shuffle(self.samples)
            
            # Aplicar el corte
            original_len = len(self.samples)
            self.samples = self.samples[:max_samples]
            print(f"--> [DEBUG Dataset] Reducido de {original_len} a {len(self.samples)} imágenes.")
        # -----------------------------

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        
        # 1. Cargar con OpenCV
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"No se pudo cargar la imagen: {path}")
        
        # Convertir BGR a RGB (OpenCV carga en BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. Preprocesado (Hair removal + Crop)
        image = remove_hairs(image)
        image = crop_lesion(image)
        
        # 3. Convertir a PIL para transforms de PyTorch
        image = Image.fromarray(image)
        
        # 4. Transformaciones (Resize, Normalize, etc.)
        if self.transform:
            image = self.transform(image)
            
        return image, label

# ==========================================
# 5. CARGA DE DATOS
# ==========================================
print("Procesando imágenes...")

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

try:
    # Usamos LesionDataset en lugar de ImageFolder
    image_datasets = {
        'train': LesionDataset(CONFIG['train_dir'], transform=data_transforms['train'], max_samples=CONFIG['max_train_samples']),
        'val': LesionDataset(CONFIG['val_dir'], transform=data_transforms['val'], max_samples=CONFIG['max_val_samples'])
    }
    
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0),
        'val': DataLoader(image_datasets['val'], batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
    }
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    
    print(f"Resumen Final: {dataset_sizes['train']} train, {dataset_sizes['val']} val")
    print(f"Clases: {class_names}")

except Exception as e:
    print(f"\n ERROR CRÍTICO AL CARGAR DATOS: {e}")
    exit()

# ==========================================
# 6. MODELO RESNET 50
# ==========================================
print("--> Configurant ResNet50...")
model_ft = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Congelar pesos base
for param in model_ft.parameters():
    param.requires_grad = False

# Reemplazar capa final
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, CONFIG['num_classes']) 

model_ft = model_ft.to(DEVICE)
criterion = nn.CrossEntropyLoss()
# Solo optimizamos la capa final
optimizer_ft = optim.Adam(model_ft.fc.parameters(), lr=CONFIG['learning_rate'])

# ==========================================
# 7. ENTRENAMIENTO
# ==========================================
def train_model(model, criterion, optimizer, num_epochs=10):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f"\nIniciando Entrenamiento ({num_epochs} épocas)...")

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterar sobre datos
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.float() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()

    time_elapsed = time.time() - since
    print(f'Training completado en {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Mejor Val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model, history

# ==========================================
# 8. EJECUCIÓN Y VISUALIZACIÓN
# ==========================================

model_trained, history = train_model(model_ft, criterion, optimizer_ft, num_epochs=CONFIG['num_epochs'])

torch.save(model_trained.state_dict(), 'resnet50_preprocessed_subset.pth')
print("Modelo guardado como 'resnet50_preprocessed_subset.pth'")

# Gráficas
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss (ResNet50 + Preproc)')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Acc')
plt.plot(history['val_acc'], label='Val Acc')
plt.legend()
plt.title('Accuracy (ResNet50 + Preproc)')
plt.grid()
plt.savefig('resnet_learning_curves.png')
plt.show()

# Reporte
print("\nGenerando Reporte de Clasificación...")
all_preds = []
all_labels = []
model_trained.eval()
with torch.no_grad():
    for inputs, labels in dataloaders['val']:
        inputs = inputs.to(DEVICE)
        outputs = model_trained(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(classification_report(all_labels, all_preds, target_names=class_names))
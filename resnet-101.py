import ssl
# 1. PARCHE SSL PARA MAC
ssl._create_default_https_context = ssl._create_unverified_context

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms, datasets
import time
import copy
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
from PIL import Image
from skimage.morphology import remove_small_objects 

# ==========================================
# 0. INICIO CRONÓMETRO GLOBAL
# ==========================================
global_start_time = time.time()

# ==========================================
# 2. CONFIGURACIÓN
# ==========================================
CONFIG = {
    'train_dir': '/Users/miriamcullere/Desktop/MMIC/CAD/DATASET/BINARY/train',
    'val_dir': '/Users/miriamcullere/Desktop/MMIC/CAD/DATASET/BINARY/val', 
    'num_classes': 2,
    # ATENCIÓN: ResNeXt-101 es pesada. Si te da error de memoria, baja esto a 16 o 8.
    'batch_size': 16, 
    'num_epochs': 15,
    'learning_rate': 1e-4
}

def get_device():
    if torch.backends.mps.is_available():
        print("--> Usando GPU Apple M2 (MPS)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        print("--> No s'esta utilitzant GPU, s'usarà CPU")
        return torch.device("cpu")

DEVICE = get_device()

# ==========================================
# 3. FUNCIONES DE PREPROCESADO (OPENCV)
# ==========================================

def remove_hairs(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, None, iterations=1)
    
    mask_bool = remove_small_objects(mask.astype(bool), min_size=30)
    mask = mask_bool.astype(np.uint8) * 255
    
    inpainted = cv2.inpaint(image, mask, 1, cv2.INPAINT_TELEA)
    return inpainted

def crop_lesion(image):
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
# 4. DATASET PERSONALIZADO (ROBUSTO)
# ==========================================

class LesionDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_samples=None):
        self.image_folder = datasets.ImageFolder(root_dir)
        self.classes = self.image_folder.classes
        self.samples = self.image_folder.samples 
        self.transform = transform

        if max_samples is not None:
            random.seed(42) 
            random.shuffle(self.samples)
            original_len = len(self.samples)
            self.samples = self.samples[:max_samples]
            print(f"--> [DEBUG] Dataset limitado de {original_len} a {len(self.samples)} imágenes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        
        # Carga segura anti-crashes
        image = cv2.imread(path)
        
        if image is None:
            print(f"--> [WARNING] Imagen corrupta ignorada: {path}")
            image = np.zeros((224, 224, 3), dtype=np.uint8)
            
        # Asegurar conversión BGR -> RGB
        if len(image.shape) == 2:
             image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocesado
        try:
            image = remove_hairs(image)
            image = crop_lesion(image)
        except Exception as e:
            print(f"Error preprocesado en {path}: {e}")

        image = Image.fromarray(image)
        
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
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
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
    image_datasets = {
        'train': LesionDataset(CONFIG['train_dir'], transform=data_transforms['train'], max_samples=15500), 
        'val': LesionDataset(CONFIG['val_dir'], transform=data_transforms['val'], max_samples=4000)
    }
    
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0),
        'val': DataLoader(image_datasets['val'], batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
    }
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    
    print(f"Resumen: {dataset_sizes['train']} imgs entreno, {dataset_sizes['val']} imgs validación")
    print(f"Clases: {class_names}")

except Exception as e:
    print(f"\n ERROR CRÍTICO AL CARGAR DATOS: {e}")
    exit()

# ==========================================
# 6. MODELO (ResNeXt-101 32x8d) - CORREGIDO
# ==========================================
print("Configurando ResNeXt-101 con Fine-Tuning...")

# CORRECCIÓN AQUÍ: Fíjate en la 'X' y la 'D' mayúsculas en 'Weights'
model_ft = models.resnext101_32x8d(weights=models.ResNeXt101_32X8D_Weights.DEFAULT)

# 1. Congelar todo inicialmente
for param in model_ft.parameters():
    param.requires_grad = False

# 2. Descongelar el último bloque
for param in model_ft.layer4.parameters():
    param.requires_grad = True

# 3. Modificar el Clasificador
num_ftrs = model_ft.fc.in_features 

model_ft.fc = nn.Sequential(
    nn.Linear(num_ftrs, 1024),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, CONFIG['num_classes'])
)

model_ft = model_ft.to(DEVICE)
criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.Adam([
    {'params': model_ft.layer4.parameters(), 'lr': 1e-5},
    {'params': model_ft.fc.parameters(), 'lr': 1e-4}
])

exp_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_ft, mode='min', factor=0.5, patience=2
)

# ==========================================
# 7. ENTRENAMIENTO
# ==========================================
def train_model(model, criterion, optimizer, scheduler, num_epochs=20):
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

        scheduler.step(history['val_loss'][-1])
        print()

    time_elapsed = time.time() - since
    print(f'Entrenamiento completado en {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Mejor Val Acc guardado: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model, history

# ==========================================
# 8. EJECUCIÓN INTELIGENTE (Cargar si existe)
# ==========================================
MODEL_PATH = 'resnext101_finetuned.pth' # <--- Nombre actualizado

if os.path.exists(MODEL_PATH):
    print(f"\n--> ¡Modelo encontrado: '{MODEL_PATH}'!")
    print("--> Cargando pesos guardados y saltando entrenamiento...")
    model_ft.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model_trained = model_ft
else:
    print(f"\n--> No se encontró '{MODEL_PATH}'. Iniciando entrenamiento desde cero...")
    model_trained, history = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=CONFIG['num_epochs'])
    torch.save(model_trained.state_dict(), MODEL_PATH)
    print(f"--> Modelo guardado como '{MODEL_PATH}'")

    # Gráficas
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Curva de Pérdida (Loss)')
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Curva de Precisión (Accuracy)')
    plt.legend()
    plt.grid()
    plt.savefig('resnext_learning_curves.png')
    plt.close()

# --- EVALUACIÓN FINAL ---
print("\nEvaluando Modelo Final...")
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

final_acc = accuracy_score(all_labels, all_preds)

print("\n" + "="*40)
print(f"--> ACCURACY FINAL: {final_acc*100:.2f}%")
print("="*40)

print("\nMatriz de Confusión:")
print(confusion_matrix(all_labels, all_preds))
print("\nReporte Detallado:")
print(classification_report(all_labels, all_preds, target_names=class_names))

global_end_time = time.time()
total_time = global_end_time - global_start_time
print(f"\n--> Tiempo TOTAL de ejecución: {total_time // 60:.0f}m {total_time % 60:.0f}s")

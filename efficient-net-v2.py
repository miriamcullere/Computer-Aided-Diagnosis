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
# IMPORTANTE: Importamos las métricas necesarias para la tabla
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
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
    'batch_size': 32, 
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
# 3. FUNCIONES DE PREPROCESADO
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
# 4. DATASET PERSONALIZADO
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
        image = cv2.imread(path)
        
        if image is None:
            # print(f"--> [WARNING] Imagen corrupta ignorada: {path}") # Comentado para no ensuciar log
            image = np.zeros((224, 224, 3), dtype=np.uint8)
            
        if len(image.shape) == 2:
             image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        try:
            image = remove_hairs(image)
            image = crop_lesion(image)
        except Exception as e:
            pass # Ignoramos errores puntuales de preprocesado

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
    # --- CONFIGURACIÓN DE TAMAÑO DE DATASET ---
    # Pon max_samples=None para usar TODAS las imágenes
    image_datasets = {
        'train': LesionDataset(CONFIG['train_dir'], transform=data_transforms['train'], max_samples=500), 
        'val': LesionDataset(CONFIG['val_dir'], transform=data_transforms['val'], max_samples=500)
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
# 6. MODELO (EfficientNet V2 Small)
# ==========================================
print("Configurando EfficientNet V2 Small...")

weights = models.EfficientNet_V2_S_Weights.DEFAULT
model_ft = models.efficientnet_v2_s(weights=weights)

for param in model_ft.parameters():
    param.requires_grad = False

for param in model_ft.features[6].parameters():
    param.requires_grad = True
for param in model_ft.features[7].parameters():
    param.requires_grad = True

num_ftrs = model_ft.classifier[1].in_features
model_ft.classifier = nn.Sequential(
    nn.Dropout(p=0.3, inplace=True),
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, CONFIG['num_classes'])
)

model_ft = model_ft.to(DEVICE)
criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.Adam([
    {'params': model_ft.features.parameters(), 'lr': 1e-5}, 
    {'params': model_ft.classifier.parameters(), 'lr': 1e-4} 
])

exp_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.5, patience=2)

# ==========================================
# 7. FUNCIÓN DE ENTRENAMIENTO
# ==========================================
def train_model(model, criterion, optimizer, scheduler, num_epochs=15):
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
# 8. EJECUCIÓN Y GENERACIÓN DE TABLA FINAL
# ==========================================
MODEL_PATH = 'efficientnetv2_s_finetuned.pth'

if os.path.exists(MODEL_PATH):
    print(f"\n--> ¡Modelo encontrado: '{MODEL_PATH}'!")
    print("--> Cargando pesos y evaluando...")
    model_ft.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model_trained = model_ft
else:
    print(f"\n--> Iniciando entrenamiento desde cero...")
    model_trained, history = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=CONFIG['num_epochs'])
    torch.save(model_trained.state_dict(), MODEL_PATH)
    print(f"--> Modelo guardado.")
    
    # Gráficas
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.legend(); plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.legend(); plt.grid()
    plt.savefig('effnetv2_curves.png')
    plt.close()

# --- EVALUACIÓN DETALLADA PARA LA TABLA ---
print("\nGenerando métricas finales...")
all_preds = []
all_labels = []
running_val_loss = 0.0

model_trained.eval()
with torch.no_grad():
    for inputs, labels in dataloaders['val']:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        
        outputs = model_trained(inputs)
        loss = criterion(outputs, labels) # Calculamos Loss también aquí
        
        _, preds = torch.max(outputs, 1)
        
        running_val_loss += loss.item() * inputs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Cálculo de métricas
final_val_loss = running_val_loss / dataset_sizes['val']
acc = accuracy_score(all_labels, all_preds)
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

# --- GENERACIÓN DE LA TABLA ---
print("\n" + "="*95)
print(f"{'Model':<20} {'Accuracy':<15} {'Precision':<15} {'Recall':<15} {'F1-score':<15} {'Loss':<15}")
print("-" * 95)
print(f"{'EfficientNetV2_S':<20} {acc:<15.4f} {precision:<15.4f} {recall:<15.4f} {f1:<15.4f} {final_val_loss:<15.4f}")
print("="*95 + "\n")

# Reporte adicional
print("Matriz de Confusión:")
print(confusion_matrix(all_labels, all_preds))
print("\nReporte por Clase:")
print(classification_report(all_labels, all_preds, target_names=class_names))

global_end_time = time.time()
total_time = global_end_time - global_start_time
print(f"--> Tiempo TOTAL: {total_time // 60:.0f}m {total_time % 60:.0f}s")

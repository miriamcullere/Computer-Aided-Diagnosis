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
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image

# ==========================================
# 0. INICIO CRONÓMETRO GLOBAL
# ==========================================
global_start_time = time.time()
print(f"--> Inicio del script: {time.ctime()}")

# ==========================================
# 2. CONFIGURACIÓN
# ==========================================
CONFIG = {
    'train_dir': '/Users/miriamcullere/Library/CloudStorage/OneDrive-UniversitatdeGirona/MMIC/CAD/PRÀCTICA_MELASMA/CHALLENGE1/train',
    'val_dir': '/Users/miriamcullere/Library/CloudStorage/OneDrive-UniversitatdeGirona/MMIC/CAD/PRÀCTICA_MELASMA/CHALLENGE1/val', 
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
# 3. DATASET PERSONALIZADO (CON LIMITADOR)
# ==========================================

class LesionDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_samples=None):
        self.image_folder = datasets.ImageFolder(root_dir)
        self.classes = self.image_folder.classes
        self.samples = self.image_folder.samples 
        self.transform = transform

        # --- LIMITADOR DE IMÁGENES ---
        if max_samples is not None:
            # Mezclar aleatoriamente antes de cortar para asegurar variedad de clases
            random.seed(42) 
            random.shuffle(self.samples)
            
            # Aplicar el límite
            original_len = len(self.samples)
            self.samples = self.samples[:max_samples]
            print(f"--> [DEBUG] Dataset limitado de {original_len} a {len(self.samples)} imágenes.")
        # -----------------------------

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        
        # Carga estándar con PIL
        try:
            image = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"Error cargando {path}: {e}")
            image = Image.new('RGB', (224, 224))
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# ==========================================
# 4. CARGA DE DATOS
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
    # APLICAMOS EL LÍMITE: 2000 imágenes para train y 2000 para val
    image_datasets = {
        'train': LesionDataset(CONFIG['train_dir'], transform=data_transforms['train'], max_samples=2000),
        'val': LesionDataset(CONFIG['val_dir'], transform=data_transforms['val'], max_samples=2000)
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
# 5. MODELO (DenseNet121 - FINE TUNING)
# ==========================================
print("Configurando DenseNet121 para Fine-Tuning...")

model_ft = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)

for param in model_ft.parameters():
    param.requires_grad = False

for param in model_ft.features.denseblock4.parameters():
    param.requires_grad = True
for param in model_ft.features.norm5.parameters():
    param.requires_grad = True

num_ftrs = model_ft.classifier.in_features
model_ft.classifier = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 2)
)

model_ft = model_ft.to(DEVICE)
criterion = nn.CrossEntropyLoss()

params_to_update = [p for p in model_ft.parameters() if p.requires_grad]
optimizer_ft = optim.Adam(params_to_update, lr=CONFIG['learning_rate'])

exp_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_ft, mode='min', factor=0.5, patience=2
)

# ==========================================
# 6. ENTRENAMIENTO
# ==========================================
def train_model(model, criterion, optimizer, scheduler, num_epochs=15):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f"\nIniciando Fine-Tuning ({num_epochs} épocas)...")

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
    print(f'Mejor Val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model, history

# ==========================================
# 7. EJECUCIÓN Y VISUALIZACIÓN
# ==========================================

model_trained, history = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=CONFIG['num_epochs'])

torch.save(model_trained.state_dict(), 'densenet_finetuned.pth')
print("Modelo guardado como 'densenet_finetuned.pth'")

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
plt.savefig('learning_curves.png')
plt.show()

print("\nGenerando Matriz de Confusión...")
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

print("\nMatriz de Confusión:")
print(confusion_matrix(all_labels, all_preds))
print("\nReporte Final:")
print(classification_report(all_labels, all_preds, target_names=class_names))

# ==========================================
# 8. CÁLCULO TIEMPO TOTAL
# ==========================================
global_end_time = time.time()
total_time = global_end_time - global_start_time
print("\n" + "="*30)
print(f"--> Tiempo TOTAL de ejecución: {total_time // 60:.0f}m {total_time % 60:.0f}s")
print("="*30)
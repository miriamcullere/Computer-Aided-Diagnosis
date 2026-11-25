import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models, transforms, datasets
import time
import copy
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import cv2
import shutil
from tqdm import tqdm
from skimage.morphology import remove_small_objects 

# ==========================================
# 0. CONFIGURACIÓN
# ==========================================
ORIGINAL_TRAIN_DIR = '/Users/miriamcullere/Desktop/MMIC/CAD/DATASET/BINARY/train'
ORIGINAL_VAL_DIR = '/Users/miriamcullere/Desktop/MMIC/CAD/DATASET/BINARY/val'

PROCESSED_BASE_DIR = '/Users/miriamcullere/Desktop/MMIC/CAD/DATASET/BINARY_PROCESSED_V2'
PROCESSED_TRAIN_DIR = os.path.join(PROCESSED_BASE_DIR, 'train')
PROCESSED_VAL_DIR = os.path.join(PROCESSED_BASE_DIR, 'val')

CONFIG = {
    'num_classes': 2,
    'batch_size': 32, 
    'num_epochs': 25,
    'learning_rate': 1e-4
}

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()

# ==========================================
# 1. FUNCIONES (Definiciones)
# ==========================================
def process_image(path):
    image = cv2.imread(path)
    if image is None: return None
    try:
        img_proc = image.copy()
        gray = cv2.cvtColor(img_proc, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        mask = cv2.dilate(mask, None, iterations=1)
        mask_bool = remove_small_objects(mask.astype(bool), min_size=30)
        mask = mask_bool.astype(np.uint8) * 255
        image_clean = cv2.inpaint(img_proc, mask, 1, cv2.INPAINT_TELEA)
        
        gray_clean = cv2.cvtColor(image_clean, cv2.COLOR_BGR2GRAY)
        _, mask_crop = cv2.threshold(gray_clean, 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            crop_factor = 0.05
            dw, dh = int(w*crop_factor), int(h*crop_factor)
            nx, ny, nw, nh = x+dw, y+dh, w-2*dw, h-2*dh
            if nw > 0 and nh > 0:
                image_clean = image_clean[ny:ny+nh, nx:nx+nw]
            else:
                image_clean = image_clean[y:y+h, x:x+w]
        return image_clean
    except Exception as e:
        return image

def prepare_dataset_on_disk(source_dir, dest_dir):
    if os.path.exists(dest_dir):
        print(f"--> La carpeta {dest_dir} ya existe. Asumimos que ya está procesada.")
        return

    print(f"--> Creando dataset procesado en: {dest_dir}...")
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_path): continue
        dest_class_path = os.path.join(dest_dir, class_name)
        os.makedirs(dest_class_path, exist_ok=True)
        files = os.listdir(class_path)
        print(f"Procesando clase '{class_name}' ({len(files)} imágenes)...")
        for fname in tqdm(files):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')): continue
            src_file = os.path.join(class_path, fname)
            dst_file = os.path.join(dest_class_path, fname)
            final_img = process_image(src_file)
            if final_img is not None:
                cv2.imwrite(dst_file, final_img)

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train': model.train()
            else: model.eval()

            running_loss = 0.0
            running_corrects = 0

            # tqdm en el dataloader
            loop = tqdm(dataloaders[phase], leave=False)
            for inputs, labels in loop:
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
                loop.set_description(f"{phase} Ep {epoch+1}")

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.float() / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                history['train_loss'].append(epoch_loss); history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss); history['val_acc'].append(epoch_acc.item())
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        scheduler.step(history['val_loss'][-1])

    time_elapsed = time.time() - since
    print(f'Completado en {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s. Mejor Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model, history

# ==========================================
# 3. EJECUCIÓN PROTEGIDA (NECESARIO EN MAC)
# ==========================================
if __name__ == '__main__':
    # Todo el código de ejecución va aquí dentro
    
    print("=== FASE 1: PREPARACIÓN DE DATOS ===")
    prepare_dataset_on_disk(ORIGINAL_TRAIN_DIR, PROCESSED_TRAIN_DIR)
    prepare_dataset_on_disk(ORIGINAL_VAL_DIR, PROCESSED_VAL_DIR)
    print("\n=== DATOS LISTOS EN DISCO ===")

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(45), 
            transforms.ColorJitter(brightness=0.3, contrast=0.3), 
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {
        'train': datasets.ImageFolder(PROCESSED_TRAIN_DIR, transform=data_transforms['train']),
        'val': datasets.ImageFolder(PROCESSED_VAL_DIR, transform=data_transforms['val'])
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    print(f"Resumen Final: {dataset_sizes['train']} train, {dataset_sizes['val']} val")

    # Sampler
    print("Configurando Sampler...")
    train_targets = [s[1] for s in image_datasets['train'].samples]
    class_sample_count = np.bincount(train_targets)
    weight = 1. / class_sample_count
    samples_weight = torch.from_numpy(np.array([weight[t] for t in train_targets]))
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=CONFIG['batch_size'], sampler=sampler, num_workers=2),
        'val': DataLoader(image_datasets['val'], batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2)
    }

    # Modelo
    print("Configurando DenseNet121...")
    model_ft = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    for param in model_ft.parameters(): param.requires_grad = False
    for param in model_ft.features.denseblock3.parameters(): param.requires_grad = True
    for param in model_ft.features.transition3.parameters(): param.requires_grad = True
    for param in model_ft.features.denseblock4.parameters(): param.requires_grad = True
    for param in model_ft.features.norm5.parameters(): param.requires_grad = True

    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 1024), nn.ReLU(), nn.Dropout(0.6),
        nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.4),
        nn.Linear(512, 2)
    )
    model_ft = model_ft.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam([
        {'params': model_ft.features.parameters(), 'lr': 1e-5},
        {'params': model_ft.classifier.parameters(), 'lr': 1e-4}
    ])
    exp_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.5, patience=2)

    # Ejecución Segura
    MODEL_PATH = 'densenet_processed_disk.pth'
    if os.path.exists(MODEL_PATH):
        print("Modelo encontrado, cargando...")
        model_ft.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model_trained = model_ft
    else:
        # Pasamos los dataloaders y sizes como argumentos ahora
        model_trained, history = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, dataset_sizes, num_epochs=CONFIG['num_epochs'])
        torch.save(model_trained.state_dict(), MODEL_PATH)

# ==========================================
    # EVALUACIÓN FINAL CON MATRIZ
    # ==========================================
    print("\nEvaluando...")
    all_preds, all_labels = [], []
    model_trained.eval()
    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs = inputs.to(DEVICE)
            outputs = model_trained(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 1. Matriz de Confusión (AQUÍ ESTÁ LA LÍNEA QUE FALTABA)
    print("\nMatriz de Confusión:")
    print(confusion_matrix(all_labels, all_preds))

    # 2. Reporte detallado (Precision, Recall, F1)
    print("\nReporte Detallado:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # 3. Accuracy Final
    acc = accuracy_score(all_labels, all_preds)
    print(f"\n--> ACCURACY FINAL: {acc*100:.2f}%")

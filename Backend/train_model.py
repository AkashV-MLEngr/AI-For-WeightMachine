import os
import json
import shutil
import copy
import time
import random

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEBSITE_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
PROJECT_DIR = os.path.abspath(os.path.join(WEBSITE_DIR, ".."))

# Original dataset: D:\Fruit vegetable detection\train
ORIGINAL_DATASET = os.path.join(PROJECT_DIR, "train")

# User-added training images
USER_DATASET = os.path.join(WEBSITE_DIR, "training_data")

# Temporary combined dataset
COMBINED_DATASET = os.path.join(BASE_DIR, "combined_training_data")

# AI model files
MODEL_DIR = os.path.join(WEBSITE_DIR, "AI model")
MODEL_PATH = os.path.join(MODEL_DIR, "fruit_vegetable_resnet18.pth")
BACKUP_MODEL_PATH = os.path.join(MODEL_DIR, "fruit_vegetable_resnet18_backup.pth")
CLASS_FILE = os.path.join(MODEL_DIR, "classes.json")
BACKUP_CLASS_FILE = os.path.join(MODEL_DIR, "classes_backup.json")

# Training report
REPORT_FILE = os.path.join(MODEL_DIR, "training_report.json")

os.makedirs(USER_DATASET, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# =========================================================
# SETTINGS
# =========================================================

SEED = 42
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.0001
VALIDATION_SPLIT = 0.20
# Conservative setting for Windows/Python 3.14
NUM_WORKERS = 0

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("================================")
print("FreshVision AI Training")
print("================================")
print("Device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# =========================================================
# CHECK DATASETS
# =========================================================

if not os.path.exists(ORIGINAL_DATASET):
    print("\nERROR: Original training dataset not found!")
    print("Expected:", ORIGINAL_DATASET)
    raise SystemExit(1)

if not os.path.exists(USER_DATASET):
    os.makedirs(USER_DATASET, exist_ok=True)


# =========================================================
# COMBINE ORIGINAL + USER DATA
# =========================================================

print("\nPreparing combined dataset...")

if os.path.exists(COMBINED_DATASET):
    shutil.rmtree(COMBINED_DATASET)

os.makedirs(COMBINED_DATASET)


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def copy_dataset(source):
    if not os.path.exists(source):
        return

    for class_name in os.listdir(source):
        class_path = os.path.join(source, class_name)

        if not os.path.isdir(class_path):
            continue

        destination = os.path.join(COMBINED_DATASET, class_name)
        os.makedirs(destination, exist_ok=True)

        for filename in os.listdir(class_path):
            source_file = os.path.join(class_path, filename)

            if not os.path.isfile(source_file):
                continue

            if os.path.splitext(filename)[1].lower() not in VALID_EXTENSIONS:
                continue

            destination_file = os.path.join(destination, filename)

            # Avoid overwriting a file if both datasets contain the same name.
            if os.path.exists(destination_file):
                stem, ext = os.path.splitext(filename)
                counter = 1
                while os.path.exists(destination_file):
                    destination_file = os.path.join(
                        destination, f"{stem}_user_{counter}{ext}"
                    )
                    counter += 1

            shutil.copy2(source_file, destination_file)


print("Copying original dataset...")
copy_dataset(ORIGINAL_DATASET)

print("Copying user training data...")
copy_dataset(USER_DATASET)


# =========================================================
# DATASET + TRANSFORMS
# =========================================================

def safe_pil_loader(path):
    # Always convert palette/RGBA/grayscale images to normal RGB.
    with Image.open(path) as img:
        return img.convert("RGB")


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Build class list once without augmentation.
base_dataset = datasets.ImageFolder(COMBINED_DATASET, loader=safe_pil_loader)
class_names = base_dataset.classes
num_classes = len(class_names)

if num_classes == 0 or len(base_dataset) == 0:
    print("\nERROR: No valid training images found.")
    raise SystemExit(1)

print("\nTotal classes:", num_classes)
print("Total images:", len(base_dataset))
print("Preparing image loader...", flush=True)

print("\nClass list:")
for i, name in enumerate(class_names):
    print(i, "->", name)


# =========================================================
# TRAIN / VALIDATION SPLIT
# =========================================================

indices = list(range(len(base_dataset)))
random.Random(SEED).shuffle(indices)

val_size = max(1, int(len(indices) * VALIDATION_SPLIT))
train_indices = indices[val_size:]
val_indices = indices[:val_size]

if len(train_indices) == 0:
    print("\nERROR: Not enough images for training.")
    raise SystemExit(1)

train_dataset_full = datasets.ImageFolder(
    COMBINED_DATASET,
    transform=train_transform,
    loader=safe_pil_loader
)

val_dataset_full = datasets.ImageFolder(
    COMBINED_DATASET,
    transform=val_transform,
    loader=safe_pil_loader
)

train_dataset = Subset(train_dataset_full, train_indices)
val_dataset = Subset(val_dataset_full, val_indices)

print("\nTraining images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# =========================================================
# DATALOADERS
# =========================================================

pin_memory = torch.cuda.is_available()

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=pin_memory
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=pin_memory
)


# =========================================================
# LOAD PREVIOUS MODEL
# =========================================================

old_checkpoint = None
old_class_names = []

if os.path.exists(MODEL_PATH):
    print("\nExisting model found.")

    try:
        old_checkpoint = torch.load(
            MODEL_PATH,
            map_location="cpu"
        )

        if os.path.exists(CLASS_FILE):
            with open(CLASS_FILE, "r", encoding="utf-8") as f:
                old_class_names = json.load(f)

        print("Previous classes:", len(old_class_names))

    except Exception as e:
        print("Could not load previous model.")
        print("Reason:", e)
        old_checkpoint = None
        old_class_names = []


# =========================================================
# CREATE MODEL
# =========================================================

print("\nCreating ResNet18...")

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)


# =========================================================
# PRESERVE OLD KNOWLEDGE
# =========================================================

if old_checkpoint is not None:
    print("\nPreserving existing model knowledge...")

    current_state = model.state_dict()

    # Preserve the backbone.
    for key, value in old_checkpoint.items():
        if key.startswith("fc."):
            continue

        if key in current_state and current_state[key].shape == value.shape:
            current_state[key] = value

    # Preserve FC weights for classes that still exist.
    if (
        "fc.weight" in old_checkpoint
        and "fc.bias" in old_checkpoint
        and old_class_names
    ):
        old_weight = old_checkpoint["fc.weight"]
        old_bias = old_checkpoint["fc.bias"]

        preserved_count = 0

        for old_index, old_name in enumerate(old_class_names):
            if old_name in class_names and old_index < old_weight.shape[0]:
                new_index = class_names.index(old_name)

                current_state["fc.weight"][new_index] = old_weight[old_index]
                current_state["fc.bias"][new_index] = old_bias[old_index]

                preserved_count += 1

        print("Preserved classes:", preserved_count)

    model.load_state_dict(current_state)

else:
    print("\nNo previous model found. Training from scratch.")


model = model.to(device)


# =========================================================
# LOSS + OPTIMIZER
# =========================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# =========================================================
# TRAINING
# =========================================================

print("\n================================")
print("TRAINING STARTED")
print("================================")

best_val_accuracy = -1.0
best_state = None
start_time = time.time()

for epoch in range(EPOCHS):

    # -------------------------
    # TRAIN
    # -------------------------
    model.train()

    running_loss = 0.0
    train_correct = 0
    train_total = 0

    total_batches = len(train_loader)

    for batch_number, (images, labels) in enumerate(train_loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_number == 1:
            print(f"  First batch completed ({batch_number}/{total_batches})", flush=True)
        elif batch_number % 25 == 0 or batch_number == total_batches:
            print(f"  Batch {batch_number}/{total_batches}", flush=True)

        _, predicted = torch.max(outputs, 1)

        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_accuracy = (
        100.0 * train_correct / train_total
        if train_total else 0.0
    )

    average_loss = (
        running_loss / len(train_loader)
        if len(train_loader) else 0.0
    )

    # -------------------------
    # VALIDATION
    # -------------------------
    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_accuracy = (
        100.0 * val_correct / val_total
        if val_total else 0.0
    )

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {average_loss:.4f} "
        f"Train Accuracy: {train_accuracy:.2f}% "
        f"Validation Accuracy: {val_accuracy:.2f}%"
    )

    # Keep the best validation model in memory.
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_state = copy.deepcopy(model.state_dict())


if best_state is None:
    print("\nERROR: Training did not produce a valid model.")
    raise SystemExit(1)


# =========================================================
# SAFE MODEL SAVE
# =========================================================

print("\n================================")
print("VALIDATION COMPLETE")
print("Best Validation Accuracy:",
      f"{best_val_accuracy:.2f}%")
print("================================")

# Backup the current working model before replacing it.
if os.path.exists(MODEL_PATH):
    shutil.copy2(MODEL_PATH, BACKUP_MODEL_PATH)

if os.path.exists(CLASS_FILE):
    shutil.copy2(CLASS_FILE, BACKUP_CLASS_FILE)

# Save candidate model first.
TEMP_MODEL_PATH = MODEL_PATH + ".tmp"
TEMP_CLASS_PATH = CLASS_FILE + ".tmp"

torch.save(best_state, TEMP_MODEL_PATH)

with open(TEMP_CLASS_PATH, "w", encoding="utf-8") as f:
    json.dump(class_names, f, indent=4)

# Only after successful writes, activate the new model.
os.replace(TEMP_MODEL_PATH, MODEL_PATH)
os.replace(TEMP_CLASS_PATH, CLASS_FILE)


# =========================================================
# TRAINING REPORT
# =========================================================

elapsed_seconds = time.time() - start_time

report = {
    "success": True,
    "classes": len(class_names),
    "images": len(base_dataset),
    "training_images": len(train_dataset),
    "validation_images": len(val_dataset),
    "epochs": EPOCHS,
    "best_validation_accuracy": round(best_val_accuracy, 2),
    "training_time_seconds": round(elapsed_seconds, 2),
    "device": str(device),
    "model": "ResNet18"
}

with open(REPORT_FILE, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=4)


# =========================================================
# COMPLETE
# =========================================================

print("\n================================")
print("TRAINING COMPLETED SUCCESSFULLY")
print("================================")
print("Model saved:", MODEL_PATH)
print("Classes:", len(class_names))
print("Images:", len(base_dataset))
print("Best validation accuracy:",
      f"{best_val_accuracy:.2f}%")
print("Training time:",
      f"{elapsed_seconds / 60:.2f} minutes")
print("\nDone!")

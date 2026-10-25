import os
import json
import shutil
import copy
import time
import random

import torch
import torch.nn as nn

from PIL import Image, ImageFile
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset


# ============================================================
# BASIC SETTINGS
# ============================================================

ImageFile.LOAD_TRUNCATED_IMAGES = True

SEED = 42

REPLAY_PER_CLASS = 10
EPOCHS = 2
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
VALIDATION_SPLIT = 0.20

EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp"
}


# ============================================================
# PATHS
# ============================================================

# This file is inside:
# D:\Fruit vegetable detection\Website\Backend

BASE = os.path.dirname(os.path.abspath(__file__))

# Website folder
WEBSITE = os.path.abspath(
    os.path.join(BASE, "..")
)

# Main project folder
PROJECT = os.path.abspath(
    os.path.join(WEBSITE, "..")
)

# Original dataset
ORIGINAL = os.path.join(
    PROJECT,
    "train"
)

# User uploaded training images
USER = os.path.join(
    BASE,
    "training_data"
)

# IMPORTANT:
# Model MUST be inside Backend\AI model
MODEL_DIR = os.path.join(
    BASE,
    "AI model"
)

MODEL = os.path.join(
    MODEL_DIR,
    "fruit_vegetable_resnet18.pth"
)

CLASSFILE = os.path.join(
    MODEL_DIR,
    "classes.json"
)

BACKUP = os.path.join(
    MODEL_DIR,
    "fruit_vegetable_resnet18_backup.pth"
)

BACKUP_CLASSES = os.path.join(
    MODEL_DIR,
    "classes_backup.json"
)

REPORT = os.path.join(
    MODEL_DIR,
    "training_report.json"
)

# Replay dataset
REPLAY = os.path.join(
    BASE,
    "incremental_replay_data"
)


# ============================================================
# RANDOM SEED
# ============================================================

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# SAFE IMAGE LOADER
# ============================================================

def safe_loader(path):
    try:
        with Image.open(path) as img:
            return img.convert("RGB")

    except Exception as error:
        print(
            f"WARNING: Could not read image: {path}"
        )
        print(
            f"Reason: {error}"
        )
        raise


# ============================================================
# DIRECTORY SETUP
# ============================================================

os.makedirs(USER, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPLAY, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print()
print("================================")
print("FreshVision Incremental Training")
print("================================")

print(
    f"Device: {DEVICE}"
)

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

print()


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD OLD CLASSES
# ============================================================

if not os.path.exists(CLASSFILE):

    print("ERROR: classes.json not found.")
    print(CLASSFILE)

    raise SystemExit(1)


try:

    with open(
        CLASSFILE,
        "r",
        encoding="utf-8"
    ) as file:

        old_classes = json.load(file)

except Exception as error:

    print()
    print("ERROR: Could not read classes.json")
    print(error)

    raise SystemExit(1)


if not isinstance(old_classes, list):

    print(
        "ERROR: classes.json must contain a list."
    )

    raise SystemExit(1)


old_classes = [
    str(class_name)
    for class_name in old_classes
]


# Check duplicate classes

if len(old_classes) != len(set(old_classes)):

    print()
    print("ERROR: Duplicate class names found in classes.json")

    raise SystemExit(1)


print(
    f"Previous classes: {len(old_classes)}"
)


# ============================================================
# LOAD EXISTING MODEL CHECKPOINT
# ============================================================

if not os.path.exists(MODEL):

    print()
    print("ERROR: Existing model not found.")
    print(MODEL)

    raise SystemExit(1)


print()
print("Checking existing model...")

try:

    checkpoint = torch.load(
        MODEL,
        map_location="cpu"
    )

except Exception as error:

    print()
    print("ERROR: Could not load existing model.")
    print(error)

    raise SystemExit(1)


if not isinstance(checkpoint, dict):

    print()
    print("ERROR: Model checkpoint is not a state_dict.")

    raise SystemExit(1)


if "fc.weight" not in checkpoint:

    print()
    print("ERROR: fc.weight not found in model.")

    raise SystemExit(1)


if "fc.bias" not in checkpoint:

    print()
    print("ERROR: fc.bias not found in model.")

    raise SystemExit(1)


old_fc_weight = checkpoint["fc.weight"]
old_fc_bias = checkpoint["fc.bias"]


# ============================================================
# VERIFY MODEL / CLASSES MATCH
# ============================================================

if old_fc_weight.ndim != 2:

    print()
    print("ERROR: Invalid fc.weight shape:")
    print(tuple(old_fc_weight.shape))

    raise SystemExit(1)


if old_fc_bias.ndim != 1:

    print()
    print("ERROR: Invalid fc.bias shape:")
    print(tuple(old_fc_bias.shape))

    raise SystemExit(1)


model_class_count = old_fc_weight.shape[0]
metadata_class_count = len(old_classes)


print(
    f"Existing model classes: {model_class_count}"
)

print(
    f"classes.json classes: {metadata_class_count}"
)


if model_class_count != metadata_class_count:

    print()
    print("================================")
    print("ERROR: MODEL / CLASSES MISMATCH")
    print("================================")

    print(
        f"Model FC classes : {model_class_count}"
    )

    print(
        f"classes.json     : {metadata_class_count}"
    )

    print()
    print(
        "Training stopped to protect old model knowledge."
    )

    raise SystemExit(1)


if old_fc_bias.shape[0] != model_class_count:

    print()
    print("ERROR: fc.weight and fc.bias mismatch.")

    raise SystemExit(1)


# ============================================================
# LOAD ORIGINAL DATASET
# ============================================================

print()
print("Loading original dataset...")

if not os.path.isdir(ORIGINAL):

    print()
    print("ERROR: Original dataset not found:")
    print(ORIGINAL)

    raise SystemExit(1)


try:

    original_dataset = datasets.ImageFolder(
        ORIGINAL,
        loader=safe_loader
    )

except Exception as error:

    print()
    print("ERROR: Could not load original dataset.")
    print(error)

    raise SystemExit(1)


original_classes = list(
    original_dataset.classes
)


print(
    f"Original dataset classes: {len(original_classes)}"
)

print(
    f"Original dataset images: {len(original_dataset)}"
)


# ============================================================
# SCAN USER TRAINING DATA
# ============================================================

print()
print("Scanning user training data...")

user_data = {}

for class_name in sorted(
    os.listdir(USER)
):

    class_path = os.path.join(
        USER,
        class_name
    )

    if not os.path.isdir(class_path):
        continue

    image_files = []

    for filename in sorted(
        os.listdir(class_path)
    ):

        file_path = os.path.join(
            class_path,
            filename
        )

        if not os.path.isfile(file_path):
            continue

        extension = os.path.splitext(
            filename
        )[1].lower()

        if extension not in EXTENSIONS:
            continue

        image_files.append(
            file_path
        )

    if len(image_files) > 0:

        user_data[class_name] = image_files


print()
print("User-uploaded images:")

total_user_images = 0

for class_name in sorted(
    user_data.keys()
):

    count = len(
        user_data[class_name]
    )

    total_user_images += count

    print(
        f"  {class_name}: {count}"
    )


print(
    f"Total user images: {total_user_images}"
)


# ============================================================
# DYNAMIC CLASS EXPANSION
# ============================================================

# Existing classes are NEVER removed.
#
# New folder/class names are automatically added.
#
# Example:
#
# 36 old classes
# + peanut
# + arai keerai
# = 38 classes
#
# Future:
#
# + badam
# + murungai keerai
# = 40 classes
#
# and so on.

classes = sorted(
    set(old_classes)
    | set(original_classes)
    | set(user_data.keys())
)


new_classes = [
    class_name
    for class_name in classes
    if class_name not in old_classes
]


print()
print(
    f"Final class count: {len(classes)}"
)


if new_classes:

    print(
        "New classes:"
    )

    for class_name in new_classes:

        print(
            f"  + {class_name}"
        )

else:

    print(
        "New classes: none"
    )


# ============================================================
# BUILD REPLAY DATASET
# ============================================================

print()
print("Preparing incremental replay dataset...")


# Delete old replay dataset

if os.path.exists(REPLAY):

    shutil.rmtree(REPLAY)


os.makedirs(
    REPLAY,
    exist_ok=True
)


# ------------------------------------------------------------
# Copy replay images from original dataset
# ------------------------------------------------------------

for class_name in old_classes:

    source_class_dir = os.path.join(
        ORIGINAL,
        class_name
    )

    replay_class_dir = os.path.join(
        REPLAY,
        class_name
    )

    if not os.path.isdir(
        source_class_dir
    ):

        continue

    os.makedirs(
        replay_class_dir,
        exist_ok=True
    )

    image_files = []

    for filename in os.listdir(
        source_class_dir
    ):

        file_path = os.path.join(
            source_class_dir,
            filename
        )

        if not os.path.isfile(file_path):
            continue

        extension = os.path.splitext(
            filename
        )[1].lower()

        if extension not in EXTENSIONS:
            continue

        image_files.append(
            file_path
        )

    random.shuffle(
        image_files
    )

    selected_files = image_files[
        :REPLAY_PER_CLASS
    ]

    for index, source_file in enumerate(
        selected_files
    ):

        extension = os.path.splitext(
            source_file
        )[1].lower()

        destination_file = os.path.join(
            replay_class_dir,
            f"replay_{index:04d}{extension}"
        )

        shutil.copy2(
            source_file,
            destination_file
        )


# ------------------------------------------------------------
# Copy ALL user images
# ------------------------------------------------------------

for class_name, image_files in user_data.items():

    destination_class_dir = os.path.join(
        REPLAY,
        class_name
    )

    os.makedirs(
        destination_class_dir,
        exist_ok=True
    )

    for index, source_file in enumerate(
        image_files
    ):

        extension = os.path.splitext(
            source_file
        )[1].lower()

        destination_file = os.path.join(
            destination_class_dir,
            f"user_{index:05d}{extension}"
        )

        shutil.copy2(
            source_file,
            destination_file
        )


# ============================================================
# VERIFY REPLAY DATASET
# ============================================================

print()

try:

    full_dataset = datasets.ImageFolder(
        REPLAY,
        loader=safe_loader
    )

except Exception as error:

    print()
    print(
        "ERROR: Could not create incremental dataset."
    )

    print(error)

    raise SystemExit(1)


print(
    f"Incremental dataset images: {len(full_dataset)}"
)


print(
    f"Dataset classes found: {len(full_dataset.classes)}"
)


# ============================================================
# VERIFY CLASS ORDER / COUNT
# ============================================================

if full_dataset.classes != classes:

    print()
    print("================================")
    print("ERROR: CLASS LIST MISMATCH")
    print("================================")

    print(
        "Expected:"
    )

    print(classes)

    print()
    print(
        "ImageFolder:"
    )

    print(full_dataset.classes)

    raise SystemExit(1)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

dataset_size = len(full_dataset)

if dataset_size < 2:

    print()
    print("ERROR: Not enough images for training.")

    raise SystemExit(1)


indices = list(
    range(dataset_size)
)

random.shuffle(
    indices
)


validation_size = max(
    1,
    int(
        dataset_size * VALIDATION_SPLIT
    )
)


validation_indices = indices[
    :validation_size
]

training_indices = indices[
    validation_size:
]


print()
print(
    f"Training images: {len(training_indices)}"
)

print(
    f"Validation images: {len(validation_indices)}"
)


# ============================================================
# CREATE DATASETS
# ============================================================

train_dataset_full = datasets.ImageFolder(
    REPLAY,
    transform=train_transform,
    loader=safe_loader
)

val_dataset_full = datasets.ImageFolder(
    REPLAY,
    transform=val_transform,
    loader=safe_loader
)


train_dataset = Subset(
    train_dataset_full,
    training_indices
)

val_dataset = Subset(
    val_dataset_full,
    validation_indices
)


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# CREATE NEW RESNET18
# ============================================================

print()
print("Creating ResNet18...")


model = models.resnet18(
    weights=None
)


# IMPORTANT:
# Output neurons = current number of classes

model.fc = nn.Linear(
    model.fc.in_features,
    len(classes)
)


# ============================================================
# RESTORE EXISTING BACKBONE
# ============================================================

current_state = model.state_dict()


for key, value in checkpoint.items():

    if key not in current_state:
        continue

    # FC will be restored separately
    if key.startswith("fc."):

        continue

    if current_state[key].shape != value.shape:

        print(
            f"Skipping incompatible layer: {key}"
        )

        continue

    current_state[key] = value


model.load_state_dict(
    current_state,
    strict=False
)


# ============================================================
# RESTORE OLD FC ROWS BY CLASS NAME
# ============================================================

print()
print("Restoring old classifier knowledge...")


new_fc_weight = model.fc.weight.data.clone()
new_fc_bias = model.fc.bias.data.clone()


for old_index, class_name in enumerate(
    old_classes
):

    if class_name not in classes:

        continue

    new_index = classes.index(
        class_name
    )

    new_fc_weight[new_index] = (
        old_fc_weight[old_index]
        .clone()
    )

    new_fc_bias[new_index] = (
        old_fc_bias[old_index]
        .clone()
    )


model.fc.weight.data.copy_(
    new_fc_weight
)

model.fc.bias.data.copy_(
    new_fc_bias
)


print(
    "Old classifier rows restored."
)


# ============================================================
# FREEZE MOST OF MODEL
# ============================================================

for parameter in model.parameters():

    parameter.requires_grad = False


# Train only layer4 and FC

for parameter in model.layer4.parameters():

    parameter.requires_grad = True


for parameter in model.fc.parameters():

    parameter.requires_grad = True


# ============================================================
# MOVE MODEL TO GPU
# ============================================================

model = model.to(
    DEVICE
)


# ============================================================
# OPTIMIZER
# ============================================================

trainable_parameters = [
    parameter
    for parameter in model.parameters()
    if parameter.requires_grad
]


optimizer = torch.optim.Adam(
    trainable_parameters,
    lr=LEARNING_RATE
)


criterion = nn.CrossEntropyLoss()


print()
print("Existing model restored.")
print()
print("Fine-tuning layer4 + fc")
print(
    f"Epochs: {EPOCHS}"
)

print(
    f"Learning rate: {LEARNING_RATE}"
)


# ============================================================
# TRAINING
# ============================================================

best_accuracy = 0.0

best_state = copy.deepcopy(
    model.state_dict()
)


start_time = time.time()


print()
print("================================")
print("INCREMENTAL TRAINING STARTED")
print("================================")


for epoch in range(
    EPOCHS
):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0


    total_batches = len(
        train_loader
    )


    for batch_number, (
        images,
        labels
    ) in enumerate(
        train_loader,
        start=1
    ):

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )


        optimizer.zero_grad()


        outputs = model(
            images
        )


        loss = criterion(
            outputs,
            labels
        )


        loss.backward()

        optimizer.step()


        running_loss += (
            loss.item()
            * images.size(0)
        )


        _, predicted = torch.max(
            outputs,
            1
        )


        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


        print(
            f"  Epoch {epoch + 1}/{EPOCHS} "
            f"Batch {batch_number}/{total_batches}",
            end="\r"
        )


    train_loss = (
        running_loss / total
        if total > 0
        else 0.0
    )


    train_accuracy = (
        100.0 * correct / total
        if total > 0
        else 0.0
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    validation_correct = 0
    validation_total = 0


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )


            outputs = model(
                images
            )


            _, predicted = torch.max(
                outputs,
                1
            )


            validation_total += (
                labels.size(0)
            )


            validation_correct += (
                predicted == labels
            ).sum().item()


    validation_accuracy = (
        100.0
        * validation_correct
        / validation_total
        if validation_total > 0
        else 0.0
    )


    print()

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {train_loss:.4f} "
        f"Train Accuracy: {train_accuracy:.2f}% "
        f"Validation Accuracy: {validation_accuracy:.2f}%"
    )


    # Save best model in memory

    if validation_accuracy >= best_accuracy:

        best_accuracy = (
            validation_accuracy
        )

        best_state = copy.deepcopy(
            model.state_dict()
        )


# ============================================================
# TRAINING FINISHED
# ============================================================

training_time = (
    time.time()
    - start_time
)


print()
print("================================")
print("INCREMENTAL TRAINING FINISHED")
print("================================")


# ============================================================
# VERIFY BEST STATE BEFORE SAVING
# ============================================================

expected_fc_weight_shape = (
    len(classes),
    512
)

expected_fc_bias_shape = (
    len(classes),
)


actual_fc_weight_shape = tuple(
    best_state["fc.weight"].shape
)

actual_fc_bias_shape = tuple(
    best_state["fc.bias"].shape
)


print()
print(
    f"Expected FC weight shape: "
    f"{expected_fc_weight_shape}"
)

print(
    f"Actual FC weight shape: "
    f"{actual_fc_weight_shape}"
)


print(
    f"Expected FC bias shape: "
    f"{expected_fc_bias_shape}"
)

print(
    f"Actual FC bias shape: "
    f"{actual_fc_bias_shape}"
)


if (
    actual_fc_weight_shape
    != expected_fc_weight_shape
):

    print()
    print(
        "ERROR: FC weight shape is incorrect."
    )

    raise SystemExit(1)


if (
    actual_fc_bias_shape
    != expected_fc_bias_shape
):

    print()
    print(
        "ERROR: FC bias shape is incorrect."
    )

    raise SystemExit(1)


# ============================================================
# BACKUP OLD MODEL
# ============================================================

print()
print("Creating backup...")


if os.path.exists(MODEL):

    shutil.copy2(
        MODEL,
        BACKUP
    )


if os.path.exists(CLASSFILE):

    shutil.copy2(
        CLASSFILE,
        BACKUP_CLASSES
    )


# ============================================================
# ATOMIC SAVE
# ============================================================

temporary_model = (
    MODEL + ".tmp"
)

temporary_classfile = (
    CLASSFILE + ".tmp"
)


print()
print("Saving new model...")


torch.save(
    best_state,
    temporary_model
)


with open(
    temporary_classfile,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        classes,
        file,
        indent=4,
        ensure_ascii=False
    )


# Replace old files

os.replace(
    temporary_model,
    MODEL
)

os.replace(
    temporary_classfile,
    CLASSFILE
)


# ============================================================
# FINAL SAVE VERIFICATION
# ============================================================

print()
print("Verifying saved files...")


try:

    saved_checkpoint = torch.load(
        MODEL,
        map_location="cpu"
    )

    with open(
        CLASSFILE,
        "r",
        encoding="utf-8"
    ) as file:

        saved_classes = json.load(
            file
        )


except Exception as error:

    print()
    print(
        "ERROR: Could not reload saved files."
    )

    print(error)

    raise SystemExit(1)


saved_fc_weight_shape = tuple(
    saved_checkpoint[
        "fc.weight"
    ].shape
)


saved_fc_bias_shape = tuple(
    saved_checkpoint[
        "fc.bias"
    ].shape
)


print()
print(
    f"Saved model FC shape: "
    f"{saved_fc_weight_shape}"
)

print(
    f"Saved classes: "
    f"{len(saved_classes)}"
)


expected_saved_weight_shape = (
    len(saved_classes),
    512
)

expected_saved_bias_shape = (
    len(saved_classes),
)


if (
    saved_fc_weight_shape
    != expected_saved_weight_shape
):

    print()
    print(
        "================================"
    )

    print(
        "SAVE VERIFICATION FAILED"
    )

    print(
        "Model and classes.json do not match."
    )

    print(
        "================================"
    )

    raise SystemExit(1)


if (
    saved_fc_bias_shape
    != expected_saved_bias_shape
):

    print()
    print(
        "================================"
    )

    print(
        "SAVE VERIFICATION FAILED"
    )

    print(
        "FC bias and classes.json do not match."
    )

    print(
        "================================"
    )

    raise SystemExit(1)


if (
    saved_classes
    != classes
):

    print()
    print(
        "ERROR: Saved class list mismatch."
    )

    raise SystemExit(1)


# ============================================================
# TRAINING REPORT
# ============================================================

report = {

    "device": str(DEVICE),

    "gpu": (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU"
    ),

    "previous_class_count": len(
        old_classes
    ),

    "final_class_count": len(
        classes
    ),

    "new_classes": new_classes,

    "classes": classes,

    "user_images": total_user_images,

    "replay_images_per_class":
        REPLAY_PER_CLASS,

    "incremental_dataset_images":
        len(full_dataset),

    "training_images":
        len(training_indices),

    "validation_images":
        len(validation_indices),

    "best_validation_accuracy":
        best_accuracy,

    "epochs":
        EPOCHS,

    "learning_rate":
        LEARNING_RATE,

    "training_time_minutes":
        training_time / 60.0,

    "model_fc_shape":
        list(saved_fc_weight_shape),

    "status":
        "completed"
}


try:

    with open(
        REPORT,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False
        )

except Exception as error:

    print()
    print(
        "WARNING: Could not write training report."
    )

    print(error)


# ============================================================
# FINAL RESULT
# ============================================================

print()
print("================================")
print("INCREMENTAL TRAINING COMPLETED")
print("================================")

print()
print(
    f"Model saved: {MODEL}"
)

print(
    f"Classes: {len(saved_classes)}"
)

print(
    f"User images: {total_user_images}"
)

print(
    f"Replay images per old class: "
    f"{REPLAY_PER_CLASS}"
)

print(
    f"Best validation accuracy: "
    f"{best_accuracy:.2f}%"
)

print(
    f"Training time: "
    f"{training_time / 60.0:.2f} minutes"
)

print()
print("================================")
print("SAVE VERIFICATION PASSED")
print("================================")

print(
    f"Final model FC shape: "
    f"{saved_fc_weight_shape}"
)

print(
    f"Expected FC shape: "
    f"{expected_saved_weight_shape}"
)

print(
    f"Saved classes: "
    f"{len(saved_classes)}"
)

print()
print("Done!")
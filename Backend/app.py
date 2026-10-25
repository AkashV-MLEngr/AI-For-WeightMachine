from fastapi import FastAPI, File, UploadFile, Form
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

import io
import os
import uuid
import subprocess
import sys
import json


# =========================================================
# APP
# =========================================================

app = FastAPI(title="FreshVision AI")


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# FOLDERS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# AI model is inside Backend/AI model
MODEL_DIR = os.path.join(
    BASE_DIR,
    "AI model"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "fruit_vegetable_resnet18.pth"
)

CLASS_FILE = os.path.join(
    MODEL_DIR,
    "classes.json"
)

# User training data
TRAINING_DATA_DIR = os.path.join(
    BASE_DIR,
    "training_data"
)

os.makedirs(
    TRAINING_DATA_DIR,
    exist_ok=True
)


# =========================================================
# ORIGINAL 36 CLASSES
# =========================================================

DEFAULT_CLASSES = [
    "apple",
    "banana",
    "beetroot",
    "bell pepper",
    "cabbage",
    "capsicum",
    "carrot",
    "cauliflower",
    "chilli pepper",
    "corn",
    "cucumber",
    "eggplant",
    "garlic",
    "ginger",
    "grapes",
    "jalepeno",
    "kiwi",
    "lemon",
    "lettuce",
    "mango",
    "onion",
    "orange",
    "paprika",
    "pear",
    "peas",
    "pineapple",
    "pomegranate",
    "potato",
    "raddish",
    "soy beans",
    "spinach",
    "sweetcorn",
    "sweetpotato",
    "tomato",
    "turnip",
    "watermelon"
]


# =========================================================
# LOAD CLASS LIST
# =========================================================

def load_class_names():

    if os.path.exists(CLASS_FILE):

        try:

            with open(
                CLASS_FILE,
                "r"
            ) as f:

                classes = json.load(f)

            if isinstance(classes, list) and len(classes) > 0:

                print(
                    "Loaded classes from classes.json:",
                    len(classes)
                )

                return classes

        except Exception as e:

            print(
                "Could not read classes.json:",
                e
            )

    print(
        "Using default 36 classes."
    )

    return DEFAULT_CLASSES.copy()


class_names = load_class_names()


# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Device:",
    device
)


# =========================================================
# LOAD MODEL
# =========================================================

def load_model():

    global model
    global class_names

    print(
        "\nLoading AI model..."
    )

    # Reload class list
    class_names = load_class_names()

    new_model = models.resnet18(
        weights=None
    )

    new_model.fc = nn.Linear(
        new_model.fc.in_features,
        len(class_names)
    )

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    new_model.load_state_dict(
        checkpoint
    )

    new_model = new_model.to(
        device
    )

    new_model.eval()

    model = new_model

    print(
        "AI model loaded successfully!"
    )

    print(
        "Model classes:",
        len(class_names)
    )


# Load model when backend starts
load_model()


# =========================================================
# IMAGE TRANSFORM
# =========================================================

transform = transforms.Compose([
    transforms.Resize(
        (224, 224)
    ),
    transforms.ToTensor()
])


# =========================================================
# HOME PAGE
# =========================================================

@app.get("/")
def home():

    return {
        "success": True,
        "message": "FreshVision AI Backend is running!",
        "model": "ResNet18",
        "classes": len(class_names),
        "device": str(device)
    }


# =========================================================
# AI PREDICTION
# =========================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    try:

        # Read uploaded image
        contents = await file.read()

        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

        # Transform image
        image_tensor = transform(
            image
        )

        image_tensor = image_tensor.unsqueeze(
            0
        )

        image_tensor = image_tensor.to(
            device
        )

        # AI prediction
        with torch.no_grad():

            output = model(
                image_tensor
            )

            probabilities = torch.softmax(
                output,
                dim=1
            )

            confidence, predicted = torch.max(
                probabilities,
                dim=1
            )

        confidence_value = confidence.item()

        predicted_index = predicted.item()

        # Check class index
        if predicted_index >= len(class_names):

            return {
                "success": False,
                "error": "Invalid model class index."
            }

        predicted_class = class_names[
            predicted_index
        ]

        # Unknown detection
        if confidence_value >= 0.70:

            result = predicted_class

        else:

            result = "Unknown"

        return {
            "success": True,
            "prediction": result
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }


# =========================================================
# ADD TRAINING IMAGE
# =========================================================

@app.post("/add-training-image")
async def add_training_image(
    file: UploadFile = File(...),
    class_name: str = Form(...)
):

    try:

        # Clean class name
        class_name = class_name.strip().lower()

        if not class_name:

            return {
                "success": False,
                "error": "Class name is required."
            }

        # Allow letters, numbers, spaces, underscore and hyphen
        class_name = "".join(
            c
            for c in class_name
            if c.isalnum()
            or c in (" ", "_", "-")
        ).strip()

        if not class_name:

            return {
                "success": False,
                "error": "Invalid class name."
            }

        # Create class folder
        class_folder = os.path.join(
            TRAINING_DATA_DIR,
            class_name
        )

        os.makedirs(
            class_folder,
            exist_ok=True
        )

        # Read image
        contents = await file.read()

        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

        # Unique filename
        unique_id = uuid.uuid4().hex[:10]

        filename = (
            f"user_{unique_id}.jpg"
        )

        file_path = os.path.join(
            class_folder,
            filename
        )

        # Save image
        image.save(
            file_path,
            "JPEG"
        )

        print(
            "Training image saved:",
            file_path
        )

        return {
            "success": True,
            "message": "Training image added successfully!",
            "class_name": class_name,
            "filename": filename
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }


# =========================================================
# ADD MULTIPLE TRAINING IMAGES / FOLDER
# =========================================================

ALLOWED_IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp"
}


def clean_class_name(value: str):
    """Make a safe class-folder name."""
    value = value.strip().lower()
    value = "".join(
        c for c in value
        if c.isalnum() or c in (" ", "_", "-")
    ).strip()
    return value


@app.post("/add-training-images")
async def add_training_images(
    files: List[UploadFile] = File(...),
    class_name: Optional[str] = Form(None)
):
    """
    Supports:
      1. Multiple images + class_name
      2. One class folder selected with webkitdirectory
      3. A parent folder containing multiple class folders

    For folder uploads, UploadFile.filename contains the relative path,
    e.g. Tomato/image1.jpg or Training/Tomato/image1.jpg.
    """

    saved = 0
    skipped = 0
    errors = []
    class_counts = {}

    try:
        # Optional class_name field.
        raw_class_name = class_name
        default_class = clean_class_name(
            class_name if class_name is not None else ""
        )

        if raw_class_name is not None and not default_class:
            return {
                "success": False,
                "message": "Invalid class name."
            }

        if not files:
            return {
                "success": False,
                "message": "No image files were uploaded."
            }

        for file in files:
            try:
                original_name = file.filename or ""
                normalized_name = original_name.replace("\\", "/")
                parts = [p for p in normalized_name.split("/") if p not in ("", ".")]

                extension = os.path.splitext(normalized_name)[1].lower()
                if extension not in ALLOWED_IMAGE_EXTENSIONS:
                    skipped += 1
                    continue

                # If class_name is supplied, every uploaded image belongs there.
                # Otherwise infer the class from the folder path.
                if default_class:
                    target_class = default_class
                else:
                    if len(parts) < 2:
                        skipped += 1
                        errors.append(
                            f"{original_name}: class folder not found"
                        )
                        continue

                    # For a folder upload, use the directory immediately
                    # containing the image as the class name.
                    target_class = clean_class_name(parts[-2])

                    if not target_class:
                        skipped += 1
                        errors.append(
                            f"{original_name}: invalid class folder"
                        )
                        continue

                class_folder = os.path.join(
                    TRAINING_DATA_DIR,
                    target_class
                )
                os.makedirs(class_folder, exist_ok=True)

                contents = await file.read()
                if not contents:
                    skipped += 1
                    continue

                # Validate that the upload is actually an image.
                image = Image.open(io.BytesIO(contents)).convert("RGB")

                unique_id = uuid.uuid4().hex[:12]
                filename = f"user_{unique_id}.jpg"
                file_path = os.path.join(class_folder, filename)

                image.save(file_path, "JPEG", quality=95)

                saved += 1
                class_counts[target_class] = class_counts.get(target_class, 0) + 1

            except Exception as file_error:
                skipped += 1
                errors.append(
                    f"{file.filename or 'unknown file'}: {str(file_error)}"
                )

        return {
            "success": saved > 0,
            "message": f"{saved} training image(s) added successfully.",
            "saved": saved,
            "skipped": skipped,
            "classes": class_counts,
            "errors": errors[:20]
        }

    except Exception as e:
        return {
            "success": False,
            "message": "Could not add training images.",
            "error": str(e)
        }


# =========================================================
# INCREMENTAL TRAINING
# =========================================================

TRAINING_SCRIPT = os.path.join(
    BASE_DIR,
    "train_model_incremental.py"
)

TRAINING_LOG = os.path.join(
    BASE_DIR,
    "incremental_training.log"
)

# Training status is kept in memory while the backend is running.
# The log is also checked so the website can recover the real status
# even if a previous request/poll missed the state transition.
training_state = {
    "running": False,
    "success": None,
    "message": "No training has been started.",
    "pid": None
}

import threading

training_lock = threading.Lock()


def _read_training_log():
    """Return the latest training log text."""
    if not os.path.exists(TRAINING_LOG):
        return ""

    try:
        with open(
            TRAINING_LOG,
            "r",
            encoding="utf-8",
            errors="replace"
        ) as f:
            return f.read()
    except Exception:
        return ""


def _get_log_tail():
    """Return only the latest 40 lines for the frontend."""
    log_text = _read_training_log()
    if not log_text:
        return ""

    return "\n".join(log_text.splitlines()[-40:])


def _run_incremental_training():
    global training_state

    try:
        print("\n================================")
        print("Starting incremental model training...")
        print("================================")

        # Always start with a fresh log for the current training run.
        with open(
            TRAINING_LOG,
            "w",
            encoding="utf-8"
        ) as log_file:

            process = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    TRAINING_SCRIPT
                ],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True
            )

            with training_lock:
                training_state["pid"] = process.pid
                training_state["message"] = "Incremental training is running..."

            return_code = process.wait()

        if return_code != 0:
            with training_lock:
                training_state["running"] = False
                training_state["success"] = False
                training_state["message"] = (
                    "Incremental training failed. "
                    "Check the training log."
                )
                training_state["pid"] = None

            print(
                "Incremental training failed. "
                "Check incremental_training.log"
            )
            return

        # Training script reached its normal end.
        print("Incremental training completed!")
        print("\nReloading updated AI model...")

        try:
            load_model()
            print("Updated model loaded!")

            with training_lock:
                training_state["running"] = False
                training_state["success"] = True
                training_state["message"] = (
                    "Model trained and updated successfully!"
                )
                training_state["pid"] = None

        except Exception as model_error:
            with training_lock:
                training_state["running"] = False
                training_state["success"] = False
                training_state["message"] = (
                    f"Training finished, but model reload failed: "
                    f"{model_error}"
                )
                training_state["pid"] = None

    except Exception as e:
        with training_lock:
            training_state["running"] = False
            training_state["success"] = False
            training_state["message"] = f"Training error: {str(e)}"
            training_state["pid"] = None

        print("Training error:", e)


@app.post("/train-model")
async def train_model():
    global training_state

    try:
        if not os.path.exists(TRAINING_SCRIPT):
            return {
                "success": False,
                "running": False,
                "message": "Incremental training script not found.",
                "script": TRAINING_SCRIPT
            }

        with training_lock:
            if training_state["running"]:
                return {
                    "success": False,
                    "running": True,
                    "message": "Training is already running.",
                    "pid": training_state["pid"]
                }

            training_state = {
                "running": True,
                "success": None,
                "message": "Incremental training started.",
                "pid": None
            }

        # Non-blocking background worker.
        thread = threading.Thread(
            target=_run_incremental_training,
            daemon=False
        )
        thread.start()

        return {
            "success": True,
            "running": True,
            "message": "Incremental training started successfully!",
            "training_script": "train_model_incremental.py"
        }

    except Exception as e:
        with training_lock:
            training_state["running"] = False
            training_state["success"] = False
            training_state["message"] = str(e)
            training_state["pid"] = None

        return {
            "success": False,
            "running": False,
            "message": "Could not start training.",
            "error": str(e)
        }


@app.get("/training-status")
async def training_status():
    """
    Return the current training state and the latest log.

    The log is checked for explicit completion/failure markers so the
    frontend does not remain stuck in 'Training...' if it misses the
    in-memory state transition.
    """
    global training_state

    log_text = _read_training_log()

    with training_lock:
        current_running = training_state["running"]
        current_success = training_state["success"]
        current_message = training_state["message"]
        current_pid = training_state["pid"]

    # Recover completion state directly from the training log.
    if current_running:
        if "INCREMENTAL TRAINING COMPLETED" in log_text:
            current_running = False
            current_success = True
            current_message = "Model trained and updated successfully!"

            with training_lock:
                training_state["running"] = False
                training_state["success"] = True
                training_state["message"] = current_message
                training_state["pid"] = None

        elif "Incremental training failed." in log_text:
            # The worker writes this message only after the subprocess
            # has returned a non-zero exit code.
            current_running = False
            current_success = False
            current_message = (
                "Incremental training failed. "
                "Check the training log."
            )

            with training_lock:
                training_state["running"] = False
                training_state["success"] = False
                training_state["message"] = current_message
                training_state["pid"] = None

    return {
        "running": current_running,
        "success": current_success,
        "message": current_message,
        "pid": current_pid,
        "log": _get_log_tail()
    }


# =========================================================
# CUSTOM OPENAPI FOR MULTIPLE FILE UPLOAD
# =========================================================

from fastapi.openapi.utils import get_openapi

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version="1.0.0",
        description="FreshVision AI Backend API",
        routes=app.routes,
    )

    # Force Swagger UI to render /add-training-images as a real
    # multipart file-array input. FastAPI/Pydantic 2 can otherwise
    # describe the array as generic strings on some versions.
    endpoint = schema.get("paths", {}).get("/add-training-images", {}).get("post")
    if endpoint is not None:
        endpoint["requestBody"] = {
            "required": True,
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "files": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "format": "binary"
                                },
                                "minItems": 1,
                                "description": (
                                    "Select one or more JPG/JPEG/PNG/WEBP/BMP images. "
                                    "All selected files are sent using the same 'files' field."
                                )
                            },
                            "class_name": {
                                "type": "string",
                                "nullable": True,
                                "description": (
                                    "Optional class name. Leave empty for folder-based "
                                    "class detection."
                                )
                            }
                        },
                        "required": ["files"]
                    },
                    "encoding": {
                        "files": {
                            "style": "form",
                            "explode": True
                        }
                    }
                }
            }
        }

    app.openapi_schema = schema
    return app.openapi_schema

app.openapi = custom_openapi

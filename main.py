import io
import json
import torch
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from PIL import Image
from models import (
    load_rice_model,
    load_flower_model,
    RICE_TRANSFORM,
    FLOWER_TRANSFORM,
)

app = FastAPI()

with open("class_mapping.json", "r") as f:
    rice_mapping = json.load(f)

with open("flower_mapping.json", "r") as f:
    flower_mapping = json.load(f)

MODELS = {
    "rice": {
        "model": load_rice_model("rice_model_weights.pth", num_classes=20),
        "mapping": rice_mapping,
        "transform": RICE_TRANSFORM,
    },
    "flower": {
        "model": load_flower_model("flower_resnet50_weights.pth", num_classes=10),
        "mapping": flower_mapping,
        "transform": FLOWER_TRANSFORM,
    },
}


def resolve_class_name(mapping, idx: int) -> str:
    if isinstance(mapping, list):
        return mapping[idx] if 0 <= idx < len(mapping) else f"Unknown Class ({idx})"
    elif isinstance(mapping, dict):
        return mapping.get(str(idx), mapping.get(idx, f"Unknown Class ({idx})"))
    return f"Unknown Class ({idx})"


@app.post("/predict")
async def predict(
    model_type: str = Form("rice"),
    file: UploadFile = File(...)
):
    if model_type not in MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid model_type: {model_type}")

    selected = MODELS[model_type]
    model = selected["model"]
    mapping = selected["mapping"]
    transform = selected["transform"]

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    tensor = transform(image).unsqueeze(0)  # type: ignore
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted_idx = torch.max(outputs, 1)
        prediction_index = int(predicted_idx.item())
        class_name = resolve_class_name(mapping, prediction_index)

    return {
        "model_used": model_type,
        "prediction": class_name
    }

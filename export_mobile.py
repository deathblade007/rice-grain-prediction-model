import os
import shutil
import torch
from models import RiceModel, load_flower_model

ASSETS_DIR = os.path.join("android-app", "app", "src", "main", "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

# 1. Export Rice Model (20 classes, 200x200)
rice_model = RiceModel(num_classes=20)
rice_state = torch.load("rice_model_weights.pth", map_location="cpu")
rice_model.load_state_dict(rice_state)
rice_model.eval()

dummy_rice = torch.randn(1, 3, 200, 200)
torch.onnx.export(
    rice_model,
    dummy_rice,
    os.path.join(ASSETS_DIR, "rice_model.onnx"),
    input_names=["input"],
    output_names=["output"],
    opset_version=17,
    export_params=True,
    dynamo=False
)

# 2. Export Flower Model (10 classes, 224x224)
flower_model = load_flower_model("flower_resnet50_weights.pth", num_classes=10)
flower_model.eval()

dummy_flower = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    flower_model,
    dummy_flower,
    os.path.join(ASSETS_DIR, "flower_model.onnx"),
    input_names=["input"],
    output_names=["output"],
    opset_version=17,
    export_params=True,
    dynamo=False
)

# 3. Copy JSON mappings into Android assets
shutil.copy("class_mapping.json", os.path.join(ASSETS_DIR, "rice_mapping.json"))
shutil.copy("flower_mapping.json", os.path.join(ASSETS_DIR, "flower_mapping.json"))

print("Successfully exported models and mappings to Android assets!")
import io
import json  
import torch
import torch.nn as nn 
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from torchvision import transforms

class Model(nn.Module):
  def __init__(self,num_classes):
    super().__init__()
    self.model = nn.Sequential(
        nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding='same'),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2),

        nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding='same'),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2),

        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same'),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.AdaptiveAvgPool2d((1, 1)),

        nn.Flatten(),

        nn.Linear(in_features=128,out_features=128),
        nn.BatchNorm1d(128),
        nn.ReLU(),

        nn.Linear(in_features=128,out_features=128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(p=0.4),

        nn.Linear(in_features=128,out_features=num_classes),
    )

  def forward(self,x):
    return self.model(x)

app = FastAPI()

with open("class_mapping.json", "r") as f:
    class_mapping = json.load(f)


num_classes = 20 
model = Model(num_classes)
model.load_state_dict(torch.load("rice_model_weights.pth", map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    tensor = transform(image).unsqueeze(0) # type: ignore
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted_idx = torch.max(outputs, 1)
        
        # Keep the index as an integer
        prediction_index = predicted_idx.item() 
        
        # Access the list directly using the integer index
        try:
            class_name = class_mapping[prediction_index]
        except IndexError:
            class_name = f"Unknown Class ({prediction_index})"
    
    return {"prediction": class_name}

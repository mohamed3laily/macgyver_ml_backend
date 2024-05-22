import torch

# Specify the model name
model = 'best'
model_path = f"./{model}.pt"

try:
    # Load the model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
    print("Model loaded successfully")
except Exception as e:
    print(f"Failed to load model: {e}")

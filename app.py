import argparse
import io
import logging
from flask import Flask, request, jsonify
from PIL import Image, ImageDraw, ImageFont
import torch
from draw_box import draw_boxes_for_dailyCHUP , draw_boxes_for_inspection

app = Flask(__name__)
models = {}
model_path = "best.pt"  # Path to your custom model file

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route("/v1/inspection/<model>", methods=["POST"])
def predict_inspection(model):
    if request.method != "POST":
        return jsonify({"error": "Only POST method is allowed"}), 405
    
    if "image" not in request.files:
        return jsonify({"error": "No image file found in the request"}), 400

    try:
        im_file = request.files["image"]
        im_bytes = im_file.read()
        im = Image.open(io.BytesIO(im_bytes))
        im = im.resize((640, 640))
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return jsonify({"error": "Failed to process the image"}), 400

    if model not in models:
        return jsonify({"error": f"Model '{model}' not found"}), 404

    try:
        results = models[model](im, size=640)  # Adjust size as needed for faster inference
        detections = results.pandas().xyxy[0].to_dict(orient="records")
        # Draw bounding boxes and labels on the image
        im_with_boxes = draw_boxes_for_inspection(im.copy(), detections)
        # Resize the image to reduce its size
        im_with_boxes.thumbnail((1000, 1000))  # Adjust the size as needed
        # Convert the annotated image to bytes
        img_byte_array = io.BytesIO()
        im_with_boxes.save(img_byte_array, format='PNG')
        img_byte_array = img_byte_array.getvalue()
        return img_byte_array, 200, {'Content-Type': 'image/png'}
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return jsonify({"error": "Failed to make prediction"}), 500

## for the daily checking up page


@app.route("/v1/dailycheckup/<model>", methods=["POST"])
def predict_daily_checkup(model):
    if request.method != "POST":
        return jsonify({"error": "Only POST method is allowed"}), 405
    
    if "image" not in request.files:
        return jsonify({"error": "No image file found in the request"}), 400

    try:
        im_file = request.files["image"]
        im_bytes = im_file.read()
        im = Image.open(io.BytesIO(im_bytes))
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return jsonify({"error": "Failed to process the image"}), 400

    if model not in models:
        return jsonify({"error": f"Model '{model}' not found"}), 404

    try:
        results = models[model](im, size=640)  # Adjust size as needed for faster inference
        detections = results.pandas().xyxy[0].to_dict(orient="records")
        # Draw bounding boxes and labels on the image
        im_with_boxes = draw_boxes_for_dailyCHUP(im.copy(), detections)
        # Resize the image to reduce its size
        im_with_boxes.thumbnail((1000, 1000))  # Adjust the size as needed
        # Convert the annotated image to bytes
        img_byte_array = io.BytesIO()
        im_with_boxes.save(img_byte_array, format='PNG')
        img_byte_array = img_byte_array.getvalue()
        return img_byte_array, 200, {'Content-Type': 'image/png'}
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return jsonify({"error": "Failed to make prediction"}), 500        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask API exposing YOLOv5 model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    parser.add_argument("--model", nargs="+", default=["best"], help="model(s) to run, i.e. --model best")
    opt = parser.parse_args()

    for m in opt.model:
        try:
            models[m] = torch.hub.load("ultralytics/yolov5", 'custom', path=model_path, force_reload=False, skip_validation=True)
            logger.info(f"Model '{m}' loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model '{m}': {e}")

    app.run(host="0.0.0.0", port=opt.port)
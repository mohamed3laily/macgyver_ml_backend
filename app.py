import io
import logging
import base64
from flask import Flask, request, jsonify
from PIL import Image
import torch
import json
from draw_box import draw_boxes_for_dailyCHUP, draw_boxes_for_inspection

app = Flask(__name__)
models = {}
model_path = "best.pt"  # Path to your custom model file

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the model directly
try:
    models["best"] = torch.hub.load("ultralytics/yolov5", 'custom', path=model_path, force_reload=True, skip_validation=True)
    logger.info(f"Model 'best' loaded successfully")
except Exception as e:
    logger.error(f"Error loading model 'best': {e}")

@app.route("/v1/inspection/best", methods=["POST"])
def predict_inspection():
    if request.method != "POST":
        return jsonify({"error": "Only POST method is allowed"}), 405

    if "image" not in request.files:
        return jsonify({"error": "No image file found in the request"}), 400

    try:
        im_file = request.files["image"]
        im_bytes = im_file.read()
        im = Image.open(io.BytesIO(im_bytes))
        
        # Resize the image to 80% of its original dimensions
        width, height = im.size
        new_width = int(width * 0.95)
        new_height = int(height * 0.95)
        im = im.resize((new_width, new_height), Image.LANCZOS)
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return jsonify({"error": "Failed to process the image"}), 400

    try:
        results = models["best"](im)  # Adjust size as needed
        detections = results.pandas().xyxy[0].to_dict(orient="records")

        # Prepare prediction JSON
        prediction_json = json.dumps(detections)

        # Draw bounding boxes on a copy of the image
        im_with_boxes = draw_boxes_for_inspection(im.copy(), detections)

        # Convert the annotated image to bytes
        img_byte_array = io.BytesIO()
        im_with_boxes.save(img_byte_array, format='JPEG', quality=95)
        img_byte_array.seek(0)
        img_base64 = base64.b64encode(img_byte_array.getvalue()).decode('utf-8')

        # Create a single JSON response
        response = {
            "message": "Prediction successful",
            "prediction": json.loads(prediction_json),  # Convert prediction back to dict
            "image": f"data:image/png;base64,{img_base64}"
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return jsonify({"error": "Failed to make prediction"}), 500

@app.route("/v1/dailycheckup/best", methods=["POST"])
def predict_daily_checkup():
    if request.method != "POST":
        return jsonify({"error": "Only POST method is allowed"}), 405
    
    if "image" not in request.files:
        return jsonify({"error": "No image file found in the request"}), 400

    try:
        im_file = request.files["image"]
        im_bytes = im_file.read()
        im = Image.open(io.BytesIO(im_bytes))
        
        # Resize the image to 80% of its original dimensions
        width, height = im.size
        new_width = int(width * 0.9)
        new_height = int(height * 0.9)
        im = im.resize((new_width, new_height), Image.LANCZOS)
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return jsonify({"error": "Failed to process the image"}), 400

    try:
        results = models["best"](im)  # Adjust size as needed for faster inference
        detections = results.pandas().xyxy[0].to_dict(orient="records")
        
        # Draw bounding boxes and labels on the image
        im_with_boxes = draw_boxes_for_dailyCHUP(im.copy(), detections)
        
        # Convert the annotated image to bytes
        img_byte_array = io.BytesIO()
        im_with_boxes.save(img_byte_array, format='JPEG', quality=95)
        img_byte_array.seek(0)
        
        return img_byte_array.getvalue(), 200, {'Content-Type': 'image/png'}
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return jsonify({"error": "Failed to make prediction"}), 500        

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

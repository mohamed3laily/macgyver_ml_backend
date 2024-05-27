from PIL import ImageDraw, ImageFont

def draw_boxes_for_inspection(image, detections):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for detection in detections:
        xmin, ymin, xmax, ymax = detection["xmin"], detection["ymin"], detection["xmax"], detection["ymax"]
        class_name = detection["name"]
        confidence = detection["confidence"]
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=4)
        # Adjust font size and color
        font_size = max(15, int((xmax - xmin) / 20))
        font = ImageFont.truetype("arial.ttf", size=font_size)
        draw.text((xmin, ymin), f"{class_name}: {confidence:.2f}", fill="white", font=font)
    return image


def draw_boxes_for_dailyCHUP(image, detections):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for detection in detections:
        xmin, ymin, xmax, ymax = detection["xmin"], detection["ymin"], detection["xmax"], detection["ymax"]
        class_name = detection["name"]
        image_lable = class_name.split("_")[0]
        if image_lable == "Coolant":
            image_lable = "Coolant tank"
        confidence = detection["confidence"]
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=4)
        # Adjust font size and color
        font_size = max(15, int((xmax - xmin) / 20))
        font = ImageFont.truetype("arial.ttf", size=font_size)
        draw.text((xmin, ymin), f"{image_lable}", fill="white", font=font)
    return image   

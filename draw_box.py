from PIL import Image, ImageDraw, ImageFont
def draw_boxes_for_inspection(image, detections):
    draw = ImageDraw.Draw(image)
    font_size = 80
    font = ImageFont.truetype("arial.ttf", size=font_size)
    additional_offset = 10  
    
    for detection in detections:
        xmin, ymin, xmax, ymax = detection["xmin"], detection["ymin"], detection["xmax"], detection["ymax"]
        class_name = detection["name"]
        confidence = detection["confidence"]
        
        text = f"{class_name}: {confidence:.2f}"
        estimated_text_height = font_size + additional_offset
        estimated_text_width = font_size * len(text) // 2  
        text_position = (xmin, ymin - estimated_text_height if ymin - estimated_text_height > 0 else ymin)
        background_rect = [xmin, text_position[1], xmin + estimated_text_width, text_position[1] + font_size]
        draw.rectangle(background_rect, fill="red")
        draw.text(text_position, text, fill="white", font=font)
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=4)
        
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
        font_size = 50
        font = ImageFont.truetype("arial.ttf", size=font_size)
        draw.text((xmin, ymin), f"{image_lable}", fill="white", font=font)
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=4)
        
    return image

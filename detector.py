import os
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import cv2
import numpy as np
import torch


class PhoneDetector:
    def __init__(self, model_path="yolov8m.pt"):  # More accurate model than yolov8n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_path).to(self.device)
        self.phone_class_id = 67

        # Font for label
        self.font = ImageFont.load_default()

    def detect_phones(self, image_path):
        img = cv2.imread(image_path)
        results = self.model(img, conf=0.25)
        detections = []

        for result in results:
            if result.boxes is not None:
                for box in result.boxes.data.cpu().numpy():
                    if len(box) >= 6:
                        x1, y1, x2, y2, conf, class_id = box[:6]
                        if int(class_id) == self.phone_class_id:
                            detections.append((int(x1), int(y1), int(x2), int(y2)))

        # Convert to PIL image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)

        # Draw boxes
        for box in detections:
            draw.rectangle(box, outline="green", width=3)

        # Label text
        label = f"Phone in hand detected: {len(detections)}" if detections else "No phone in hand detected: 0"

        # Create white strip
        text_bbox = draw.textbbox((0, 0), label, font=self.font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        new_height = pil_img.height + text_height + 20
        final_img = Image.new("RGB", (pil_img.width, new_height), (255, 255, 255))
        final_img.paste(pil_img, (0, 0))

        # Draw label in white strip
        draw_final = ImageDraw.Draw(final_img)
        draw_final.text(((pil_img.width - text_width) // 2, pil_img.height + 10), label, fill="black", font=self.font)

        # Save output
        output_path = os.path.join("output", os.path.basename(image_path))
        os.makedirs("output", exist_ok=True)
        final_img.save(output_path)

        return output_path, label
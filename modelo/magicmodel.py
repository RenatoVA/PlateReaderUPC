from ultralytics import YOLO
import cv2
import easyocr
import numpy as np
import base64
import io
from PIL import Image


class LicensePlateRecognizer:
    def __init__(self, model_path="license_plate_detector.pt"):
        # Cargar el modelo YOLO
        self.model = YOLO(model_path)
        # Inicializar EasyOCR
        self.reader = easyocr.Reader(['en'])

    # Funciones de Preprocesamiento
    @staticmethod
    def get_grayscale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def remove_noise(image):
        return cv2.GaussianBlur(image, (5, 5), 0)

    @staticmethod
    def resize_image(image, scale):
        width = int(image.shape[1] * scale)
        height = int(image.shape[0] * scale)
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
        return resized

    @staticmethod
    def sharpen_image(image):
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        sharpened = cv2.filter2D(image, -1, kernel)
        return sharpened

    def process_plate(self, plate):
        # Aplicar preprocesamiento a una placa recortada
        plate_gray = self.get_grayscale(plate)
        plate_resized = self.resize_image(plate_gray, 2.0)
        plate_sharpened = self.sharpen_image(plate_resized)
        plate_cleaned = self.remove_noise(plate_sharpened)
        return plate_cleaned

    def encode_image_to_base64(self, image):
        """Codifica una imagen de OpenCV en formato Base64."""
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')

    def recognize(self, image):
        try:
            # Realizar inferencia con YOLO
            results = self.model.predict(source=image, conf=0.5, save=False)

            # Recortar placas detectadas y dibujar cuadros en la imagen original
            detected_texts = []
            for result in results:
                for box in result.boxes:
                    x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                    confidence = box.conf[0].item() * 100
                    class_id = int(box.cls[0])

                    # Dibujar cuadro en la imagen original
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    label = f"Confidence: {confidence:.2f}%"
                    cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Recortar la placa y procesarla
                    cropped_plate = image[y_min:y_max, x_min:x_max]
                    preprocessed_plate = self.process_plate(cropped_plate)

                    # Extraer texto con EasyOCR
                    result = self.reader.readtext(preprocessed_plate)
                    for bbox, text, conf in result:
                        detected_texts.append({"text": text, "confidence": conf})

            # Verificar si se extrajo algún texto
            if not detected_texts:
                return {"error": "No se pudo extraer texto de las placas detectadas."}

            # Convertir la imagen procesada a Base64
            processed_image_base64 = self.encode_image_to_base64(image)

            # Devolver resultados
            return {
                "plates_detected": len(detected_texts),
                "texts": detected_texts,
                "processed_image": processed_image_base64
            }

        except Exception as e:
            return {"error": str(e)}

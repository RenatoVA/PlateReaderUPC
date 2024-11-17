from flask import Flask, render_template, request, jsonify
from modelo.magicmodel import LicensePlateRecognizer
import numpy as np
import base64
import cv2

app = Flask(__name__)

model=LicensePlateRecognizer(model_path="modelo/plate_reader.pt")


def decode_base64_to_image(base64_string):
    """Convierte una imagen codificada en Base64 a un array de NumPy."""
    try:
        # Decodificar la imagen Base64
        image_data = base64.b64decode(base64_string)
        # Convertir a NumPy array
        np_arr = np.frombuffer(image_data, np.uint8)
        # Decodificar la imagen como formato OpenCV
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        raise ValueError("Error al decodificar la imagen en Base64: " + str(e))
#llamada al index que es la pagina principal
@app.route('/')
def home():
    return render_template('index.html')  
#metodo de preccion, aqui hay que modificarlo por el modelo nuevo
@app.route('/extract_plate_text', methods=['POST'])
def extract_plate_text():
    """Endpoint para procesar una imagen en Base64 y extraer texto de placas."""
    try:
        data = request.get_json()

        if 'image_base64' not in data:
            return jsonify({"error": "Falta la clave 'image_base64' en la solicitud."}), 400

        base64_string = data['image_base64']
        image = decode_base64_to_image(base64_string)

        if image is None:
            return jsonify({"error": "No se pudo decodificar la imagen en Base64."}), 400

        # Usar el reconocedor de placas
        result = model.recognize(image)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
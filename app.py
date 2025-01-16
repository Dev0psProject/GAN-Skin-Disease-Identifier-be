# app.py - Flask backend for skin disease classification
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from flask_cors import CORS

app = Flask(__name__)

CORS(app)

# Load the pre-trained model
model = load_model('model/skin_disease_classifier.h5')

# Define class labels based on the dataset
class_labels = ['Biduran', 'Keloid', 'Kurap', 'Melanoma', 'Vitiligo']  # Replace with actual labels in order

# Endpoint for prediction
@app.route('/api/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    # Load and preprocess the image
    image_file = request.files['image']
    image = Image.open(image_file).convert('RGB')
    image = image.resize((128, 128))  # Resize to match model input shape
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Expand dimensions to fit model input

    # make prediction
    predictions = model.predict(image)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_labels[predicted_class_index]
    confidence = predictions[0][predicted_class_index]

    # Descriptions for each skin disease
    descriptions = {
        'Biduran': 'Biduran, atau urtikaria, adalah kondisi kulit yang ditandai dengan gatal, kemerahan, dan munculnya bentol-bentol merah atau putih yang sering berpindah-pindah tempat. Kondisi ini biasanya disebabkan oleh reaksi alergi, infeksi, atau stres.',
        'Keloid': 'Keloid adalah jaringan parut yang tumbuh berlebihan di area yang terluka, menyebabkan benjolan yang keras dan menonjol. Keloid bisa muncul setelah luka sembuh, seperti setelah operasi, luka bakar, atau jerawat.',
        'Kurap': 'Kurap adalah infeksi jamur pada kulit yang menyebabkan ruam merah, bersisik, dan gatal. Infeksi ini bisa terjadi di berbagai bagian tubuh, seperti kulit kepala, kaki (tinea pedis), atau tubuh (tinea corporis).',
        'Melanoma': 'Melanoma adalah jenis kanker kulit yang berkembang dari sel pigmentasi kulit (melanosit). Melanoma sering dimulai sebagai tanda atau tahi lalat baru yang tidak biasa atau perubahan pada tahi lalat yang ada.',
        'Vitiligo': 'Vitiligo adalah kondisi di mana kulit kehilangan pigmen melanin, menyebabkan bercak-bercak putih di berbagai bagian tubuh. Penyebabnya belum sepenuhnya dipahami, namun diperkirakan merupakan gangguan autoimun.'
    }

    # Send response
    return jsonify({
        'disease': predicted_class,
        'confidence': float(confidence),
        'description': descriptions[predicted_class]
    })

if __name__ == '__main__':
    app.run(debug=True)

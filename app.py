from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
app = Flask(__name__)

# Load the trained model
model_path = './my_model.h5'
model = load_model(model_path)

# Assuming `y_train` is your list of labels
y_train = ['Eczema', 'Psoriasis']
le = LabelEncoder()
le.fit(y_train)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['image']
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224, 224))
        img = preprocess_input(np.array([img]))

        predictions = model.predict(img)
        predicted_class_index = np.argmax(predictions)
        predicted_class = le.classes_[predicted_class_index]

        # Suggest doctor and cure tips
        doctor_name, cure_tips = get_doctor_suggestion(predicted_class)

        return jsonify({'prediction': predicted_class, 'doctor': doctor_name, 'cure_tips': cure_tips})
    except Exception as e:
        return jsonify({'error': str(e)})

def get_doctor_suggestion(predicted_class):
    if predicted_class == 'Psoriasis':
        return 'Dr. Miss Ekta', 'Psoriasis is a chronic skin condition. Dr. Miss Ekta suggests keeping your skin moisturized, avoiding triggers like stress and certain foods, and using prescribed medications and creams.'
    elif predicted_class == 'Eczema':
        return 'Dr. Ravindran', 'Eczema, also known as atopic dermatitis, can be managed with the right care. Dr. Ravindran suggests keeping your skin well-hydrated, identifying and avoiding triggers, and using prescribed creams and ointments as recommended.'

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, jsonify
from celery import Celery
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten
from PIL import Image
import io

app = Flask(__name__)

def make_celery(app_name=__name__):
    celery = Celery(app_name, broker='redis://localhost:6379/0')
    celery.config_from_object('celeryconfig')
    return celery

celery = make_celery()

@celery.task()
def train_model_async():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    train_images = np.random.rand(100, 28, 28)
    train_labels = np.random.randint(10, size=(100,))
    model.fit(train_images, train_labels, epochs=1)
    model.save('mnist_model.h5')
    return 'Model trained successfully'

@app.route('/train', methods=['POST'])
def train():
    train_model_async.delay()
    return jsonify({'message': 'Training started'})

@celery.task()
def predict_async(data):
    model = load_model('mnist_model.h5')
    prediction = model.predict(np.array([data]))[0]
    predicted_class = np.argmax(prediction)
    return predicted_class

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    file = request.files['image'].read()
    image = Image.open(io.BytesIO(file)).convert('L')
    image = np.resize(image, (28, 28)) / 255.0
    image = np.array(image, dtype=np.float32)
    task = predict_async.delay(image.tolist())
    return jsonify({'task_id': task.id}), 202

if __name__ == '__main__':
    app.run(debug=True)
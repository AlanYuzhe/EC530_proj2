from flask_restful import Resource, Api, fields, marshal_with, reqparse
from flask_sqlalchemy import SQLAlchemy
import os
import logging
from datetime import datetime
from flask import Flask, request, jsonify
from celery import Celery
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten
from PIL import Image
import io
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
api = Api(app)
app.config['SERVER_NAME'] = 'localhost:5000'
app.config['APPLICATION_ROOT'] = '/'
app.config['PREFERRED_URL_SCHEME'] = 'http'

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(os.path.abspath(os.path.dirname(__file__)), 'image_dataset.db')

logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@app.before_request
def before_request_logging():
    request.start_time = datetime.now()

@app.after_request
def after_request_logging(response):
    if request.path == '/favicon.ico':
        return response
    elif request.path.startswith('/static'):
        return response

    now = datetime.now()
    duration = (now - request.start_time).total_seconds()
    logging.info(f"Handled request: {request.path} [Status: {response.status_code}] in {duration}s")
    return response
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(os.path.abspath(os.path.dirname(__file__)), 'image_dataset.db')
db = SQLAlchemy(app)

class ImageDataset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.String(500), nullable=True)
    type = db.Column(db.String(50), nullable=True)
    creation_date = db.Column(db.String(50), nullable=True)
    image_format = db.Column(db.String(10), nullable=False)  # Image format (e.g., JPG, PNG)
    image_size = db.Column(db.String(50), nullable=True)  # Image dimensions (e.g., 1920x1080)
    image_path = db.Column(db.Text, nullable=False)  # Path or URL to the image
    labels = db.Column(db.Text, nullable=True)  # Image labels or categories for model training
    annotation = db.Column(db.Text, nullable=True)  # Annotations for the image

    def __init__(self, name, description, type, creation_date, image_format, image_size, image_path, labels, annotation):
        self.name = name
        self.description = description
        self.type = type
        self.creation_date = creation_date
        self.image_format = image_format
        self.image_size = image_size
        self.image_path = image_path
        self.labels = labels
        self.annotation = annotation

dataset_fields = {
    'id': fields.Integer,
    'name': fields.String,
    'description': fields.String,
    'type': fields.String,
    'creation_date': fields.String,
    'image_format': fields.String,
    'image_size': fields.String,
    'image_path': fields.String,
    'labels': fields.String,
    'annotation': fields.String,
}

parser = reqparse.RequestParser()
parser.add_argument('name', required=True, help="Name cannot be blank")
parser.add_argument('description')
parser.add_argument('type')
parser.add_argument('creation_date')
parser.add_argument('image_format', required=True)
parser.add_argument('image_size')
parser.add_argument('image_path', required=True)
parser.add_argument('labels')
parser.add_argument('annotation')

class ImageDatasetAPI(Resource):
    @marshal_with(dataset_fields)
    def get(self, dataset_id):
        dataset = ImageDataset.query.filter_by(id=dataset_id).first()
        if dataset:
            return dataset
        else:
            return {'message': 'Dataset not found'}, 404

    @marshal_with(dataset_fields)
    def post(self):
        args = parser.parse_args()
        dataset = ImageDataset(name=args['name'], description=args['description'], type=args['type'],
                               creation_date=args['creation_date'], image_format=args['image_format'],
                               image_size=args['image_size'], image_path=args['image_path'],
                               labels=args['labels'], annotation=args['annotation'])
        db.session.add(dataset)
        db.session.commit()
        return dataset, 201

api.add_resource(ImageDatasetAPI, '/datasets', '/datasets/<int:dataset_id>')


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        app.logger.error('No image provided')
        return jsonify({'message': 'No image provided'}), 400

    file = request.files['image']
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        file.save(file_path)
        app.logger.info(f'File {filename} saved successfully at {file_path}')
    except Exception as e:
        app.logger.error(f'Failed to save file {filename}: {e}')
        return jsonify({'message': 'Failed to save file'}), 500

    name = request.form.get('name')
    description = request.form.get('description')
    image_type = request.form.get('type')
    image_format = file.content_type
    image_size = os.path.getsize(file_path)

    if not name or not image_format:
        return jsonify({'message': 'Name and image format are required'}), 400

    try:
        new_image = ImageDataset(
            name=name,
            description=description,
            type=image_type,
            creation_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            image_format=image_format,
            image_size=str(image_size),
            image_path=file_path,
            labels=request.form.get('labels'),
            annotation=request.form.get('annotation')
        )
        db.session.add(new_image)
        db.session.commit()
        app.logger.info(f'New image record {new_image.id} added successfully')
    except Exception as e:
        app.logger.error(f'Error adding image record: {e}')
        db.session.rollback()
        return jsonify({'message': 'Database error'}), 500

    db.session.add(new_image)
    db.session.commit()

    return jsonify({'message': 'Image data uploaded successfully', 'id': new_image.id}), 201

def make_celery(app_name=__name__):
    celery = Celery(app_name, broker='redis://localhost:6379/0')
    celery.config_from_object('celeryconfig')
    return celery

celery = make_celery()

@celery.task()
def train_model_async():
    try:
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
        app.logger.info('Model training started')
    except Exception as e:
        app.logger.error(f'Model training failed: {e}')


@app.route('/train', methods=['POST'])
def train():
    try:
        train_model_async.delay()
        app.logger.info('Training task submitted successfully')
        return jsonify({'message': 'Training started'}), 200
    except Exception as e:
        app.logger.error(f'Training task submission failed: {e}')
        return jsonify({'error': 'Training task submission failed', 'details': str(e)}), 500

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
    with app.app_context():
        db.create_all()
    app.run(debug=True)

from flask import Flask, request
from flask_restful import Resource, Api, fields, marshal_with, reqparse
from flask_sqlalchemy import SQLAlchemy
import os
import logging
import datetime

app = Flask(__name__)
api = Api(app)

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
# Configure the SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(os.path.abspath(os.path.dirname(__file__)), 'image_dataset.db')
db = SQLAlchemy(app)

# Define the database model for an image dataset
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

# Define output data schema
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

# Define request parser
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

# Resource class for Image Dataset API
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

# Add the resource to the API
api.add_resource(ImageDatasetAPI, '/datasets', '/datasets/<int:dataset_id>')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  
    app.run(debug=False)

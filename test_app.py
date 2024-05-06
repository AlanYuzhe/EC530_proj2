import pytest
from flask import url_for
from app import app, db
from PIL import Image
import io
from unittest.mock import patch

@pytest.fixture
def client():
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    with app.app_context():
        db.create_all()
        with app.test_client() as client:
            yield client

def test_image_upload(client):
    with open("test.jpg", "rb") as img:
        img_data = img.read()
    data = {
        'name': 'Test Image',
        'description': 'A test image',
        'type': 'example',
        'creation_date': '2024-05-05',
        'image_size': '640x480',
        'image_format': 'JPG',
        'labels': 'test',
        'annotation': 'None'
    }

    data['image'] = (io.BytesIO(img_data), 'test.jpg')

    response = client.post(url_for('upload_image'), data=data, content_type='multipart/form-data')
    assert response.status_code == 201


@patch('app.train_model_async.delay')
def test_train_route(mock_train, client):
    response = client.post(url_for('train'))
    assert response.status_code == 200
    assert response.get_json() == {'message': 'Training started'}
    mock_train.assert_called_once()

@patch('app.predict_async.delay')
def test_predict_route(mock_predict, client):
    mock_task = patch('your_flask_file.celery.AsyncResult')
    mock_task.id = 'test_task_id'
    mock_predict.return_value = mock_task

    with open("test.jpg", "rb") as img:
        img_bytes = img.read()
    data = {
        'image': (io.BytesIO(img_bytes), 'test.jpg')
    }
    response = client.post(url_for('predict'), data=data, content_type='multipart/form-data', follow_redirects=True)
    assert response.status_code == 202
    assert 'task_id' in response.get_json()
    mock_predict.assert_called_once()


import pytest
from queue_flask import app
from unittest.mock import patch, MagicMock
import numpy as np
from PIL import Image
import io

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_train_route(client):
    with patch('queue_flask.train_model_async.delay') as mock_train:
        response = client.post('/train')
        assert response.status_code == 200
        assert response.json == {'message': 'Training started'}
        mock_train.assert_called_once()

def test_predict_route_no_image(client):
    response = client.post('/predict')
    assert response.status_code == 400
    assert response.json == {'error': 'No image provided'}

def test_predict_route_with_image(client):
    with patch('queue_flask.predict_async.delay') as mock_predict:
        mock_task = MagicMock()
        mock_task.id = 'test_task_id'
        mock_predict.return_value = mock_task

        # Creating a dummy image
        img = Image.new('L', (32, 32), color = 'black')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes = img_bytes.getvalue()

        response = client.post('/predict', data={'image': (io.BytesIO(img_bytes), 'test.png')})
        assert response.status_code == 202
        assert response.json == {'task_id': 'test_task_id'}
        mock_predict.assert_called_once()
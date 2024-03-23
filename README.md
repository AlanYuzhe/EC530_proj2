# EC530_proj2

## Image Dataset Management API

## Overview
This project is a simple yet powerful implementation of an image classification system built with Flask for the web framework, Celery for handling asynchronous tasks, and TensorFlow for the underlying machine learning model.  
 It includes an API for uploading images to be classified and for training the model with new datasets.

## Features
Logging: Detailed logging of each request with response status and duration.  
SQLite Database Integration: Utilizes SQLite for storing image dataset information.  
RESTful API: Provides GET and POST methods for accessing and adding dataset information.  
Asynchronous Task Handling: Uses Celery with Redis as the message broker to manage long-running tasks like model training and image prediction without blocking the main application.  

## Requirements
Flask  
Flask-RESTful  
Flask-SQLAlchemy  
Celery  
pytest  
redis  

## Testing the Application
To run the tests, ensure you have pytest installed in your development environment.   
The test suite includes checks for:  

The /train endpoint to ensure it triggers the asynchronous model training task and responds with the appropriate message.  
The /predict endpoint to validate behavior when no image is provided and verify that it correctly handles image data for prediction tasks, including mocking asynchronous prediction tasks and ensuring the expected response is returned.



# EC530_proj2

## Image Dataset Management API

## Overview
This project implements an image classification system using Flask as the web framework, Celery for handling asynchronous tasks, and TensorFlow for the machine learning model. It includes an API for uploading images for classification and training the model with new datasets.

## Features
Logging: Detailed logging for each request including response status and duration.
SQLite Database Integration: Uses SQLite to store information about the image datasets.
RESTful API: Offers GET and POST methods for accessing and updating dataset information.
Asynchronous Task Handling: Utilizes Celery with Redis as the message broker to manage long-running tasks such as model training and image prediction without blocking the main application.
Image Upload and Storage: Images are uploaded via the API and stored on the server, allowing easy access and management.
GitHub Actions Integration: Automates testing and other workflows using GitHub Actions.

## Requirements
To install the necessary requirements for this project, ensure you use the provided 'environment.yml' file which specifies all the necessary dependencies. This approach helps in maintaining consistency across various development environments. To setup the environment, run:
    `conda env create -f environment.yml`  
Activate the environment with:
    `conda activate <env_name>`  

## Key Dependencies
Flask
Flask-RESTful
Flask-SQLAlchemy
Celery
pytest
Redis

## Testing the Application
To run the tests, ensure you have pytest installed in your activated conda environment. You can run the test suite with the following command:
    `pytest`

The /train endpoint to ensure it triggers the asynchronous model training task and responds with the appropriate message.  
The /predict endpoint to validate behavior when no image is provided and verify that it correctly handles image data for prediction tasks, including mocking asynchronous prediction tasks and ensuring the expected response is returned.

## Using GitHub Actions
This project uses GitHub Actions for continuous integration, which automates testing and other workflows. The .github/workflows directory contains the workflow files that define the actions to be executed on push or pull requests.

## Managing Images
Images uploaded through the API are saved to a designated directory on the server as specified in the configuration. Ensure that the server has adequate storage and that the directory permissions are set correctly to allow for image storage.

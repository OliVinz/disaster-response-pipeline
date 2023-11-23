# disaster-response-pipeline
Udacity Data Science Nanodegree - Disaster Response Pipeline Project

## Table of Contents
1. Introduction
2. Files
3. Instructions
4. Acknowledgements

## Introduction and Importance
A project part of the data science nanodegree program. A model was trained on labelled disaster messages to categorise new messages in real time to simulate actual disaster message categorisation.
Projects and models like these can help business and organisations to quickly act in emergencies to classify messages and hand them to the appropriate places. This will allow for faster reaction times than without models.

## File Descriptions
### Folder: app
**run.py** - python script needed to run web app.

Folder: templates - html files (go.html & master.html) required for the web app.

### Folder: data
**disaster_messages.csv** - real messages sent during disaster events

**disaster_categories.csv** - corresponding categories of the messages

**process_data.py** - ETL pipeline used to load, clean, extract feature and store data in SQLite database

**ETL Pipeline Preparation.ipynb** - Jupyter Notebook used for analysis and to prepare ETL pipeline

**DisasterResponse.db** - cleaned data stored in table df_clean in SQlite database

### Folder: models
**train_classifier.py** - ML pipeline used to load cleaned data, train model and save trained model as pickle (.pkl) file

**ML Pipeline Preparation.ipynb** - Jupyter Notebook used for analysis and to prepare ML pipeline

## Instructions
1. Run the following commands in the root directory. The jupyter notebooks help with the explanation of the part of the codes.

    - To run the ETL pipeline
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run the ML pipeline 
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app folder using cd app
    `python run.py`

3. Go to http://0.0.0.0:3001/ to view the web app

## Licensing, Authors, Acknowledgements
* Credit to [Udacity](https://www.udacity.com/) for the course materials and script templates.
* Credit to [Appen](https://www.appen.com) for providing the datasets used to create the model and visualisations.

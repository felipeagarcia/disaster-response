# Disaster Response Pipeline Project

This project consists on a web page built to predict disasters based on text messages

### Prerequisites

In order to run this application, you need to install it's depencies. First, you will need Python 3 and Python 3 - Pip already installed

To install the python packages, run on the root folder:
```sh
pip install -r requirements.txt
```

We recommend to perform this step inside a virtual environment (see https://docs.python.org/3/library/venv.html).

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/disaster_response.db models/classifier.joblib`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Files:

- app/ - contains the webpage files
- app/run.py - the main file to run the app, loads the model and starts the web server
- data/ - contains the datasets and the datapipeline files
- data/process_data.py - preprocess the datasets and save the data into a db
- models/ - contains the machine learning pipeline file
- models/train_classifier.py - builds the machine learning pipeline, train, evaluate and save the model

import sys
import re
import pandas as pd
import numpy as np

from joblib import dump

from sqlalchemy import create_engine

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# regex to match urls
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def load_data(database_filepath):
    '''
    Loads the data from the database and splits it into X, y and
    category_names

    INPUTS
    database_filepath - string with the database path, e.g. "../my_database.db"

    OUTPUTS
    X - pandas series with messages column
    Y - pandas df with labels
    category_names - labels names
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    # create df from db file
    df = pd.read_sql("SELECT * FROM disaster_response", engine)

    # Creates the input df
    X = df['message']

    # Create the labels df
    Y = df.drop(columns=['message', 'original', 'genre', 'id'])

    # Name of each label
    category_names = Y.columns
    return np.array(X), np.array(Y), category_names

def tokenize(text):
    '''
    Normalize the text, lemmatize and returns tokens

    INPUTS
    text - string with text to be tokenized

    OUTPUTS
    clean_tokens - list of tokens
    '''
    # converts to lowercase and remove punctuation
    normalized_text = re.sub(r'[^a-zA-Z0-9]',' ', text.lower())

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    # lemmatize and tokenize the text
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    Creates and return the machine learning pipeline

    OUTPUTS
    cv - model pipeline with GridSearchCV
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()), 
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'vect__max_df': (0.75, 1.0),
        'vect__max_features': (None, 5000, 10000),
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__min_samples_split': [2, 3, 4],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Measures the model accuracy

    INPUTS
    model - trained model to be evaluated
    X_test - model test input series
    Y_test - model test labels df
    category_labels - list of category names
    '''
    y_pred = model.predict(X_test)
    labels = np.unique(y_pred)
    accuracy = (y_pred == Y_test).mean()

    print("Labels:", labels)
    print("Accuracy:", accuracy)
    print("\nBest Parameters:", model.best_params_)


def save_model(model, model_filepath):
    '''
    Saves the model as a joblib file

    INPUTS
    model - model to be saved
    model_filepath - path to save the model    
    '''
    dump(model, '{}.joblib'.format(model_filepath))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()

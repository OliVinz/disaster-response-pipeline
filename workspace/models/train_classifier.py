# import libraries
import re
import sys

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
import pickle


def load_data(database_filepath):
    """
       Function:
       loading data from a sqlite database

       Args:
       database_filepath: the path of the database

       Return:
       X (DataFrame) : features dataframe
       Y (DataFrame) : target dataframe
       category_names(list of str) : target labels list
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('messages', engine)
    X = df['message']  # Message Column
    Y = df[[col for col in df.columns if col not in ['id', 'message', 'original', 'genre']]]  
    category_names = list(Y.columns)
    return X, Y,category_names


def tokenize(text):
    """
    Function: tokenizing text by using regex, lemmatizing and cleaning steps
    Args:
      text(str): input text
    Return:
      cleaned_tokens(list of str): cleaned, tokenized list of tokens
    """
    regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected = re.findall(regex, text)
    for url in detected:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    cleaned_tokens = []
    for tok in tokens:
        t = lemmatizer.lemmatize(tok).lower().strip()
        cleaned_tokens.append(t)

    return cleaned_tokens


def build_model():
    """
     Function: building a model that is able to classify messages

     Return:
       cv(list of str): model
    """
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))])

    parameters = {#'clf__estimator__n_estimators': [20, 80, 150],
        #'clf__estimator__min_samples_split': [3, 5, 7],
        #'clf__class_weight':['balanced', 'balanced_subsample'],
        #'clf__max_depth':[3,4,5],
        'vect__max_features': (None, 2000),
        'tfidf__use_idf': (False, True)
                 }
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=2)
    return cv



def evaluate_model(model, X_test, Y_test, category_names):
    """
    Function: Function to evalute the created model and output several metrics
    Args:
    model: classification model
    X_test: messages
    Y_test: test target
    """
    y_pred = model.predict(X_test)
    for i in range(36):
        print(Y_test.columns[i], ':')
        print(classification_report(Y_test.iloc[:,i], y_pred[:,i], target_names=category_names))


def save_model(model, model_filepath):
    """
    Function: Store the created model in a pickle file
    Args:
    model: the model
    model_filepath (str): the path of pickle file to store
    """
    pickle.dump(model, open(model_filepath, 'wb'))


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
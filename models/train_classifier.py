# import libraries
import sys 

# For Data Loading/Processing
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy import inspect

# For Natural Language Processing
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

#For Machine Learning
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

from sklearn.model_selection import GridSearchCV

# For saving the Model
import pickle as pkl


#download and define stop words and lemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()

def load_data(database_filepath):
    '''
    load files from database and split into features and categories
    
    Args:
        database_filepath: path to database file
        
    Return:
        X:              dataframe containing features
        Y:              dataframe containing categories
        category_names: list containing category names
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('df', con = engine)
    
    #Split df into features (X) and targets (Y)
    categories = df.columns.difference(['id', 'message', 'original', 'genre'])
    X = df['message']
    Y = df[categories]
    
    return X, Y, categories

def tokenize(text):
    '''
    normalize, clean and tokenize text
    
    Args:
        text: input text for tokenization
    
    Return:
        clean_tokens: cleaned tokens
    '''
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    
    #simplify and lemmatize tokesn
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)  
    
    return clean_tokens


def build_model():
    '''
    builds and returns the classifier
    
    Args:
        None
        
    Return:
        cv: classifier
    '''
    #create the pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # Setting parameters for grid search
    parameters = {       
    # uncommenting more parameters will give better exploring power but will
    # increase processing time in a combinatorial way
        #'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000, 50000),
        #'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
        'tfidf__norm': ('l1', 'l2'), 
        # 'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 100, 150, 200],    
    }
    
    # Computing Grid Search
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=3, n_jobs=-1,verbose=1)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """ 
    Description: Predicts and evaluates input model.
  
    This function predicts categories for a testset and compares it to the actual real categories. It then prints the evaluation onto the terminal.
    Args: 
        model:          The trained machine learning model.
        X_test:         The test messages.
        Y_test:         The categories of those messages.
        category_names: A list containing the category names.
    
    Return: 
        None
    """
    # predict test values
    Y_pred = model.predict(X_test)

    # calculate f1-score, precision and recall
    for i in range(Y_test.shape[1]):
        result = classification_report(Y_test.iloc[:,i], Y_pred[:,i])
        print("Report for", category_names[i].strip(), ":")
        print(result)


def save_model(model, model_filepath):
    '''
    saves the model as pickle
    
    Args:
        model: model to be saved
        model_filepath: location and filename
        
    Return:
        None        
    '''
    
    # save model to pickle
    pkl.dump(model, open(model_filepath, 'wb'))


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
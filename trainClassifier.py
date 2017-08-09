#! python

import os
import time
import pickle
from sys import argv, exit

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier, Perceptron, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report as clsr
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split as tts
from sklearn.feature_extraction import DictVectorizer

if __name__ == "__main__":
    # Verify argument
    if len(argv) < 2:
        print("""requires features.pickle file as argument""")
        exit()

    # Set IO paths
    DIR         = os.getcwd()
    INPATH = DIR + "/features/"+argv[1]+".features.pickle"

    # List of classifiers to be used
    CLASSIFIERS = [SGDClassifier, Perceptron, LogisticRegression, LinearSVC]
    
    # Load preposition instances: features + labels
    with open(INPATH,'rb') as f:
        samples = pickle.load(f)

    # Prepare data
    print('Loading data')
    y = [instance['label'] for instance in samples]
    for instance in samples:
        del instance['label']
    X = samples

    # Label encode the targets
    print('Encoding labels')
    labels = LabelEncoder()
    y = labels.fit_transform(y)

    # Split into train vs test
    print('Train test splitting')
    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.1)

    # Build and eval models using different classifiers
    for classifier in CLASSIFIERS:
        # Build on train
        print('Building...', classifier)
        model = Pipeline([
            ('vectorizer', DictVectorizer()),
            ('normalizer', TfidfTransformer()),
            ('classifier', classifier()),
        ])
        model.fit(X_train,y_train)
        
        # Eval on test
        print('Testing...')
        y_pred = model.predict(X_test)
        print(clsr(y_test, y_pred, target_names=labels.classes_))

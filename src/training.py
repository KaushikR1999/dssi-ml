"""
This module automates model training.
"""

import logging

from sklearn.datasets import load_iris
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from src import model_registry
from src import evaluation
from src.config import appconfig

logging.basicConfig(level=logging.INFO)

features = appconfig['Model']['features'].split(',')
numerical_features = appconfig['Model']['numerical_features'].split(',')
label = appconfig['Model']['label']

def run():
    """
    Main script to perform model training.
        Parameters:
        Returns:
            None
    """
    logging.info('Loading the Iris dataset from scikit-learn...')
    iris = load_iris(as_frame=True)
    df = iris.frame
    
    numerical_transformer = MinMaxScaler()
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
        ]
    )
    
    # Train-Test Split
    logging.info('Start Train-Test Split...')
    X_train, X_test, y_train, y_test = train_test_split(df[features], \
                                                        df[label], \
                                                        test_size=appconfig.getfloat('Model','test_size'), \
                                                        random_state=0,
                                                        stratify=df[label])
    
    # Train Classifier
    logging.info('Start Training...')
    random_forest = RandomForestClassifier(n_estimators=appconfig.getint('Hyperparameters','rf_n_estimators'),
                                           max_depth=appconfig.getint('Hyperparameters','rf_max_depth'), 
                                           class_weight = appconfig.get('Hyperparameters','rf_class_weight'),
                                           n_jobs=appconfig.getint('Hyperparameters','rf_n_jobs'),
                                           random_state=0)
    
    clf = Pipeline(steps=[("preprocessor", preprocessor),\
                          ("binary_classifier", random_forest)
                         ])
    clf.fit(X_train, y_train)
    
    # Evaluate and Deploy
    if evaluation.run(y_test, clf.predict(X_test)):
        logging.info('Persisting model...')
        mdl_meta = { 'name': appconfig['Model']['name'], 'metrics': evaluation.get_eval_metrics(y_test, clf.predict(X_test)) }
        model_registry.register(clf, features, mdl_meta)
    
    logging.info('Training completed.')
    return None

if __name__ == "__main__":
    run()

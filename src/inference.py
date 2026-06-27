"""
This module encapsulates model inference.
"""

import pandas as pd
from sklearn.datasets import load_iris
from src.model_registry import retrieve
from src.config import appconfig

def get_prediction(**kwargs):
    """
    Get prediction for given data.
        Parameters:
            kwargs: Keyworded argument list containing the data for prediction
        Returns:
            Predicted class in str
    """
    clf, features = retrieve(appconfig['Model']['name'])
    pred_df = pd.DataFrame([kwargs])
    pred = clf.predict(pred_df[features])
    return load_iris().target_names[pred[0]]

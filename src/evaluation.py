import logging
from sklearn.metrics import accuracy_score
from src.config import appconfig
from src import model_registry

logging.basicConfig(level=logging.INFO)

accuracy_min = float(appconfig['Evaluation']['accuracy'])
_model_name = appconfig['Model']['name']

def get_eval_metrics(y_true, y_pred):
    """Return metrics suitable for the three-class Iris problem."""
    return {'accuracy': round(accuracy_score(y_true, y_pred), 2)}

def run(y_true, y_pred):
    """Evaluate model accuracy against the configured and registered model.
        Parameters:
            y_true (array-like): True labels
            y_pred (array-like): Predicted labels
        Returns:
            bool: True if evaluation passes, False otherwise
    """
    logging.info('Evaluating model...')
    accuracy = accuracy_score(y_true, y_pred)

    if accuracy < accuracy_min:
        logging.warning(
            f"Model evaluation failed config threshold: "
            f"accuracy={accuracy:.2f} (min {accuracy_min:.2f})"
        )
        return False

    current_metadata = model_registry.get_metadata(_model_name)
    if current_metadata is not None:
        current_metrics = current_metadata['metrics']
        if accuracy < current_metrics['accuracy']:
            logging.warning(
                f"Model evaluation failed vs current model v{current_metadata['version']}: "
                f"accuracy={accuracy:.2f} vs {current_metrics['accuracy']:.2f}"
            )
            return False

    logging.info('Model evaluation passed.')
    return True

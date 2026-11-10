import os
import sys
import dill
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, models, params):
    try:
        report = {}
        tuned_models = {}

        for model_name, model in models.items():
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=params[model_name],
                scoring="r2",
                cv=3,
                n_jobs=1,
                refit=True,
            )
            grid_search.fit(X_train, y_train)
            report[model_name] = grid_search.best_score_
            tuned_models[model_name] = grid_search.best_estimator_
            logging.info(
                f"Tuned {model_name}: CV R2={grid_search.best_score_:.4f}, "
                f"parameters={grid_search.best_params_}"
            )

        return report, tuned_models
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
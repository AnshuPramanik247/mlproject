import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object

@dataclass
class ModelTrainerconfig:
    trained_model_file_path=os.path.join('artifacts','models.pkl')


class ModelTrainer:
    def __init__(self):
        self.modelTrainerconfig = ModelTrainerconfig()

    def initiate_model_training(self,train_array,test_array,preprocessor_path):
        try:
            logging.info('split training and test input data')
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )


            models={
                'Random Forest' : RandomForestRegressor(),
                'Decision Tree' : DecisionTreeRegressor(),
                'Gradient Boosting' : GradientBoostingRegressor(),
                'Linear Regression' : LinearRegression(),
                'k_neighbours Regressor' : KNeighborsRegressor(),
                'XGB Regressor' : XGBRegressor(),
                'Cat boost' : CatBoostRegressor(),
                'Ada Boost' : AdaBoostRegressor()
            }

            param_grids = {
                'Random Forest': {
                    'n_estimators': [100, 200],
                    'max_depth': [None, 10],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                },
                'Decision Tree': {
                    'max_depth': [None, 5, 10, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                },
                'Gradient Boosting': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.03, 0.1],
                    'max_depth': [2, 3],
                    'min_samples_split': [2, 5],
                },
                'Linear Regression': {
                    'fit_intercept': [True, False],
                },
                'k_neighbours Regressor': {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'p': [1, 2],
                },
                'XGB Regressor': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5],
                    'learning_rate': [0.03, 0.1],
                    'subsample': [0.8, 1.0],
                },
                'Cat boost': {
                    'iterations': [300, 500],
                    'depth': [4, 6],
                    'learning_rate': [0.03, 0.1],
                    'verbose': [False],
                },
                'Ada Boost': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.03, 0.1, 1.0],
                },
            }

            cv_scores, tuned_models = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                models=models,
                params=param_grids,
            )
            best_model_name = max(cv_scores, key=cv_scores.get)
            best_model_score = cv_scores[best_model_name]
            best_model = tuned_models[best_model_name]

            if best_model_score<0.6:
                raise CustomException(
                    f"No model achieved the minimum cross-validation R2 score of 0.6 "
                    f"(best: {best_model_name}, score: {best_model_score:.4f})"
                )

            test_score = r2_score(y_test, best_model.predict(X_test))
            logging.info(
                f"Best model: {best_model_name}, CV R2={best_model_score:.4f}, "
                f"test R2={test_score:.4f}"
            )
            save_object(
                file_path=self.modelTrainerconfig.trained_model_file_path,
                obj=best_model
            )

            return test_score
        except Exception as e:
         raise CustomException(e,sys)
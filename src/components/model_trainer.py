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
from src.utils import save_object,evaluate_models

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


            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
            )

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException('NO best model found')
            logging.info(f"Best model found: {best_model_name} with score {best_model_score}")

            save_object(
                file_path=self.modelTrainerconfig.trained_model_file_path,
                obj=best_model
            )

            predict=best_model.predict(X_test)
            r2_square= r2_score(y_test,predict)
            return r2_square
        except Exception as e:
         raise CustomException(e,sys)
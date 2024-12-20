import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgbm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

class KNNModel:
    def __init__(self):
        # Hyperparameters definition
        self.params = {
            "n_neighbors": [3, 5, 7],
            "weights": ["uniform", "distance"],  # Distance-based weighting
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            "metric": ["euclidean", "manhattan"]
        }

    def train(self, x_train, y_train, cv=5, scoring="accuracy"):
        # Define the model
        model = KNeighborsClassifier()

        # Hyperparameter optimization
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=self.params,
            scoring=scoring,
            cv=cv,
            n_jobs=-1
        )
        grid_search.fit(x_train, y_train)

        # Get best model
        best_model = grid_search.best_estimator_
        print("Best KNN Parameters:", grid_search.best_params_)
        
        # Return best model and grid search results for further analysis
        return best_model, grid_search.cv_results_

    def get_best_params(self):
        """Returns the best hyperparameters."""
        if self.best_model is None:
            raise ValueError("The model is not trained yet. Call train() first.")
        return self.best_model.get_params()

    def get_cv_results(self):
        """Returns a DataFrame with the cross-validation results."""
        if self.cv_results_ is None:
            raise ValueError("The model is not trained yet. Call train() first.")
        return self.cv_results_


# Random Forest Class
class RandomForestModel:
    def __init__(self):
        # Hyperparameters definition
        self.params = {
            "n_estimators": [10, 50, 100],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"]
        }

    def train(self, x_train, y_train):
        # Define the model
        model = RandomForestClassifier(class_weight="balanced", random_state=42)
        
        # Hyperparameter optimization
        grid_search = GridSearchCV(estimator=model,
                                   param_grid=self.params,
                                   scoring="accuracy",
                                   cv=5,
                                   n_jobs=-1)
        grid_search.fit(x_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        print("Best Random Forest Parameters:", grid_search.best_params_)
        
        return best_model, grid_search.cv_results_

    def get_best_params(self):
        """Returns the best hyperparameters."""
        if self.best_model is None:
            raise ValueError("The model is not trained yet. Call train() first.")
        return self.best_model.get_params()

    def get_cv_results(self):
        """Returns a DataFrame with the cross-validation results."""
        if self.cv_results_ is None:
            raise ValueError("The model is not trained yet. Call train() first.")
        return self.cv_results_
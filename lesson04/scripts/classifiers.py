import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgbm

# KNN Class
class KNNModel:
    def __init__(self):
        # Hyperparameters definition
        self.params = {
            "n_neighbors": [5, 7],
            "weights": ["uniform", "distance"],
            "algorithm": ["auto", "ball_tree", "kd_tree"],
            "metric": ["euclidean", "manhattan"]
        }

    def train(self, x_train, y_train):
        # Define the model
        model = KNeighborsClassifier()
        
        # Hyperparameter optimization
        grid_search = GridSearchCV(estimator=model,
                                   param_grid=self.params,
                                   scoring="accuracy",
                                   cv=5,
                                   n_jobs=-1)
        grid_search.fit(x_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        print("Best KNN Parameters:", grid_search.best_params_)
        
        return best_model


# SVM Class
class SVMModel:
    def __init__(self):
        # Hyperparameters definition
        self.params = {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf", "poly"],
            "gamma": ["scale", "auto"],
            "degree": [2, 3],
            "class_weight": ["balanced"]
        }

    def train(self, x_train, y_train):
        # Define the model
        model = SVC()
        
        # Hyperparameter optimization 
        grid_search = GridSearchCV(estimator=model,
                                   param_grid=self.params,
                                   scoring="accuracy",
                                   cv=5,
                                   n_jobs=-1)
        grid_search.fit(x_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        print("Best SVM Parameters:", grid_search.best_params_)
        
        return best_model


# Random Forest Class
class RandomForestModel:
    def __init__(self):
        # Hyperparameters definition 
        self.params = {
            "n_estimators": [10, 20, 40],
            "max_depth": [10, 20, 30],
            "min_samples_split": [5, 10, 20],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"]
        }

    def train(self, x_train, y_train):
        # Define the model
        model = RandomForestClassifier()
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
        
        return best_model


# XGBoost Class
class XGBoostModel:
    def __init__(self):
        # Hyperparameters definition
        self.params = {            
            "n_estimators": [20, 50, 100],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [8, 16, 32],
            "subsample": [0.7, 0.8],
            "colsample_bytree": [0.8, 0.9]            
        }

    def train(self, x_train, y_train):
        # Define the model
        model = xgb.XGBClassifier()
        
        # Hyperparameter optimization 
        grid_search = GridSearchCV(estimator=model,
                                   param_grid=self.params,
                                   scoring="accuracy",
                                   cv=5,
                                   n_jobs=-1)
        grid_search.fit(x_train, y_train)
        
        # Get best model 
        best_model = grid_search.best_estimator_
        print("Best XGBoost Parameters:", grid_search.best_params_)
        
        return best_model


# LightGBM Class
class LightGBMModel:
    def __init__(self):
        # Hyperparameters definition 
        self.params = {
            "num_leaves": [31, 50, 100],
            "max_depth": [10, 20, 30],
            "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators": [50, 100, 200],
            "boosting_type": ["gbdt", "dart"]
        }

    def train(self, x_train, y_train):
        # Define the model
        model = lgbm.LGBMClassifier()
        # Hyperparameter optimization 
        grid_search = GridSearchCV(estimator=model,
                                   param_grid=self.params,
                                   scoring="accuracy",
                                   cv=5,
                                   n_jobs=-1)
        grid_search.fit(x_train, y_train)
        
        # Get best model 
        best_model = grid_search.best_estimator_
        print("Best LightGBM Parameters:", grid_search.best_params_)
        
        return best_model

import xgboost as xgb
import matplotlib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split

matplotlib.use('Qt5Agg')


def xgboost_train(X_train, y_train):
    # XGBoost with validation set
    X_train_xgb, X_val, y_train_xgb, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Hyperparameter tuning for XGBoost with validation set
    param_grid_xgb = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [50, 100, 200],
    }

    grid_xgb = GridSearchCV(
        xgb.XGBClassifier(scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train), random_state=42),
        param_grid_xgb, cv=10, scoring='roc_auc')
    grid_xgb.fit(X_train_xgb, y_train_xgb, eval_metric='logloss', eval_set=[(X_val, y_val)], early_stopping_rounds=10,
                 verbose=False)

    # Best hyperparameters
    best_params_xgb = grid_xgb.best_params_

    # XGBoost with best hyperparameters and validation set
    best_xgb = xgb.XGBClassifier(**best_params_xgb, random_state=42)
    best_xgb.fit(X_train, y_train, eval_metric='logloss', eval_set=[(X_val, y_val)], early_stopping_rounds=10,
                 verbose=False)

    return best_xgb


def logistic_train(X_train_sc, y_train_sc):
    # Hyperparameter tuning for Logistic Regression
    param_grid_logreg = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'liblinear'],
        # 'verbose': 1
    }

    grid_logreg = GridSearchCV(LogisticRegression(random_state=42), param_grid_logreg, cv=10, scoring='roc_auc')
    grid_logreg.fit(X_train_sc, y_train_sc)

    # Logistic Regression with best hyperparameters
    best_logreg = grid_logreg.best_estimator_

    return best_logreg


# get predictions
def model_predict(model, X_train, X_test):
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_test)

    return y_pred_train, y_pred_val
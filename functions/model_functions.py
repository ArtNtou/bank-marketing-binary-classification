import xgboost as xgb
import matplotlib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split

matplotlib.use('Qt5Agg')


def xgboost_train(X_train, y_train):
    """
    Function that train finds the best possible xgboost model and trains it

    Parameters:
        - X_train_sc (dataframe): features of training set
        - y_train_sc (pandas Series): label of classification of training set

    Return:
        - best_xgb :

    """

    # XGBoost with validation set
    X_train_xgb, X_val, y_train_xgb, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Hyperparameter tuning for XGBoost with validation set
    param_grid_xgb = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.5],
        'n_estimators': [100, 200, 400],
    }

    # Create XGBClassifier object with scale_pos_weight and early_stopping_rounds parameters
    xgb_clf = xgb.XGBClassifier(scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),
                                random_state=42, eval_metric='logloss',
                                early_stopping_rounds=10)

    # Create GridSearchCV object
    grid_xgb = GridSearchCV(xgb_clf, param_grid_xgb, cv=10, scoring='roc_auc', verbose=False)

    grid_xgb.fit(X_train_xgb, y_train_xgb, eval_set=[(X_val, y_val)], verbose=False)

    # Best hyperparameters
    best_params_xgb = grid_xgb.best_params_

    # XGBoost with best hyperparameters and validation set
    best_xgb = xgb.XGBClassifier(**best_params_xgb, scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),
                                 random_state=42)
    best_xgb.fit(X_train, y_train, verbose=False)

    return best_xgb


def logistic_train(X_train_sc, y_train_sc):
    """
    Function that train finds the best possible logistic regression and trains it

    Parameters:
        - X_train_sc (dataframe): features of training set
        - y_train_sc (pandas Series): label of classification of training set

    Return:
        - best_logreg (LogisticRegression): a logistic regression model

    """

    # Hyperparameter tuning for Logistic Regression
    param_grid_logreg = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'liblinear'],
        'multi_class': ['auto', "ovr"]
        # 'verbose': 1
    }

    grid_logreg = GridSearchCV(LogisticRegression(random_state=42), param_grid_logreg, cv=10, scoring='roc_auc')
    grid_logreg.fit(X_train_sc, y_train_sc)

    # Logistic Regression with best hyperparameters
    best_logreg = grid_logreg.best_estimator_

    return best_logreg


# get predictions
def model_predict(model, X_train, X_test):
    """
    Function make predictions based on features

    Parameters:
        - X_train (dataframe): features of training set
        - X_test (dataframe): features of test set

    Return:
        - best_logreg (LogisticRegression): a logistic regression model

    """

    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_test)

    return y_pred_train, y_pred_val

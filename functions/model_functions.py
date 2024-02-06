import xgboost as xgb
import matplotlib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split

matplotlib.use('Qt5Agg')


def xgboost_train(X_train, y_train):
    """
    Trains an XGBoost model with hyperparameter tuning and validation set.

    :param X_train: Features of the training set.
    :type X_train: pandas.DataFrame
    :param y_train: Labels of the training set.
    :type y_train: pandas.Series

    :return: The best trained XGBoost model.
    :rtype: xgb.XGBClassifier
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
    Trains a logistic regression model with hyperparameter tuning.

    :param X_train_sc: Features of the training set.
    :type X_train_sc: pandas.DataFrame
    :param y_train_sc: Labels of the training set.
    :type y_train_sc: pandas.Series

    :return: The best trained logistic regression model.
    :rtype: sklearn.linear_model.LogisticRegression
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
    Makes predictions based on features.

    :param model: Trained model.
    :type model: object
    :param X_train: Features of the training set.
    :type X_train: pandas.DataFrame
    :param X_test: Features of the test set.
    :type X_test: pandas.DataFrame

    :return: Predictions for the training and test sets.
    :rtype: tuple
    """

    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_test)

    return y_pred_train, y_pred_val

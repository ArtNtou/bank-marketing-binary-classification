import pandas as pd
from sklearn.preprocessing import StandardScaler


def train_transformation(train_data):
    """
        Preprocess train set

        Parameters:
          - train_data (dataframe): train dataset with features and label

        Return:
          - X_test (dataframe): contains features of test dateset
          - y_test (pandas Series): binary label of test dataset
          - transformation (dictionary): contains information about the transformations
              - month_dict (dictionary): information about months from string to float
              - binary_transformation (dictionary): strings to integer
              - binary_columns (list): binary columns
              - education_transformation (dictionary): strings about education to integer
              - categorical columns (list): categorical columns that needed transformations

        """

    # fill na values in text columns with unknown
    train_data = train_data.fillna("unknown")

    # train_data[train_data['y'] == 'yes']['poutcome'].value_counts()/len(train_data[train_data['y'] == 'yes'])
    # while poutcome has 82% nonexistent values the difference in success, when y equals yes and y equals no,
    # make us keep this column and not dropping it

    # remove outliers
    numerical_features = ['age', 'campaign', 'duration']
    for cols in numerical_features:
        Q1 = train_data[cols].quantile(0.25)
        Q3 = train_data[cols].quantile(0.75)
        IQR = Q3 - Q1

        filter_outlier = (train_data[cols] >= Q1 - 1.5 * IQR) & (train_data[cols] <= Q3 + 1.5 * IQR)

    # print(filter_outlier.value_counts() / len(filter_outlier))
    # print(filter_test.value_counts() / len(filter_test))
    # 7% percent of data belong to above outliers in both train and test set

    # remove outliers
    train_data = train_data[filter_outlier]

    # transform values to float
    month_dict = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10,
                  'nov': 11, 'dec': 12}
    train_data['month'] = train_data['month'].map(month_dict)

    # housing default and loan preprocessing
    binary_transformation = {'yes': 1, 'no': 0, 'unknown': -1}
    binary_columns = ['housing', 'default', 'loan', 'y']
    for i in binary_columns:
        train_data[i] = train_data[i].map(binary_transformation)

    # transform strings to integer with order
    education_transformation = {"primary": 1, "secondary": 2, "tertiary": 3, "unknown": -1}
    train_data['education'] = train_data['education'].map(education_transformation)

    # categorical data to binary columns
    categorical_columns = ['job', 'marital', 'contact', 'poutcome']

    train_data_modified = pd.get_dummies(train_data, columns=categorical_columns, prefix=categorical_columns,
                                         drop_first=True)

    train_data_modified = train_data_modified.replace({True: 1, False: 0})

    transformations = {"month_dict": month_dict, "binary_transformation": binary_transformation,
                       "binary_columns": binary_columns, "education_transformation": education_transformation,
                       "categorical_columns": categorical_columns}

    return train_data_modified, transformations


# preprocessing on test data
def test_transformation(X_test, y_test, scaling, transformations):
    """
    Does the same transformation of training for the test dataset

    Parameters:
      - X_test (dataframe): contains features of test dateset
      - y_test (pandas Series): binary label of test dataset
      - transformations (dictionary): contains information about the transformations
          - scaler_dict (dictionary): contains scaler model and the columns needed to be scaled
          - month_dict (dictionary): information about months from string to float
          - binary_transformation (dictionary): strings to integer
          - binary_columns (list): binary columns
          - education_transformation (dictionary): strings about education to integer
          - categorical columns (list): categorical columns that needed transformations

    Return:
      - X_test (DataFrame): transformed features ready to be used for prediction
      - y_test (pandas Series): label

    """

    # merge them for analysis
    test_data = pd.merge(X_test, y_test, left_index=True, right_index=True)

    # checkfor uknown, non-existent etc. for all columns in order to better understand data
    # fill na values in text columns with unknown
    test_data = test_data.fillna("unknown")

    test_data['month'] = test_data['month'].map(transformations['month_dict'])

    for i in transformations['binary_columns']:
        test_data[i] = test_data[i].map(transformations['binary_transformation'])

    test_data['education'] = test_data['education'].map(transformations['education_transformation'])

    test_final = pd.get_dummies(test_data, columns=transformations['categorical_columns'],
                            prefix=transformations['categorical_columns'], drop_first=True)
    
    # for logistic regression is True and xgboost is False
    if scaling:
        # find scaler and scaling columns
        scaling_col = transformations['scaler_dict']['scaling_columns']
        scaler = transformations['scaler_dict']['scaler_model']
    
        # scale data using information from training
        test_final[scaling_col] = scaler.transform(test_final[scaling_col])

    test_final = test_final.replace({True: 1, False: 0})

    X_test = test_final.drop(columns='y')
    y_test = test_final['y']

    return X_test, y_test

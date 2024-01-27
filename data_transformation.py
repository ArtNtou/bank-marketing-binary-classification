import pandas as pd
from sklearn.preprocessing import StandardScaler

# preprocessing on test data
def test_transformation(X_test, y_test, scaling, month_dict, dictionary, binary_columns, dictionary_edu,
                        categorical_columns):
    # merge them for analysis
    dtr = pd.merge(X_test, y_test, left_index=True, right_index=True)

    # checkfor uknown, non-existent etc. for all columns in order to better understand data
    # fill na values in text columns with unknown
    dtr = dtr.fillna("unknown")

    dtr['month'] = dtr['month'].map(month_dict)

    for i in binary_columns:
        dtr[i] = dtr[i].map(dictionary)

    dtr['education'] = dtr['education'].map(dictionary_edu)

    dtsenc = pd.get_dummies(dtr, columns=categorical_columns, prefix=categorical_columns, drop_first=True)

    scaling_col = ['age', 'balance', 'day_of_week', 'month', 'duration', 'campaign', 'pdays', 'previous']

    if scaling:
        dtsenc = dtsenc.copy()
        scaler = StandardScaler()
        dtsenc[scaling_col] = scaler.fit_transform(dtsenc[scaling_col])

    dtsenc = dtsenc.replace({True: 1, False: 0})

    X_test = dtsenc.drop(columns='y')
    y_test = dtsenc['y']

    return X_test, y_test
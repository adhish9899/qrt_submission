'''
QRT 2021 data challenge. 
author: Adhish Aggarwal
'''

import pandas as pd
import numpy as np
import datetime as dt

from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression

from utils import weighted_accuracy

from tqdm import tqdm
import copy

# NUMBER OF CORRELATED FEATURES TO USE. 
N_FEAT = 30

# READING TRAINING AND TESTING DATA
X_train = pd.read_csv("X_train.csv").set_index("ID")
Y_train = pd.read_csv("y_train.csv").set_index("ID")
X_test = pd.read_csv("X_test.csv").set_index("ID")

# FILLING NAN's IN TRAINING DATA (X_train) USING LINEAR REGRESSION
corr_df = X_train.drop(["ID_DAY", "ID_TARGET"], axis=1).corr()
nan_dict = {}
for col in corr_df.columns:

    corr_factors = corr_df.loc[col].drop(col).sort_values()
    cols = corr_factors.iloc[-N_FEAT:].index.tolist()

    data_x = X_train[cols]
    x_test = X_train[X_train[col].isnull()].drop([col], axis=1)

    x_train = X_train[~X_train[col].isnull()].drop([col], axis=1)
    y_train = X_train[~X_train[col].isnull()][col]

    x_train = x_train.fillna(0)
    x_test = x_test.fillna(0)

    model = LinearRegression()
    model.fit(x_train, y_train)

    print(col, model.score(x_train, y_train))

    pred_values = model.predict(x_test)
    nan_dict[col] = (x_test.index.tolist(), pred_values)

## REPLACING NAN's WITH PREDICTED VALUES.
for key, items in nan_dict.items():
    X_train.loc[items[0], key] = items[1]

### SPLITTING INTO TRAIN AND VALID
train_valid_dict = {}
for id_ in X_train.ID_TARGET.unique():
    train_valid_dict[id_] = {}

    id_df_x = X_train[X_train.ID_TARGET == id_]
    id_df_y = Y_train.loc[id_df_x.index]

    id_df_x = id_df_x.drop(["ID_DAY", "ID_TARGET"], axis=1)

    ### SPLITTING USING "train_test_split"
    X_train_id, X_valid_id, y_train_id, y_valid_id = train_test_split(id_df_x, id_df_y, test_size=0.2, random_state=2)

    ## TOP "N_FEAT" CORRELATED FEATURES WITH "ID_TARGET" IN TRAINING DATA
    corr_df = pd.concat([X_train_id, y_train_id], axis=1).corr().loc["RET_TARGET"].drop(["RET_TARGET"])
    features = corr_df.abs().sort_values().iloc[-N_FEAT:].index.tolist()

    X_train_id = X_train_id[features]
    X_valid_id = X_valid_id[features]

    train_valid_dict[id_]["train"] = (X_train_id, y_train_id)
    train_valid_dict[id_]["valid"] = (X_valid_id, y_valid_id)
    train_valid_dict[id_]["features"] = features

## USING PLS (PARTIAL LEAST SQUARE) REGRESSION TO PREDICT. 
max_ = {}
for key in tqdm(train_valid_dict):
    X_train_id, Y_train_id = train_valid_dict[key]["train"]
    X_valid_id, Y_valid_id = train_valid_dict[key]["valid"]
    max_[key] = {"valid": [0, 0, 0, None]}

    ## USING 2-3 COMPONENTS TO REDUCE OVERFITTING. 
    for i in range(2,4):
        pls2 = PLSRegression(n_components=i)
        pls2.fit(X_train_id, Y_train_id)

        train_pred_array = pls2.predict(X_train_id)
        train_pred_array = train_pred_array.reshape(-1,)

        valid_pred_array = pls2.predict(X_valid_id)
        valid_pred_array = valid_pred_array.reshape(-1,)


        train_actual = Y_train_id.values.reshape(-1,)
        valid_actual = Y_valid_id.values.reshape(-1,)

        train_acc = round(weighted_accuracy(train_actual, train_pred_array),3)
        valid_acc = round(weighted_accuracy(valid_actual, valid_pred_array),3)

        if valid_acc > max_[key]["valid"][0]:
            max_[key]["valid"][0] = valid_acc
            max_[key]["valid"][1] = i
            max_[key]["valid"][2] = train_acc
            max_[key]["valid"][3] = copy.deepcopy(pls2) # SAVING THE MODEL WITH BEST VALID ACCURACY



# FINAL SUBMISSION. PREDICTING ON TEST DATA 
submission_dict = {"ID": [], "RET_TARGET": []}
for i in tqdm(range(len(X_test))):
    id_target = int(X_test.iloc[i]["ID_TARGET"])
    test_row = X_test.iloc[i].fillna(0).drop(["ID_DAY", "ID_TARGET"])
    test_row = test_row.loc[train_valid_dict[id_target]["features"]]

    predicton = max_[id_target]["valid"][3].predict(test_row.values.reshape(1,-1))[0,0]
    id_ = X_test.index[i]

    submission_dict["ID"].append(id_)
    submission_dict["RET_TARGET"].append(np.sign(predicton))

##
df_sub = pd.DataFrame(submission_dict).set_index("ID")
df_sub.to_csv("submission.csv")

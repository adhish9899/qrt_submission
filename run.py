import pandas as pd
import numpy as np
import datetime as dt

from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression, PLSCanonical
from sklearn.linear_model import LinearRegression, Ridge

from tqdm import tqdm
import copy

X_train = pd.read_csv("X_train_itDkypA.csv").set_index("ID")
Y_train = pd.read_csv("y_train_3LeeT2g.csv").set_index("ID")
X_test = pd.read_csv("X_test_Beg4ey3.csv").set_index("ID")

corr_df = X_train.drop(["ID_DAY", "ID_TARGET"], axis=1).corr()

nan_dict = {}
for col in corr_df.columns:

    corr_factors = corr_df.loc[col].drop(col).sort_values()
    cols = corr_factors.iloc[-30:].index.tolist()

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

##
for key, items in nan_dict.items():
    X_train.loc[items[0], key] = items[1]

### SPLITTING INTO TRAIN AND VALID
train_valid_dict = {}

for id_ in X_train.ID_TARGET.unique():
    train_valid_dict[id_] = {}

    id_df_x = X_train[X_train.ID_TARGET == id_]
    id_df_y = Y_train.loc[id_df_x.index]

    id_df_x = id_df_x.drop(["ID_DAY", "ID_TARGET"], axis=1)

    ### SCALING
    X_train_id, X_valid_id, y_train_id, y_valid_id = train_test_split(id_df_x, id_df_y, test_size=0.2, random_state=2)

    ## TOP 30 FEATURES CORR WISE
    corr_df = pd.concat([X_train_id, y_train_id], axis=1).corr().loc["RET_TARGET"].drop(["RET_TARGET"])
    features = corr_df.abs().sort_values().iloc[-30:].index.tolist()
#     features = corr_df.abs().sort_values().iloc[:30].index.tolist()

    X_train_id = X_train_id[features]
    X_valid_id = X_valid_id[features]

    train_valid_dict[id_]["train"] = (X_train_id, y_train_id)
    train_valid_dict[id_]["valid"] = (X_valid_id, y_valid_id)
    train_valid_dict[id_]["features"] = features

print(list(train_valid_dict.keys())[:5])

def weighted_accuracy_normal(y_test, y_pred):
    y_abs = np.abs(y_test)
    norm = y_abs.sum()
    score = ((np.sign(y_pred) == np.sign(y_test)) * y_abs).sum() / norm
    return score

max_ = {}
# import ipdb; ipdb.set_trace()
for key in tqdm(train_valid_dict):
#     print(key)
    X_train_id, Y_train_id = train_valid_dict[key]["train"]
    X_valid_id, Y_valid_id = train_valid_dict[key]["valid"]
    max_[key] = {"valid": [0, 0, 0, None]}

    for i in range(2,4): #X_train_id.shape[1]
#         print(i)

        pls2 = PLSRegression(n_components=i)
        pls2.fit(X_train_id, Y_train_id)

        train_pred_array = pls2.predict(X_train_id)
        train_pred_array = train_pred_array.reshape(-1,)

        valid_pred_array = pls2.predict(X_valid_id)
        valid_pred_array = valid_pred_array.reshape(-1,)


        train_actual = Y_train_id.values.reshape(-1,)
        valid_actual = Y_valid_id.values.reshape(-1,)

        train_acc = round(weighted_accuracy_normal(train_actual, train_pred_array),3)
        valid_acc = round(weighted_accuracy_normal(valid_actual, valid_pred_array),3)

        if valid_acc > max_[key]["valid"][0]:
            max_[key]["valid"][0] = valid_acc
            max_[key]["valid"][1] = i
            max_[key]["valid"][2] = train_acc
            max_[key]["valid"][3] = copy.deepcopy(pls2)


print("VALID", np.mean([max_[x]["valid"][0] for x in max_.keys()]))
print("TRAIN", np.mean([max_[x]["valid"][2] for x in max_.keys()]))


####
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
df_sub.head()

df_sub.to_csv("submission.csv")
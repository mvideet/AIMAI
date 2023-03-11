import pandas as pd
import openai
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import xgboost as xgb
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle
base_name = "combined_kept_reason"

openai.api_key = "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"


# def filter(name):
#     checkSet2 = [
#         "MRI BRAIN WO/W CONTRAST",
#         "MRI COMPLETE SPINE WO/W CONTRAST",
#         "CT SOFT TISSUE NECK WITH CONTRAST",
#         "MRI PITUITARY WO/W CONTRAST",
#         "MR ANGIO BRAIN WITHOUT CONTRAST",
#         "CT MAXILLOFACIAL WITHOUT CONTRAST",
#         "MRI BRAIN AND ORBITS WO/W CONTRAST",
#         "CT HEAD WITHOUT CONTRAST",
#     ]

#     df = pd.read_csv(name + ".csv")
#     for k in range(len(df["PROC_NAME"])):
#         if checkSet2.count(df.loc[k, "PROC_NAME"]) == 0:
#             df = df.drop(k, axis=0)

#     df.to_csv(name + "_filtered.csv", index=False)


# def embed(name):
#     df = pd.read_csv(name + ".csv")

#     def get_embedding(text, model="text-embedding-ada-002"):
#         # print(text)
#         if isinstance(text, float):
#             return "NaN"
#         else:
#             text = text.replace("\n", " ")
#             return openai.Embedding.create(input=[text], model=model)["data"][0][
#                 "embedding"
#             ]

#     df["embedded_symptoms"] = df.Symptoms.apply(
#         lambda x: get_embedding(x, model="text-embedding-ada-002")
#     )
#     df.to_csv(name + "_embedded.csv", index=False)


# def clean(name):
#     df = pd.read_csv(name + ".csv")
#     for k in range(len(df["embedded_symptoms"])):
#         if isinstance(df.loc[k, "embedded_symptoms"], float):
#             df = df.drop(k, axis=0)
#     # df = df.drop(df.columns[0], axis=1)
#     df.to_csv(name + "_cleaned.csv", index=False)


# def predict(name):
#     from sklearn.model_selection import train_test_split

#     df = pd.read_csv(name + ".csv")

#     df["embedded_symptoms"] = df.embedded_symptoms.apply(eval).apply(np.array)

#     encoder = LabelEncoder()

#     # Fit the encoder to the data
#     df["PROC_NAME"] = encoder.fit_transform(df["PROC_NAME"])

#     X_train, X_test, y_train, y_test = train_test_split(
#         list(df.embedded_symptoms.values),
#         df.PROC_NAME,
#         test_size=0.2,
#         random_state=42,
#     )

#     print("xgb")

#     from numpy import mean
#     from numpy import std
#     from sklearn.datasets import make_classification
#     from sklearn.model_selection import cross_val_score
#     from sklearn.model_selection import RepeatedStratifiedKFold
#     from xgboost import XGBRFClassifier
#     import xgboost as xgb

#     xgb_clf = xgb.XGBClassifier(objective="multi:softmax", num_class=8)
#     xgb_clf.fit(X_train, y_train, verbose=2)

#     y_pred = xgb_clf.predict(X_test)

#     from sklearn.metrics import accuracy_score

#     accuracy = accuracy_score(y_test, y_pred)
#     print("Accuracy: {:.2f}%".format(accuracy * 100))


def svm(name):
    df = pd.read_csv(name + ".csv")
    print(df.columns)
    print("finished reading")
    df["embedded_symptoms"] = df.embedded_symptoms.apply(eval).apply(np.array)
    print("Finished Embedding")
    encoder = LabelEncoder()

    # Fit the encoder to the data
    df["label"] = encoder.fit_transform(df["label"])
    print("finished transforming")
    X_train, X_test, y_train, y_test = train_test_split(
        list(df.embedded_symptoms.values),
        df.label,
        test_size=0.05,
        random_state=42,
    )
    print("finished splitting")
    x_train = np.array(X_train)
    Y_train = np.array(y_train)
    x_test = np.array(X_test)
    Y_test = np.array(y_test)

    try:
        print(X_test.shape)
    except:
        print(x_test.shape)

    clf = OneVsRestClassifier(SVC(kernel="poly",probability=True))
    clf.fit(x_train, Y_train)
    with open('svm_model_probabilityOVR.pkl', 'wb') as file:
        pickle.dump(clf, file)
    print("finished fitting")

def random_forest(name):
    df = pd.read_csv(name + ".csv")
    print(df.columns)
    print("finished reading")
    df["embedded_symptoms"] = df.embedded_symptoms.apply(eval).apply(np.array)
    print("Finished Embedding")
    encoder = LabelEncoder()

    # Fit the encoder to the data
    df["label"] = encoder.fit_transform(df["label"])
    print("finished transforming")
    X_train, X_test, y_train, y_test = train_test_split(
        list(df.embedded_symptoms.values),
        df.label,
        test_size=0.05,
        random_state=42,
    )
    print("finished splitting")
    x_train = np.array(X_train)
    Y_train = np.array(y_train)
    x_test = np.array(X_test)
    Y_test = np.array(y_test)

    try:
        print(X_test.shape)
    except:
        print(x_test.shape)

    clf = RandomForestClassifier()
    clf.fit(x_train, Y_train)
    print("finished fitting")
    Y_pred = clf.predict(x_test)
    print(accuracy_score(Y_test, Y_pred))
    with open('random_forest_model.pkl', 'wb') as file:
        pickle.dump(clf, file)




print("SVM")
svm("combo11")

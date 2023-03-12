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

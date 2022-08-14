import signatory
import torch
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from plot_metric.functions import BinaryClassification
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.svm import TimeSeriesSVC
from tslearn.preprocessing import TimeSeriesScalerMinMax
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import joblib


def read_alcoholic(subset='1'):
    # S1: S1 obj - a single object shown;
    s1 = 0
    # S12: S2 nomatch - object 2 shown in a non-matching condition (S1 differed from S2)
    s12 = 0
    # S21: S2 match - object 2 shown in a matching condition (S1 was identical to S2),
    s21 = 0
    # initialise numpy arrays to fill with time series
    s1_X_train_unscaled = np.zeros((160, 256, 64))
    s1_y_train = np.zeros(160)
    s21_X_train_unscaled = np.zeros((159, 256, 64))
    s21_y_train = np.zeros(159)
    s12_X_train_unscaled = np.zeros((149, 256, 64))
    s12_y_train = np.zeros(149)

    # assign numerical values to the classes
    classifier = {'a': 1, 'c': 0}

    # run through every file in the train directory and import it, using information from the
    # matching condition column to determine which experiment was being conducted.
    # using this, put the data into the corresponding numpy array
    filenames_list = os.listdir('SMNI_CMI_TRAIN/Train')

    for file_name in tqdm(filenames_list):
        temp_df = pd.read_csv('SMNI_CMI_TRAIN/Train/' + file_name)
        if temp_df["matching condition"][0] == "S1 obj":
            s1_X_train_unscaled[s1] = np.transpose(np.array(temp_df["sensor value"]).reshape([64, 256]))
            s1_y_train[s1] = classifier[temp_df['subject identifier'][0]]
            s1 += 1
        if temp_df["matching condition"][0] == "S2 match":
            s21_X_train_unscaled[s21] = np.transpose(np.array(temp_df["sensor value"]).reshape([64, 256]))
            s21_y_train[s21] = classifier[temp_df['subject identifier'][0]]
            s21 += 1
        if temp_df["matching condition"][0] == "S2 nomatch,":
            s12_X_train_unscaled[s12] = np.transpose(np.array(temp_df["sensor value"]).reshape([64, 256]))
            s12_y_train[s12] = classifier[temp_df['subject identifier'][0]]
            s12 += 1

    # t1: S1 obj - a single object shown;
    t1 = 0
    # t12: S2 nomatch - object 2 shown in a non-matching condition (S1 differed from S2)
    t12 = 0
    # t21: S2 match - object 2 shown in a matching condition (S1 was identical to S2),
    t21 = 0
    t1_X_test_unscaled = np.zeros((160, 256, 64))
    t1_y_test = np.zeros(160)
    t21_X_test_unscaled = np.zeros((160, 256, 64))
    t21_y_test = np.zeros(160)
    t12_X_test_unscaled = np.zeros((160, 256, 64))
    t12_y_test = np.zeros(160)

    # same as above but for test data
    classifier = {'a': 1, 'c': 0}

    # list of filenames in the directory
    filenames_list = os.listdir('SMNI_CMI_TEST')

    for file_name in tqdm(filenames_list):
        if file_name == "Test":
            pass
        else:
            temp_df = pd.read_csv('SMNI_CMI_TEST/' + file_name)
            if temp_df["matching condition"][0] == "S1 obj":
                t1_X_test_unscaled[t1] = np.transpose(np.array(temp_df["sensor value"]).reshape([64, 256]))
                t1_y_test[t1] = classifier[temp_df['subject identifier'][0]]
                t1 += 1
            if temp_df["matching condition"][0] == "S2 match":
                t21_X_test_unscaled[t21] = np.transpose(np.array(temp_df["sensor value"]).reshape([64, 256]))
                t21_y_test[t21] = classifier[temp_df['subject identifier'][0]]
                t21 += 1
            if temp_df["matching condition"][0] == "S2 nomatch,":
                t12_X_test_unscaled[t12] = np.transpose(np.array(temp_df["sensor value"]).reshape([64, 256]))
                t12_y_test[t12] = classifier[temp_df['subject identifier'][0]]
                t12 += 1

    if subset == '1':
        X_train = s1_X_train_unscaled
        y_train = s1_y_train
        X_test = t1_X_test_unscaled
        y_test = t1_y_test
    elif subset == '12':
        X_train = s12_X_train_unscaled
        y_train = s12_y_train
        X_test = t12_X_test_unscaled
        y_test = t12_y_test
    elif subset == '21':
        X_train = s21_X_train_unscaled
        y_train = s21_y_train
        X_test = t21_X_test_unscaled
        y_test = t21_y_test
    else:
        X_train = None
        y_train = None
        X_test = None
        y_test = None

    return X_train, y_train, X_test, y_test


def ml_method_setup(method, X_train, sig_train, y_train, dataset):
    if method == 'ts_knn':
        try:
            clf = joblib.load(open(f'models/{dataset}_ts_knn.pkl', 'rb'))
        except:
            clf = GridSearchCV(
                Pipeline([
                    ('knn', KNeighborsTimeSeriesClassifier())
                ]),
                {'knn__n_neighbors': [3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25], 'knn__weights': ['uniform', 'distance']},
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
                n_jobs=-1
            )
            clf.fit(X_train, y_train)
            joblib.dump(clf.best_estimator_, f'models/{dataset}_ts_knn.pkl')

    elif method == 'ts_svc':
        try:
            clf = joblib.load(open(f'models/{dataset}_ts_svc.pkl', 'rb'))
        except:
            clf = GridSearchCV(
                Pipeline([
                    ('svc', TimeSeriesSVC(random_state=0, probability=True))
                ]),
                {'svc__kernel': ['gak', 'rbf', 'poly'], 'svc__shrinking': [True, False],
                 'svc__C': [0.1, 0.2, 0.5, 1, 2, 5, 10]},
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
                n_jobs=-1
            )
            clf.fit(X_train, y_train)
            joblib.dump(clf.best_estimator_, f'models/{dataset}_ts_svc.pkl')

    elif method == 'lr':
        try:
            clf = joblib.load(open(f'models/{dataset}_lr.pkl', 'rb'))
        except:
            lr = LogisticRegression(random_state=0)
            parameters = {'C': [0.1, 0.2, 0.5, 1, 2, 5, 10],
                          'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
            clf = GridSearchCV(lr, parameters, n_jobs=-1)
            clf.fit(sig_train, y_train)
            joblib.dump(clf.best_estimator_, f'models/{dataset}_lr.pkl')

    elif method == 'svc':
        try:
            clf = joblib.load(open(f'models/{dataset}_svc.pkl', 'rb'))
        except:
            svc = SVC(random_state=0, probability=True)
            parameters = {'kernel': ['rbf', 'poly'], 'shrinking': [True, False],
                          'C': [0.1, 0.2, 0.5, 1, 2, 5, 10]}
            clf = GridSearchCV(svc, parameters, n_jobs=-1)
            clf.fit(sig_train, y_train)
            joblib.dump(clf.best_estimator_, f'models/{dataset}_svc.pkl')

    elif method == 'knn':
        try:
            clf = joblib.load(open(f'models/{dataset}_knn.pkl', 'rb'))
        except:
            knn = KNeighborsClassifier()
            parameters = {'n_neighbors': range(3, 30, 2), 'weights': ['uniform', 'distance']}
            clf = GridSearchCV(knn, parameters, n_jobs=-1)
            clf.fit(sig_train, y_train)
            joblib.dump(clf.best_estimator_, f'models/{dataset}_knn.pkl')

    elif method == 'ada':
        try:
            clf = joblib.load(open(f'models/{dataset}_ada.pkl', 'rb'))
        except:
            ada = AdaBoostClassifier(random_state=0)
            parameters = {'n_estimators': [50, 100, 150], 'learning_rate': [0.1, 0.5, 1, 2]}
            clf = GridSearchCV(ada, parameters, n_jobs=-1)
            clf.fit(sig_train, y_train)
            joblib.dump(clf.best_estimator_, f'models/{dataset}_ada.pkl')

    elif method == 'rf':
        try:
            clf = joblib.load(open(f'models/{dataset}_rf.pkl', 'rb'))
        except:
            rf = RandomForestClassifier(random_state=0)
            parameters = {'min_weight_fraction_leaf': [0.01, 0.1, 0.5],
                          'bootstrap': [True, False],
                          'max_depth': (2, 5, 10),
                          'max_leaf_nodes': (2, 5, 10),
                          'n_estimators': (100, 200, 300)}
            clf = GridSearchCV(rf, parameters, n_jobs=-1)
            clf.fit(sig_train, y_train)
            joblib.dump(clf.best_estimator_, f'models/{dataset}_rf.pkl')

    else:
        clf = None

    return clf


def auto_ml(X_train, y_train, X_test, y_test, method, sig_level, dataset, ts_scale=True, standard_scale=True):
    start = time.time()

    # initialise scalers
    ts_scaler = TimeSeriesScalerMinMax()
    scaler = StandardScaler()

    if ts_scale:
        X_train = ts_scaler.fit_transform(X_train)
        X_test = ts_scaler.fit_transform(X_test)

    X_train_torch = torch.from_numpy(X_train).cuda()
    X_test_torch = torch.from_numpy(X_test).cuda()

    sig_train_unscaled = signatory.signature(X_train_torch, sig_level)
    sig_train_unscaled = sig_train_unscaled.cpu().numpy()
    sig_test_unscaled = signatory.signature(X_test_torch, sig_level)
    sig_test_unscaled = sig_test_unscaled.cpu().numpy()

    if standard_scale:
        sig_train = scaler.fit_transform(sig_train_unscaled)
        sig_test = scaler.fit_transform(sig_test_unscaled)
    else:
        sig_train = sig_train_unscaled
        sig_test = sig_test_unscaled

    clf = ml_method_setup(method, X_train, sig_train, y_train, dataset)

    # fit to data
    y_pred_proba = clf.predict_proba(sig_test)[:, 1]
    y_pred = clf.predict(sig_test)
    cv_score = cross_val_score(clf, sig_train, y_train, scoring='roc_auc', n_jobs=-1)
    bc = BinaryClassification(y_test, y_pred_proba, labels=["Class 0", "Class 1"])

    # Figures
    plt.figure(figsize=(5, 5))
    bc.plot_roc_curve()
    plt.title("Receiver Operating Characteristic Using Logistic Regression")
    plt.show()
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)

    print("accuracy is " + str(accuracy))
    print("auc is " + str(roc_auc))
    print("f1-measure " + str(f1))
    print(cv_score, "mean CV on Train = " + str(cv_score.mean()))

    end = time.time()
    print(end - start)


def main(dataset, method, sig_level, ts_scale=True, standard_scale=True):
    if dataset == 'alcoholic_1':
        X_train, y_train, X_test, y_test = read_alcoholic(subset='1')
    elif dataset == 'alcoholic_12':
        X_train, y_train, X_test, y_test = read_alcoholic(subset='12')
    elif dataset == 'alcoholic_21':
        X_train, y_train, X_test, y_test = read_alcoholic(subset='21')
    else:
        X_train, y_train, X_test, y_test = (None, None, None, None)

    auto_ml(X_train, y_train, X_test, y_test, method, sig_level, dataset,
            ts_scale=ts_scale, standard_scale=standard_scale)

main('alcoholic_1', 'lr', sig_level=2)
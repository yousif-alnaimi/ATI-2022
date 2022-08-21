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
import h5py


def write_alcoholic(subset='1'):
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

    f = h5py.File("data/data.h5", 'a')
    grp = f.create_group(f"alcoholic_{subset}")
    grp.create_dataset("X_train", data=X_train, compression="gzip", compression_opts=7)
    grp.create_dataset("y_train", data=y_train, compression="gzip", compression_opts=7)
    grp.create_dataset("X_test", data=X_test, compression="gzip", compression_opts=7)
    grp.create_dataset("y_test", data=y_test, compression="gzip", compression_opts=7)

    return X_train, y_train, X_test, y_test


def read_data(dataset="alcoholic_1"):
    data = h5py.File("data/data.h5", 'r')
    X_train = data[f"{dataset}/X_train"][:]
    y_train = data[f"{dataset}/y_train"][:]
    X_test = data[f"{dataset}/X_test"][:]
    y_test = data[f"{dataset}/y_test"][:]

    return X_train, y_train, X_test, y_test


def ml_method_setup(method, X_train, sig_train, y_train, file_name):
    if method == 'ts_knn':
        try:
            clf = joblib.load(open(f'models/{file_name}.pkl', 'rb'))
        except:  # noqa E722
            clf = GridSearchCV(
                Pipeline([
                    ('knn', KNeighborsTimeSeriesClassifier())
                ]),
                {'knn__n_neighbors': [3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25], 'knn__weights': ['uniform', 'distance']},
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
                n_jobs=-1,
                verbose=10
            )
            clf.fit(X_train, y_train)
            joblib.dump(clf.best_estimator_, f'models/{file_name}.pkl')

    elif method == 'ts_svc':
        try:
            clf = joblib.load(open(f'models/{file_name}.pkl', 'rb'))
        except:  # noqa E722
            clf = GridSearchCV(
                Pipeline([
                    ('svc', TimeSeriesSVC(random_state=0, probability=True))
                ]),
                {'svc__kernel': ['gak', 'rbf', 'poly'], 'svc__shrinking': [True, False],
                 'svc__C': [0.1, 0.2, 0.5, 1, 2, 5, 10]},
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
                n_jobs=-1,
                verbose=10
            )
            clf.fit(X_train, y_train)
            joblib.dump(clf.best_estimator_, f'models/{file_name}.pkl')

    elif method == 'lr':
        try:
            clf = joblib.load(open(f'models/{file_name}.pkl', 'rb'))
        except:  # noqa E722
            lr = LogisticRegression(random_state=0)
            parameters = {'C': [0.1, 0.2, 0.5, 1, 2, 5, 10],
                          'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
            clf = GridSearchCV(lr, parameters, n_jobs=-1, verbose=10)
            clf.fit(sig_train, y_train)
            joblib.dump(clf.best_estimator_, f'models/{file_name}.pkl')

    elif method == 'svc':
        try:
            clf = joblib.load(open(f'models/{file_name}.pkl', 'rb'))
        except:  # noqa E722
            svc = SVC(random_state=0, probability=True)
            parameters = {'kernel': ['rbf', 'poly'], 'shrinking': [True, False],
                          'C': [0.1, 0.2, 0.5, 1, 2, 5, 10]}
            clf = GridSearchCV(svc, parameters, n_jobs=-1, verbose=10)
            clf.fit(sig_train, y_train)
            joblib.dump(clf.best_estimator_, f'models/{file_name}.pkl')

    elif method == 'knn':
        try:
            clf = joblib.load(open(f'models/{file_name}.pkl', 'rb'))
        except:  # noqa E722
            knn = KNeighborsClassifier()
            parameters = {'n_neighbors': range(3, 30, 2), 'weights': ['uniform', 'distance']}
            clf = GridSearchCV(knn, parameters, n_jobs=-1, verbose=10)
            clf.fit(sig_train, y_train)
            joblib.dump(clf.best_estimator_, f'models/{file_name}.pkl')

    elif method == 'ada':
        try:
            clf = joblib.load(open(f'models/{file_name}.pkl', 'rb'))
        except:  # noqa E722
            ada = AdaBoostClassifier(random_state=0)
            parameters = {'n_estimators': [50, 100, 150], 'learning_rate': [0.1, 0.5, 1, 2]}
            clf = GridSearchCV(ada, parameters, n_jobs=-1, verbose=10)
            clf.fit(sig_train, y_train)
            joblib.dump(clf.best_estimator_, f'models/{file_name}.pkl')

    elif method == 'rf':
        try:
            clf = joblib.load(open(f'models/{file_name}.pkl', 'rb'))
        except:  # noqa E722
            rf = RandomForestClassifier(random_state=0)
            parameters = {'min_weight_fraction_leaf': [0.01, 0.1, 0.5],
                          'bootstrap': [True, False],
                          'max_depth': (2, 5, 10),
                          'max_leaf_nodes': (2, 5, 10),
                          'n_estimators': (100, 200, 300)}
            clf = GridSearchCV(rf, parameters, n_jobs=-1, verbose=10)
            clf.fit(sig_train, y_train)
            joblib.dump(clf.best_estimator_, f'models/{file_name}.pkl')

    else:
        clf = None

    return clf


def time_augment(arr):
    return np.vstack((arr.T, np.linspace(0, 1, num=arr.shape[0]))).T


def auto_ml(X_train, y_train, X_test, y_test, method, sig_level, dataset, ts_scale=True, standard_scale=True,
            time_aug=False):
    start = time.time()

    method_dict = {"rf": "Random Forests", "ada": "AdaBoost", "knn": "K Nearest Neighbours",
                   "svc": "Support Vector Machines", "lr": "Logistic Regression",
                   "ts_svc": "Time Series Support Vector Machines",
                   "ts_knn": "Time Series K Nearest Neighbours"}

    # initialise scalers
    ts_scaler = TimeSeriesScalerMinMax()
    scaler = StandardScaler()

    if time_aug:
        X_train = np.array(list(map(time_augment, X_train)))
        X_test = np.array(list(map(time_augment, X_test)))

    if ts_scale:
        X_train = ts_scaler.fit_transform(X_train)
        X_test = ts_scaler.fit_transform(X_test)

    if torch.cuda.is_available():
        X_train_torch = torch.from_numpy(X_train).cuda()
        X_test_torch = torch.from_numpy(X_test).cuda()

        sig_train_unscaled = signatory.signature(X_train_torch, sig_level)
        sig_train_unscaled = sig_train_unscaled.cpu().numpy()
        sig_test_unscaled = signatory.signature(X_test_torch, sig_level)
        sig_test_unscaled = sig_test_unscaled.cpu().numpy()

    else:
        X_train_torch = torch.from_numpy(X_train)
        X_test_torch = torch.from_numpy(X_test)

        sig_train_unscaled = signatory.signature(X_train_torch, sig_level)
        sig_train_unscaled = sig_train_unscaled.numpy()
        sig_test_unscaled = signatory.signature(X_test_torch, sig_level)
        sig_test_unscaled = sig_test_unscaled.numpy()

    if standard_scale:
        sig_train = scaler.fit_transform(sig_train_unscaled)
        sig_test = scaler.fit_transform(sig_test_unscaled)
    else:
        sig_train = sig_train_unscaled
        sig_test = sig_test_unscaled

    # file name formatting
    file_name = f"{dataset}_{method}"
    if not method.startswith("ts"):
        file_name += f"_sig{sig_level}"
    if ts_scale:
        file_name += "_ts_scale"
    if standard_scale:
        file_name += "_standard_scale"
    if time_aug:
        file_name += "_time_aug"

    clf = ml_method_setup(method, X_train, sig_train, y_train, file_name)

    # fit to data
    if method.startswith("ts"):
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        y_pred = clf.predict(X_test)
        cv_score = cross_val_score(clf, X_train, y_train, scoring='roc_auc', n_jobs=-1)
    else:
        y_pred_proba = clf.predict_proba(sig_test)[:, 1]
        y_pred = clf.predict(sig_test)
        cv_score = cross_val_score(clf, sig_train, y_train, scoring='roc_auc', n_jobs=-1)

    # Figures
    if len(np.unique(y_test)) == 2:
        plt.figure(figsize=(5, 5))
        bc = BinaryClassification(y_test, y_pred_proba, labels=["Class 0", "Class 1"])
        bc.plot_roc_curve()
        plt.title(f"Receiver Operating Characteristic Using {method_dict[method]}")
        plt.savefig(f"graphs/{file_name}.png")

    score_df = pd.DataFrame(columns=["Accuracy", "AUC", "F1 Measure", "Mean CV on Train"], index=["value"])
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)

    score_df["Accuracy"] = accuracy
    score_df["AUC"] = roc_auc
    score_df["F1 Measure"] = f1
    score_df["Mean CV on Train"] = cv_score.mean()
    score_df.to_csv(f"results/{file_name}.csv")

    print("accuracy is " + str(accuracy))
    print("auc is " + str(roc_auc))
    print("f1-measure " + str(f1))
    print(cv_score, "mean CV on Train = " + str(cv_score.mean()))

    end = time.time()
    print(end - start)


def main(dataset, method, sig_level, ts_scale=True, standard_scale=True, time_aug=False):
    X_train, y_train, X_test, y_test = read_data(dataset)

    auto_ml(X_train, y_train, X_test, y_test, method, sig_level, dataset,
            ts_scale=ts_scale, standard_scale=standard_scale, time_aug=time_aug)


method_list = ["rf", "ada", "knn", "svc", "lr", "ts_svc", "ts_knn"]
for i in ['alcoholic_1', 'alcoholic_12', 'alcoholic_21']:
    # write_alcoholic(subset=i[10:])
    for j in method_list:
        for st in [True, False]:
            for ts in [True, False]:
                for sig in [1, 2]:
                    for ta in [True, False]:
                        print(i, j, st, ts, sig, ta)
                        main(i, j, sig_level=sig, time_aug=ta, standard_scale=st, ts_scale=ts)

import signatory
from tslearn.datasets import UCR_UEA_datasets
import torch
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

device = torch.device("cuda:0")
data = UCR_UEA_datasets().load_dataset("Phoneme")

t_data = torch.from_numpy(data[0]).cuda()

X_train = signatory.signature(t_data, 12).cpu()
y_train = data[1]
X_test = signatory.signature(torch.from_numpy(data[2]).cuda(), 12).cpu()
y_test = data[3]

start = time.time()
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
end = time.time()
y_pred = clf.predict_proba(X_test)
y_pred_class = clf.predict(X_test)
# print(roc_auc_score(y_test, y_pred, multi_class='ovo'))
print(accuracy_score(y_test, y_pred_class))
# print(end - start)

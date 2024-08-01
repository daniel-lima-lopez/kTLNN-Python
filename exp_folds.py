import TLNN as tl
import numpy as np
import sklearn as sk
import pandas as pd
import matplotlib.pyplot as plt

# load iris dataset
name = 'wine'
dataset = pd.read_csv(f'Datasets/{name}.csv')
X = dataset.drop('class', axis=1).values
y = dataset['class'].values

# 5-fold indices split
kf = sk.model_selection.KFold(n_splits=10, shuffle=True, random_state=0)

# values for k
ks = [1,3,5,7,9,11]
acc = [] # information of accuracy
for ki in ks:
    print(f'\n### K = {ki} ###')
    aux_acc = []
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        # data split
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]

        # predictions
        classifier = tl.kTLNN(k=ki, kb_factor=1.4)
        classifier.fit(X_train, y_train)
        preds = classifier.predict(X_test)
        accF = sk.metrics.accuracy_score(y_true=y_test, y_pred=preds)
        aux_acc.append(accF)

        print(f"- Fold {i+1}: {accF}")
    print(f'- Mean: {np.mean(aux_acc)}')
    acc.append(aux_acc)

#acc = np.array(acc)

# grafico
fig, ax = plt.subplots()
ax.set_ylabel('accuracy')
ax.set_xlabel('K values')
ax.set_title(f'Accuracy on 10-folds on {name} dataset')
ax.boxplot(acc, tick_labels=ks)

fig.savefig(f'imgs/ks_{name}',bbox_inches ="tight",dpi=300)
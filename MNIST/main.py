import sys
assert sys.version_info >= (3,7)
from packaging import version
import sklearn
assert version.parse(sklearn.__version__)>= version.parse("1.0.1")
import matplotlib.pyplot as plt

plt.rc('font' , size = 14)
plt.rc('axes' , labelsize = 14 , titlesize = 14)
plt.rc('legend' , fontsize = 14)
plt.rc('xtick' , labelsize = 10)
plt.rc('ytick' , labelsize = 10)

from pathlib import Path
IMAGES_PATH = Path() / "images" / "classification"
IMAGES_PATH.mkdir(parents=True , exist_ok = True)

def save_fig(fig_id , tight_layout = True , fig_extension = "png" , resolution = 300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path , format = fig_extension , dpi = resolution)

from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784' , as_frame = False)
# print(mnist.DESCR)
# print(mnist.keys())

X , Y = mnist.data , mnist.target
# print(X)
# print(Y)

def plot_digit(image_data):
    image = image_data.reshape(28,28)
    plt.imshow(image , cmap="binary")
    plt.axis("off")

some_digit = X[0]
plot_digit(some_digit)
save_fig("some_digit_plot")
# plt.show()


plt.figure(figsize=(9,9))
for idx , image_data in enumerate(X[:100]):
    plt.subplot(10 , 10 , idx + 1)
    plot_digit(image_data)
plt.subplots_adjust(wspace=0 , hspace=0)
save_fig("more_digits_plot" , tight_layout=False)
# plt.show()


X_train , X_test , Y_train , Y_test = X[:60000] , X[60000:] , Y[:60000] , Y[60000:]

y_train_5 = (Y_train == '5')
y_test_5 = (Y_test == '5')

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state= 42)
sgd_clf.fit(X_train , y_train_5)
print(sgd_clf.predict([some_digit]))

from sklearn.model_selection import cross_val_score
print(cross_val_score(sgd_clf , X_train , y_train_5 , cv= 3 , scoring="accuracy"))

from sklearn.dummy import DummyClassifier
dummy_clif = DummyClassifier()
dummy_clif.fit(X_train , y_train_5)
print(any(dummy_clif.predict(X_train)))

print(cross_val_score(dummy_clif , X_train , y_train_5 , cv = 3 , scoring="accuracy"))

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
skfolds = StratifiedKFold(n_splits=3)
for train_index , test_index in skfolds.split(X_train , y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]

    clone_clf.fit(X_train_folds , y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct/ len(y_pred))

from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf , X_train , y_train_5 , cv= 3)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train_5 , y_train_pred)
print(cm)

y_train_perfect_predicitions = y_train_5
print(confusion_matrix(y_train_5 ,y_train_perfect_predicitions))

from sklearn.metrics import precision_score , recall_score
print(precision_score(y_train_5 , y_train_pred))
print(recall_score(y_train_5 , y_train_pred))

from sklearn.metrics import f1_score
print(f1_score(y_train_5, y_train_pred))

y_scores = sgd_clf.decision_function([some_digit])
print(y_scores)

threshold = 0
y_some_digit_pred = (y_scores > threshold)
print(y_some_digit_pred)

threshold = 3000
y_some_digit_pred = (y_scores > threshold)
print(y_some_digit_pred)

y_scores = cross_val_predict(sgd_clf , X_train, y_train_5 , cv =3 , method="decision_function")

from sklearn.metrics import precision_recall_curve
precisions ,recalls , thresholds = precision_recall_curve(y_train_5 , y_scores)

plt.figure(figsize=(8,4))
plt.plot(thresholds , precisions[:-1] , "b--" , label= "Precision" , linewidth = 2)
plt.plot(thresholds , recalls[:-1], "g-" , label = "Recall" , linewidth =2 )
plt.vlines(threshold , 0 , 1.0 , "k" , "dotted" , label = "threshold")

idx = (thresholds >= threshold).argmax()
plt.plot(thresholds[idx], precisions[idx] , "bo")
plt.plot(thresholds[idx] , recalls[idx] ,"go")
plt.axis([-50000 , 50000 , 0 ,1])
plt.grid()
plt.xlabel("Threshold")
plt.legend(loc = "center right")
save_fig("precision_recall_vs_threshold_plot")

plt.show()

import matplotlib.patches as patches
import matplotlib.pyplot as plt # Assuming you imported this earlier!

plt.figure(figsize=(6,5))

# 1. THE MAIN CURVE
plt.plot(recalls, precisions, linewidth=2, label="precision/recall curve")

# 2. THE HIGHLIGHTED POINT (Fixed brackets here)
plt.plot([recalls[idx], recalls[idx]], [0, precisions[idx]], "k:") # Vertical line
plt.plot([0.0, recalls[idx]], [precisions[idx], precisions[idx]], "k:") # Horizontal line
plt.plot([recalls[idx]], [precisions[idx]], "ko", label="point at threshold 3,000") # The black dot

# 3. THE ARROW & TEXT
plt.gca().add_patch(patches.FancyArrowPatch(
    (0.79, 0.60), (0.61, 0.78),
    connectionstyle="arc3,rad=.2",
    arrowstyle="Simple, tail_width=1.5, head_width=8, head_length=10",
    color="#444444"))
plt.text(0.56, 0.62, "Higher\nthreshold", color="#333333")

# 4. GRAPH FORMATTING
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.axis([0, 1, 0, 1])
plt.grid()
plt.legend(loc="lower left")
save_fig("precision_vs_recall_plot")
plt.show()

idx_for_90_precision = (precisions>= 0.90).argmax()
threshold_for_90_precision = thresholds[idx_for_90_precision]
print(threshold_for_90_precision)

from sklearn.metrics import roc_curve
fpr , tpr , thresholds = roc_curve(y_train_5 , y_scores)
idx_for_threshold_at_90 = (thresholds <= threshold_for_90_precision).argmax()
tpr_90 , fpr_90 = tpr[idx_for_threshold_at_90] , fpr[idx_for_threshold_at_90]

plt.figure(figsize=(6,5))
plt.plot(fpr , tpr , linewidth = 2 ,label = "ROC curve")
plt.plot([0,1] , [0,1] , 'k:' , label = "Random classidier ROC curve")
plt.plot([fpr_90] , [tpr_90] , "ko", label = "threshold for 90% precision")
plt.grid()
plt.legend(loc = "lower right" , fontsize = 12)
plt.xlabel('Fasle Positive Rate (fall- out)' , color = "#333333")
plt.ylabel('True Positive rate (Recall)' )
print(save_fig("roc_curve_plot"))
plt.show()


from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_train_5 , y_scores))

from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state= 42)
y_probas_forest = cross_val_predict(forest_clf, X_train , y_train_5 , cv =3 , method="predict_proba")

print(y_probas_forest[:2])


y_scores_forest = y_probas_forest[: ,1]
precision_forest ,recalls_forest , thresholds_forest = precision_recall_curve(y_train_5, y_scores_forest)
plt.plot(recalls_forest , precision_forest , "b-" , linewidth = 2 , label = "Random forest")
plt.plot(recalls , precisions , "--" ,linewidth =3 , label = "SGD" )
plt.grid()
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc = "lower left")
save_fig("Forest with precision")
plt.show()
y_train_pred_forest = y_probas_forest[:, 1] >= 0.5  # positive proba ≥ 50%
f1_score(y_train_5, y_train_pred_forest)
print(recall_score(y_train_5, y_train_pred_forest))
print(precision_score(y_train_5, y_train_pred_forest))


from sklearn.svm import SVC
svm_clf = SVC(random_state= 42)
svm_clf.fit(X_train[:2000] , Y_train[:2000])
svm_clf.predict([some_digit])
some_digit_scores = svm_clf.decision_function([some_digit])
print(some_digit_scores.round(2))
class_id = some_digit_scores.argmax()
print(class_id)
svm_clf.classes_
svm_clf.classes_[[class_id]]

from sklearn.multiclass import OneVsOneClassifier
ovr_clf = OneVsOneClassifier(SVC(random_state=42))
ovr_clf.fit(X_train[:2000] , Y_train[:2000])
ovr_clf.predict([some_digit])
len(ovr_clf.estimators_)

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train , Y_train)
sgd_clf.predict([some_digit])

sgd_clf.decision_function([some_digit]).round(2)
cross_val_score(sgd_clf , X_train , Y_train , cv=3 , scoring= "accuracy")


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype("float64"))
cross_val_score(sgd_clf, X_train_scaled ,Y_train , cv=3 , scoring="accuracy")

from sklearn.metrics import ConfusionMatrixDisplay
y_train_pred = cross_val_predict(sgd_clf , X_train_scaled , y_train_5 , cv=3)
ConfusionMatrixDisplay.from_predictions( Y_train , y_train_pred)
plt.show()

ConfusionMatrixDisplay.from_predictions(Y_train , y_train_pred , normalize="true" , values_format=".0%")
plt.show()

sample_weight = (y_train_pred != Y_train)
ConfusionMatrixDisplay.from_predictions(Y_train, y_train_pred,
                                        sample_weight=sample_weight,
                                        normalize="true", values_format=".0%")
plt.show()

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
y_train_large = (Y_train >= '7')
y_train_odd = (Y_train.astype('int8')%2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)  

y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
f1_score(y_multilabel, y_train_knn_pred, average="macro")
f1_score(y_multilabel, y_train_knn_pred, average="weighted")

from sklearn.multioutput import ClassifierChain

chain_clf = ClassifierChain(SVC(), cv=3, random_state=42)
chain_clf.fit(X_train[:2000], y_multilabel[:2000])
print(chain_clf.predict([some_digit]))

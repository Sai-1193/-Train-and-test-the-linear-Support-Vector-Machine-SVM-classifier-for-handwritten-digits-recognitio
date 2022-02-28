import numpy as np
from mnist import MNIST
from mlxtend.data import loadlocal_mnist
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import scale


images, labels = loadlocal_mnist(labels_path="C:\\Users\\saika\\Downloads\\Assign1\\train-labels-idx1-ubyte", images_path="C:\\Users\\saika\\Downloads\\Assign1\\train-images-idx3-ubyte")
threshold = []
for image in images:
    threshold.append(image>90)
threshold = np.array(threshold)
Y = labels[0:9999]
X = threshold[0:9999]
X = X/255.0
X_train, X_test, y_train, y_test = train_test_split(scale(X), Y, test_size=0.3, train_size=0.2, random_state=10)
clf = SVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_rbf = metrics.accuracy_score(y_true=y_test, y_pred=y_pred)
print("Accuracy:", accuracy_rbf)
print("Confusion Matrix:", '\n', metrics.confusion_matrix(y_true=y_test, y_pred=y_pred))
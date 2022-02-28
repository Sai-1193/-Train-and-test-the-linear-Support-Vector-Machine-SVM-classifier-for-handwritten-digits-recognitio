import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from mnist import MNIST
from PIL import Image, ImageDraw
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, confusion_matrix


def getplot(cm_array):
    df = pd.DataFrame(cm_array, range(len(cm_array)), range(len(cm_array)))
    sns.set(font_scale=1.5)
    sns.heatmap(df, annot=True, annot_kws={"size": 20})
    plt.show()


def getAccuracy(train_x,train_y,imageIndex,labelsIndex,clf):

    clf.fit(train_x, train_y)

    test_x = imageIndex
    expected=labelsIndex.tolist()

    print("Compute predictions")
    predicted = clf.predict(test_x)

    print("Accuracy: ", accuracy_score(expected, predicted))
    cm = confusion_matrix(y_true=expected, y_pred=predicted)
    getplot(cm)

# Load dataset
mndata = MNIST('')
images, labels = mndata.load_training()

#part-1
train_x = images[1000:1999]
train_y = labels[1000:1999]
imageIndex=images[3000:3099]
labelsIndex=labels[3000:3099]
clf = LinearSVC(dual=False)
#getAccuracy(train_x,train_y,imageIndex,labelsIndex,clf)

#part-2
train_x = images[20000:29999]
train_y = labels[20000:29999]
imageIndex = images[30000:30100]
labelsIndex = labels[30000:30100]
clf = LinearSVC(dual=False, max_iter=10000)
#getAccuracy(train_x,train_y,imageIndex,labelsIndex,clf)

#part-3
train_x = images[20000:29999]
train_y = labels[20000:29999]
imageIndex = images[30000:31000]
labelsIndex = labels[30000:31000]
clf = LinearSVC(dual=False, max_iter=10000)
#getAccuracy(train_x,train_y,imageIndex,labelsIndex,clf)

#polynomial_kernel
train_x = images[20000:29999]
train_y = labels[20000:29999]
imageIndex = images[30000:31000]
labelsIndex = labels[30000:31000]
clf = SVC(kernel='poly', degree=8, max_iter=10000)
#getAccuracy(train_x,train_y,imageIndex,labelsIndex,clf)

#gaussian_kernel
train_x = images[20000:29999]
train_y = labels[20000:29999]
imageIndex = images[30000:31000]
labelsIndex = labels[30000:31000]
clf = SVC(kernel='rbf', max_iter=10000)
#getAccuracy(train_x,train_y,imageIndex,labelsIndex,clf)

#sigmoid_kernel
train_x = images[20000:29999]
train_y = labels[20000:29999]
imageIndex = images[30000:31000]
labelsIndex = labels[30000:31000]
clf = SVC(kernel='sigmoid', max_iter=10000)
getAccuracy(train_x,train_y,imageIndex,labelsIndex,clf)




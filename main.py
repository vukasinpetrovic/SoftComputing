import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)

def reshape_data(input_data):
    nsamples, nx, ny = input_data.shape
    return input_data.reshape((nsamples, nx*ny))

def resize_image(img):
    w, h = 512, 512
    return cv2.resize(img, (w, h), interpolation = cv2.INTER_NEAREST)


train_csv = pd.read_csv("images/train/train_labels.csv")
train_images = []
train_labels = []

for i in range(len(train_csv)):
    img = resize_image(load_image("images/train/" + train_csv["file"][i]))
    train_images.append(img)
    train_labels.append(train_csv["labels"][i])

print("Loaded training images: ", train_labels)

test_csv = pd.read_csv("images/test/test_labels.csv")
test_images = []
test_labels = []

for i in range(len(test_csv)):
    img = resize_image(load_image("images/test/" + test_csv["file"][i]))
    test_images.append(img)
    test_labels.append(test_csv["labels"][i])

print("Loaded test images: ", test_labels)

nbins = 20
cell_size = (24, 24)
block_size = (6, 6)

hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
                                  img.shape[0] // cell_size[0] * cell_size[0]),
                        _blockSize=(block_size[1] * cell_size[1],
                                    block_size[0] * cell_size[0]),
                        _blockStride=(cell_size[1], cell_size[0]),
                        _cellSize=(cell_size[1], cell_size[0]),
                        _nbins=nbins)

print("Computing, please wait...")

features_train = []
for img in train_images:
    features_train.append(hog.compute(img))

x = reshape_data(np.array(features_train))
y = np.array(train_labels)

clf_svm = SVC(kernel='linear')
clf_svm = clf_svm.fit(x, y)

features_test = []
for img in test_images:
    features_test.append(hog.compute(resize_image(img)))

x_test = reshape_data(np.array(features_test))
y_test = np.array(test_labels)

test_prediction = clf_svm.predict(x_test)

accuracy = accuracy_score(y_test, test_prediction)
print("Prediction accuracy: ", accuracy, " (", (accuracy * 100), "%)")

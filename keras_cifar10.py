from sklearn.preprocessing import LabelBinarizer
from sklearn.datasets import fetch_openml
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt 
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="Path to plots")

args = vars(ap.parse_args())

print("[INFO] loading CIFAR-10 data...")

((trainX, trainY), (testX, testY)) = cifar10.load_data()

trainX = trainX.astype("float")/255.0
# trainY = trainY.astype("float")/255.0
testX = testX.astype("float")/255.0
# testY = testY.astype("float")/255.0
trainX = trainX.reshape(trainX.shape[0], 32*32*3)
testX = testX.reshape(testX.shape[0], 32*32*3)

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

label_names = ["airplane", "automobiles", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

model = Sequential()

model.add(Dense(1024, input_shape=(32*32*3,), activation="relu"))
model.add(Dense(512, activation="relu"))
model.add(Dense(10, activation="softmax"))

print("[INFO] training network...")
sgd = SGD(0.01)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=100, batch_size=32)

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=28)

print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=label_names))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")

plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("# epoch")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])

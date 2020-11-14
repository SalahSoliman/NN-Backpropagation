import argparse
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())





plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), np.arange(0, 100), label="train_loss")

plt.plot(np.arange(0, 100), np.arange(0, 100), label="train_loss")
plt.plot(np.arange(0, 100), np.arange(0, 100), label="val_loss")
plt.plot(np.arange(0, 100), np.arange(0, 100), label="train_acc")
plt.plot(np.arange(0, 100), np.arange(0, 100), label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("# epoch")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])
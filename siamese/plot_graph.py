import sys
import numpy as np
import matplotlib.pyplot as plt
import random

def smooth(arr, r):
    smoothed = np.zeros(len(arr))
    for i in range(len(arr)//r):
        chunk = arr[i*r:i*r+r]
        avg = np.sum(chunk) / len(chunk)
        smoothed[i*r:i*r+r] = avg
    return smoothed[smoothed >0]

tr_loss = np.load("train_loss", allow_pickle= True)
plt.figure()
plt.title('Loss')

t1 = plt.plot(smooth(tr_loss,100))
plt.grid(color='k', linestyle='-', linewidth=0.1)
plt.show()

tr_acc = np.load("train_acc", allow_pickle= True)
plt.figure()
plt.title('Accuracy')
x = 100*np.arange(len(tr_acc))

t2 = plt.plot(x, smooth(tr_acc,1))
plt.grid(color='k', linestyle='-', linewidth=0.1)
plt.show()
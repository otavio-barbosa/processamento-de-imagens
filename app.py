import cv2
from skimage import feature
from sklearn import metrics, preprocessing
import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np

matplotlib.use('tkagg')
image = cv2.imread('./normal/IM-0115-0001.jpeg', cv2.IMREAD_GRAYSCALE)
lbp = feature.local_binary_pattern(image, P=8, R=1, method="uniform")
hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10 + 2), range=(0, 10 + 1))

hist = hist.astype("float")
hist /= (hist.sum() + 1e-7)

plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Imagem Original")

plt.subplot(1, 2, 2)
plt.imshow(lbp, cmap="gray")
plt.title("Local Binary Pattern (LBP)")


plt.show()

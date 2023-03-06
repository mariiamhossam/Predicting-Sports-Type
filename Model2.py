import warnings
warnings.filterwarnings('ignore')
warnings.warn('ConvergenceWarning')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D,MaxPooling2D,Dense,Dropout,Input,Flatten,Activation
#from Preprocessing import *
import cv2
import os
import random
import pandas as pd
import numpy as np
from PIL import Image
from os import listdir
from os.path import splitext
#importing libraries and modules
from tkinter import filedialog
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
imSize=200
model=tf.keras.models.load_model("VGG.h5")
def create_test_data():
    testing_data = []
    image_names = []

    for img in os.listdir("Test"):
        image_names.append(img)
        path = os.path.join("Test", img)
        image = cv2.imread(path)
        image = image[:, :, [2, 1, 0]]  # Make it RGB
        image = cv2.resize(image, (imSize, imSize))
        testing_data.append(image)

    #     testing_data = np.array(testing_data)
    #     indx = np.arange(testing_data.shape[0])
    #     testing_data = testing_data[indx]
    #     testing_data=testing_data.tolist()
    # np.save('test_data.npy', testing_data)
    return testing_data, image_names
testing_data,image_names=create_test_data()

testing_data=np.array(testing_data).reshape(-1,imSize, imSize, 3)
prediction = model.predict(testing_data)
prediction=prediction.tolist()
pred=[]
for i in prediction:
    pred.append(i.index(max(i)))
df = pd.DataFrame()
df['image_name']=image_names
df['label']=pred
df.to_csv("submission.csv", index=False)

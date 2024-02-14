

import cv2 
from glob import glob
import pathlib
import numpy as np
import pandas as pd
import tensorflow as tf 
import matplotlib.pyplot as plt

# from google.colab import drive
# drive.mount('/content/drive')

from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Flatten,Dense,Conv3D,MaxPool3D #cnn layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

bird_path=pathlib.Path(r"C:\Users\salma\OneDrive\Desktop\Project\Data\Train")
# bird_path=pathlib.Path(r"C:\Users\91701\OneDrive\Desktop\Project Saafi\Animal\Dataset\Birds_25\train")

A=list(bird_path.glob("Barbet/*.jpg"))
B=list(bird_path.glob("Crow/*.jpg"))
C=list(bird_path.glob("Hornbill/*.jpg"))
D=list(bird_path.glob("Kingfisher/*.jpg"))
E=list(bird_path.glob("Myna/*.jpg"))
F=list(bird_path.glob("Peacock/*.jpg"))
G=list(bird_path.glob("Pitta/*.jpg"))
H=list(bird_path.glob("Rosefinch/*.jpg"))
I=list(bird_path.glob("Tailorbird/*.jpg"))
J=list(bird_path.glob("Wagtail/*.jpg"))
len(A), len(B), len(C), len(D), len(E), len(F), len(G), len(H), len(I), len(J)


bird_dict = {"Barbet":A,
             "Crow":B,
             "Hornbill":C,
             "Kingfisher":D,
             "Myna":E,
             "Peacock":F,
             "Pitta":G,
             "Rosefinch":H,
             "Tailorbird":I,
             "Wagtail":J
             }
bird_class= {"Barbet":0,
             "Crow":1,
             "Hornbill":2,
             "Kingfisher":3,
             "Myna":4,
             "Peacock":5,
             "Pitta":6,
             "Rosefinch":7,
             "Tailorbird":8,
             "Wagtail":9
             }


x=[]
y=[]

print("starting.....")
for i in bird_dict:
  bird_name=i
  bird_path_list=bird_dict[bird_name]
  print("Image resizing....")
  for path in bird_path_list:
    img=cv2.imread(str(path))
    img=cv2.resize(img,(224,224))
    img=img/255
    x.append(img)
    cls=bird_class[bird_name]
    y.append(cls)

len(x)
print("complete")
x=np.array(x)
y=np.array(y)

from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=0.75,random_state=1)

len(xtrain),len(ytrain),len(xtest),len(ytest)

xtrain.shape

"""xtrain.shape,xtest.shape"""

xtrain.shape,xtest.shape

"""xtrain.shape,xtest.shape"""

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')

print("[INFO] summary for base model...")
print(base_model.summary())

from tensorflow.keras.layers import MaxPooling2D
# from keras.layers.core import
from keras.layers import Dropout

from tensorflow.keras.models import Model
# construct the head of the model that will be placed on top of the
# the base model
headModel = base_model.output
headModel = MaxPooling2D(pool_size=(2, 2))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(32, activation="relu")(headModel)
headModel = Dropout(0.2)(headModel)
headModel = Dense(30, activation="softmax")(headModel)
# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=base_model.input, outputs=headModel)
# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in base_model.layers:
	layer.trainable = False

from tensorflow.keras.optimizers import Adam
# compile our model (this needs to be done after our setting our
# layers to being non-trainable)
print("[INFO] compiling model...")
opt = Adam(lr=1e-4)
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])
# train the head of the network for a few epochs (all other layers
# are frozen) -- this will allow the new FC layers to start to become
# initialized with actual "learned" values versus pure random
print("[INFO] training head...")
#H = model.fit(
	#data_generator.flow(xtrain, ytrain, batch_size=32),
#	steps_per_epoch=len(xtrain) // 32,
	#validation_data=valAug.flow(xtest, ytest),
	#validation_steps=len(xtest) // 32,
#	epochs=5)

model_hist=model.fit(xtrain,ytrain,epochs=25,validation_data=(xtest,ytest),batch_size=180)

model.save("Model.h5")
# model.save('my_model.keras')

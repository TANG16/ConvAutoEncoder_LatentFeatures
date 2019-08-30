#This part of code generates training images in batch
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
gen = ImageDataGenerator()
import cv2

datagen = ImageDataGenerator(width_shift_range=0.2)



#We use ImageDataGenerator and flow_from_directory to Generate One hot encoddings of the training images
train_im = ImageDataGenerator(

        rescale=1./255,

        shear_range=0.2,

        horizontal_flip=False)
         
def train_images():
    train_generator = train_im.flow_from_directory(

        './provide/path/to/the/training_images/',

        target_size=(64, 64),
        color_mode='rgb',
        batch_size=64,
        shuffle = True,
        class_mode='categorical')
    x =  train_generator
    return x[0][0], x[0][1]



if __name__=='__main__':
    x,y = train()
    for i in x:
        plt.imshow(i)
        plt.show()

        






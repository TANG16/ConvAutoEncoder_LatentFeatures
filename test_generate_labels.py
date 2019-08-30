import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
gen = ImageDataGenerator()
import cv2


#(X, y), _ = mnist.load_data()

#X = X.reshape(X.shape[0], 1, 28, 28)

#X = X[:100]





datagen = ImageDataGenerator(width_shift_range=0.2)

#datagen.fit(X)




"""
imgs = []

batches = 0

for i in datagen.flow(X, batch_size=32):

    batches += 1

    for x in i:

        img = np.asarray(x).reshape((28, 28))

        plt.imshow(img, cmap='gray')

        plt.show()




def train_images():
        train_generator = datagen.flow_from_directory(
        './dataset_ex',  # this is the target directory
        target_size=(100, 100),  # all images will be resized
        color_mode='rgb',
        classes=None,
        batch_size=30,
        shuffle=True,
        seed=None,
        save_to_dir=None,
        save_prefix='',
        class_mode='categorical')
        x = train_generator
        return  x[0][0], x[0][1]



"""
train_im = ImageDataGenerator(

        rescale=1./255,

        shear_range=0.2,

        #zoom_range=1,

        horizontal_flip=False)
         
def train_images():
    train_generator = train_im.flow_from_directory(

        '../../Desktop/datasets/outex_train',

        target_size=(64, 64),
        color_mode='rgb',
        batch_size=64,
        shuffle = True,
        class_mode='categorical')
    x =  train_generator
    return x[0][0], x[0][1]



if __name__=='__main__':
    x,y = train()
    #print(len(x[0][0]))
    #print(x[0][1])
    for i in x:
        plt.imshow(i)
        plt.show()

        




    #cv2.destroyAllWindows()
        #plt.imshow(cv2.cvtColor(i, cv2.COLOR_RGB2BGR))
        #plt.show()




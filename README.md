# Convolutional AutoEncoder for similar image search

This project provides utilities to build a  Convolutional AutoEncoder for similar object search

This project is on TensorFlow and Keras.


## Experiments

`convolutional_autoencoder_outex.py` shows an example of a CAE for the Outex Dataset.

The structure of this conv autoencoder is shown below:

![autoencoder structure](https://cloud.githubusercontent.com/assets/13087207/23317657/540f170a-fa9d-11e6-9bcb-8b529a805a9f.png)

The encoding part has 2 convolution layers (each followed by a max-pooling layer) and a fully connected layer. This part
would encode an input image into a 20-dimension vector (representation). And then the decoding part, which has 1 fully connected layer
and 2 convolution layers, would decode the representation back to a 64x64 image (reconstruction).

Training was done using GTX1060 GPU, batch size 100, 100000 passes.

Trained weights (saved in the saver directory) of the 1st convolutional layer are shown below:
![conv_1_weights](https://github.com/surajitsaikia27/ConvAutoEncoder_LatentFeatures/blob/master/intial.png)

And here's some of the reconstruction results:
![reconstructions](https://github.com/surajitsaikia27/ConvAutoEncoder_LatentFeatures/blob/master/activation.png)

'outex_extractor.py' is to extract deep features of images

'test_generate_images.py' is to generate one hot encoded training images

## Implementation


#!/usr/bin/env python
# coding: utf-8
#
# Learn a model based on a defined set of image classes and then use it to classify novel images.
#
# Architecture based on LeNet and the following tutorials:
#   https://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/
#   https://d4nst.github.io/2017/01/12/image-orientation/

import argparse
import numpy as np
import cv2
import os
from imutils import paths

IMAGE_SIZE = (28, 28)
TEST_DATA_SPLIT = 0.25

def build_lenet(input_shape, num_classes, weightsPath=None):

    from keras.models import Sequential
    from keras.layers.convolutional import Conv2D
    from keras.layers.convolutional import MaxPooling2D
    from keras.layers.core import Activation, Dropout, Flatten, Dense

    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    if weightsPath is not None:
        model.load_weights(weightsPath)

    return model
 

def image_to_feature(image, size=IMAGE_SIZE):
    return cv2.resize(image, size)


def rotate(image, angle=90):
    rows = image.shape[0]
    cols = image.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    return cv2.warpAffine(image,M,(cols,rows))
    

def add_dataset(class_name, class_dir, raw_data, raw_labels):

    imagePaths = list(paths.list_images(class_dir))

    for (i, imagePath) in enumerate(imagePaths):
        
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)    

        raw_data.append(image_to_feature(image))
        raw_labels.append(class_name)

    print("Loaded %d images for class '%s'" % (i,class_name))


def train(classes_filepath, model_filepath):

    from keras.optimizers import SGD, Adadelta
    from keras.losses import categorical_crossentropy
    from keras.callbacks import EarlyStopping
    from keras.utils import np_utils
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split

    raw_data = []
    raw_labels = []

    classes = [d for d in os.listdir(classes_filepath) if os.path.isdir(os.path.join(classes_filepath, d))]
    num_classes = len(classes)

    for class_name in classes:
        add_dataset(class_name, os.path.join(classes_filepath, class_name), raw_data, raw_labels)

    # encode the labels, converting them from strings to integers
    le = LabelEncoder()
    labels = le.fit_transform(raw_labels)

    # scale the input image pixels to the range [0, 1], then transform
    # the labels into vectors in the range [0, num_classes] -- this
    # generates a vector for each label where the index of the label
    # is set to `1` and all other entries to `0`
    data = np.array(raw_data) / 255.0
    data = data[:, :, :, np.newaxis]
    labels = np_utils.to_categorical(labels, num_classes)
     
    # partition the data into training and testing splits, using 75%
    # of the data for training and the remaining 25% for testing
    (trainData, testData, trainLabels, testLabels) = train_test_split(
        data, labels, test_size=TEST_DATA_SPLIT, random_state=42)
        
    print("Train data shape: "+ str(trainData.shape))
    print("Test data shape: "+ str(testData.shape))

    # define the model
    model = build_lenet((IMAGE_SIZE[0], IMAGE_SIZE[1], 1), num_classes=num_classes, weightsPath=None)
    print(model.summary())

    # train
    verbose=1
    epochs=20
    batch_size=128
    opt = SGD(lr=0.01)
    #opt = Adadelta()
    model.compile(loss=categorical_crossentropy, optimizer=opt, metrics=["accuracy"])
    early_stopping = EarlyStopping(monitor='loss', patience=2, verbose=1, mode='auto')
    model.fit(trainData, trainLabels, batch_size=batch_size, epochs=epochs, callbacks=[early_stopping], verbose=verbose)

    # evaluate
    (loss, accuracy) = model.evaluate(testData, testLabels, batch_size=batch_size, verbose=verbose)
    print("Test set loss: {:.4f}, accuracy: {:.2f}%".format(loss, accuracy * 100))

    # save weights to disk
    model.save(model_filepath, overwrite=True)

 
def classify(image_path, model_filepath):
    
    from keras.models import load_model
    model = load_model(model_filepath)

    filepaths = [image_path,]
    if (os.path.isdir(image_path)):
        print("Scanning input directory for images")
        from os import listdir
        from os.path import isfile, join
        filepaths = [join(image_path, f) for f in listdir(image_path) if isfile(join(image_path, f))]

    print("Input, Prediction, Confidence")
    for filepath in filepaths:
        # load the image, resize it to a fixed size (ignoring
        # aspect ratio), and then extract features from it
        filename = image_path[filepath.rfind("/") + 1:]
        image = cv2.imread(filepath)
        if image is None:
            print("Cannot read image: "+filepath)
            continue

        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        features = image_to_feature(image) / 255.0
        features = features[np.newaxis, :, :, np.newaxis]

        probs = model.predict(features)[0]
        prediction = probs.argmax(axis=0)
        predicted_class = prediction # todo: translate the class index

        print("%s, %s, %2.4f" % (filepath, predicted_class, probs[prediction]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learn and use a CNN to classify images')
    parser.add_argument('-M', '--model', type=str, required=True, help='Path to model file')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-C', '--classes', type=str, nargs='?', default=None, help='Folder containing image classes, with each containing some examples')
    group.add_argument('-I', '--input', type=str, nargs='?', default=None, help='Folder containing to image(s) to classify')
    args = parser.parse_args()

    if args.classes:
        train(args.classes, args.model)
    else:
        classify(args.input, args.model)



'''
models.py
Contains custom utilities for creating DNN models.
Author: Vitoria Barin Pacela
e-mail: vitoria.barimpacela@helsinki.fi
'''
import h5py
import numpy as np
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Activation, Input, Dense, Dropout, merge, Reshape, Convolution3D, MaxPooling3D, Flatten


def loadModel(name, weights=False):
    '''
    Adapted from Kaustuv Datta and Jayesh Mahapatra's CaloImageMacros.
    Loads models from json file.
    :param name: (String) name of the json file.
    :param weights: (boolean) whether or not to load the weights.
    :return: loaded model.
    '''

    json_file = open('%s.json' % name, 'r')
    loaded = json_file.read()
    json_file.close()

    model = model_from_json(loaded)

    # load weights into new model
    if weights == True:
        model.load_weights('%s.h5' % name)
    # print model.summary()

    print("Loaded model from disk")
    return model


def saveModel(model, name="regression"):
    '''
    Saves model as json file.
    Adapted from Kaustuv Datta and Jayesh Mahapatra's CaloImageMacros.
    :param model: model to be saved.
    :param name: (String) name of the model to be saved.
    :return: saved model.
    '''
    model_name = name
    model.summary()
    model.save_weights('%s.h5' % model_name, overwrite=True)
    model_json = model.to_json()
    with open("%s.json" % model_name, "w") as json_file:
        json_file.write(model_json)


def saveLosses(hist, name="regression"):
    '''
    Adapted from Kaustuv Datta and Jayesh Mahapatra's CaloImageMacros.
    Saves model losses into an HDF5 file.
    :param hist: array of losses in the trained model.
    :param name: (String) name of the file to be saved
    '''
    loss = np.array(hist.history['loss'])
    vaLoss = np.array(hist.history['val_loss'])

    f = h5py.File("%s_losses.h5" % name, "w")
    f.create_dataset('loss', data=loss)
    f.create_dataset('val_loss', data=vaLoss)
    f.close()


def defModel(loss='mse', name="regression"):
    '''
    Defines regression model.
    :param loss: Keras' loss function to be used. Recommended: mse, mean_absolute_error,
     mean_squared_logarithmic_error, mean_absolute_percentage_error.
    :param name: (String) name to save the file as.
    :return: model.
    '''

    # ECAL input
    input1 = Input(shape=(25, 25, 25))
    r = Reshape((25, 25, 25, 1))(input1)
    model1 = Convolution3D(3, 4, 4, 4, activation='relu')(r)
    model1 = MaxPooling3D()(model1)
    model1 = Flatten()(model1)

    # HCAL input
    input2 = Input(shape=(5, 5, 60))
    r = Reshape((5, 5, 60, 1))(input2)
    model2 = Convolution3D(10, 2, 2, 6, activation='relu')(r)
    #model2 = Convolution3D(10, 2, 2, 6, activation='relu')(r)
    model2 = MaxPooling3D()(model2)
    model2 = Flatten()(model2)

    # join the two input models
    bmodel = merge([model1, model2], mode='concat')  # branched model

    # fully connected ending
    bmodel = (Dense(1000, activation='relu'))(bmodel)
    bmodel = (Dropout(0.5))(bmodel)

    # oc = Dense(1,activation='sigmoid', name='particle_label')(bmodel) # output particle classification
    oe = Dense(1, activation='linear', name='energy')(bmodel)  # output energy regression

    # classification, will not use yet
    # bimodel = Model(input=[input1,input2], output=[oc,oe])
    # bimodel.compile(loss=['binary_crossentropy', 'mse'], optimizer='sgd')
    # bimodel.summary()

    # energy regression model
    model = Model(input=[input1, input2], output=oe)
    model.compile(loss=loss, optimizer='adam')
    model.summary()
    saveModel(model, name=name)
    return model
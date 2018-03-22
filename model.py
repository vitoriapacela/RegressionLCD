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
    :parameter name: name of the json file.
    :type name: str
    :parameter weights: whether or not to load the weights.
    :type weights: bool
    :return: loaded model.
    '''

    json_file = open('%s.json' % name, 'r')
    loaded = json_file.read()
    json_file.close()

    model = model_from_json(loaded)

    # load weights into new model
    if weights == True:
        model.load_weights('%s.h5' % name)
    # print(model.summary())

    print("Loaded model from disk")
    return model


def saveModel(model, name="regression"):
    '''
    Saves model as json file.
    Adapted from Kaustuv Datta and Jayesh Mahapatra's CaloImageMacros.
    :parameter model: model to be saved.
    :parameter name: name of the model to be saved.
    :type name: str
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
    :parameter hist: array of losses in the trained model.
    :parameter name: name of the file to be saved
    :type name: str
    '''
    loss = np.array(hist.history['loss'])
    vaLoss = np.array(hist.history['val_loss'])

    f = h5py.File("%s_losses.h5" % name, "w")
    f.create_dataset('loss', data=loss)
    f.create_dataset('val_loss', data=vaLoss)
    f.close()


def dnnModel(modName='dnn'):
    # ECAL input
    input1 = Input(shape=(51, 51, 25))
    model1 = Flatten()(input1)

    # HCAL input
    input2 = Input(shape=(11, 11, 60))
    model2 = Flatten()(input2)

    # Merging inputs
    bmodel = merge([model1, model2], mode='concat')

    bmodel = (Dense(1280, activation='relu'))(bmodel)

    bmodel = (Dropout(0.5))(bmodel)

    oe = Dense(1, activation='linear')(bmodel)  # output energy regression

    # energy regression model
    model = Model(input=[input1, input2], output=oe)
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    saveModel(model, name= modName)
    return model

def cnnModel(loss='mse', name="regression"):
    '''
    Regression model that has ECAL and HCAL as inputs.
    :parameter loss: Keras' loss function to be used. Recommended: mse, mean_absolute_error,
     mean_squared_logarithmic_error, mean_absolute_percentage_error.
    :type loss: str
    :parameter name: name to save the file as.
    :type name: str
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
    model2 = MaxPooling3D()(model2)
    model2 = Flatten()(model2)

    # join the two input models
    bmodel = merge([model1, model2], mode='concat')  # branched model

    # fully connected ending
    bmodel = (Dense(1000, activation='relu'))(bmodel)
    bmodel = (Dropout(0.5))(bmodel)

    # oc = Dense(1,activation='sigmoid', name='particle_label')(bmodel) # output particle classification
    oe = Dense(1, activation='linear', name='energy')(bmodel)  # output energy regression

    # energy regression model
    model = Model(input=[input1, input2], output=oe)
    model.compile(loss=loss, optimizer='adam')
    model.summary()
    saveModel(model, name=name)
    return model


def modelSum(loss='mse', name="regression"):
    '''
    Regression model that has as inputs ECAL, HCAL and a numpy array containing the sum over ECAL and HCAL.
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
    model2 = Convolution3D(10, 2, 2, 2, activation='relu')(r)
    model2 = MaxPooling3D()(model2)
    model2 = Flatten()(model2)

    # ECAL sum
    input3 = Input(shape=(1,))

    # HCAL sum
    input4 = Input(shape=(1,))

    # join the three input models
    bmodel = merge([model1, model2, input3, input4], mode='concat')

    # fully connected ending
    bmodel = (Dense(1000, activation='relu'))(bmodel)
    bmodel = (Dropout(0.5))(bmodel)

    oe = Dense(1, activation='linear', name='energy')(bmodel)  # output energy regression

    # energy regression model
    model = Model(input=[input1, input2, input3, input4], output=oe)
    model.compile(loss=loss, optimizer='adam')
    model.summary()
    saveModel(model, name=name)

    return model


def modelHCALSum(loss='mse', name="regression"):
    '''
    Regression model that has as inputs ECAL, HCAL and the sum over HCAL.
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
    model2 = Convolution3D(10, 2, 2, 2, activation='relu')(r)
    model2 = MaxPooling3D()(model2)
    model2 = Flatten()(model2)

    # HCAL sum
    model3 = Input(shape=(1,))
    # r = Reshape((1, 1))(input3)
    # model3 = Flatten()(r)

    # join the three input models
    bmodel = merge([model1, model2, model3], mode='concat')

    # fully connected ending
    bmodel = (Dense(1000, activation='relu'))(bmodel)
    bmodel = (Dropout(0.5))(bmodel)

    oe = Dense(1, activation='linear', name='energy')(bmodel)  # output energy regression

    # energy regression model
    model = Model(input=[input1, input2, input3], output=oe)
    model.compile(loss=loss, optimizer='adam')
    model.summary()
    saveModel(model, name=name)

    return model

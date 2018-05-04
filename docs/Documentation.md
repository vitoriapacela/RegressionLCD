# Documentation

Before using this library, you must have your data in the HDF5 data format split into train, validation, and test sets explicitally in separate directories.

The activity diagrams were created with [yUML](https://yuml.me/diagram/scruffy/activity/samples). The source code is found in `test_diag.txt` and `train_diag.txt`.

## DNN training

In the `train.py` script, the paths to training and validation directories must be given. For each of those, the number of samples in the directory is calculated by using the function `nSamples()` from the module `preprocessing.py`.
The respective data generators for each directory are created with `gen_from_data()` from the library `CMS_Deep_Learning`.

The DNN model is created with RegressionLCD's `dnnModel()` from the `model.py` module.

After that, trainig is performed with Keras' `fit_generator()`, taking as arguments the training set generator, the validation set generator, the number of samples in the training set, the number of samples in the validation set, and the number of epochs.
After each epoch, the model weights are updated in the path given if the validation loss in such epoch has decreased.

Following, the loss function value for both the training and validation sets are saved for each epoch with `model.saveLosses()`, and can then be visualized with `analysis.show_losses`.

The illustration is found in `train_diagram.jpg`.

## Model testing

In the `test.py` script, the model (`mod`) and its weights(`w`) must be loaded from the json and hdf5 files created during training.
The path to the test directory must also be created.

The test data is loaded with CMS_DeepLearning.io's `simple_grab()`, taking as parameters the model, its weights, and the test directory. It applies the model to the test data to generate the predictions. For such, it receives the input keys "ECAL" and "HCAL" to be retrieved from the data, as well as the ground truth value "energy". 

Finally, it returns the ground truth values and their respective predictions obtained from the model.

Such prediction outputs are numpy arrays that must be reshaped using `numpy.reshape()`.

Since it is slow to get the outputs of simple_grab, they are saved into an HDF5 file using `analysis.saveTruePred()` for easy and fast retrieval. 

Histogram plots are created with `analysis.histEDif()` and `analysis.histRelDif()`.

For an analysis of how the model performs depending on the energy of the particles, it is necessary to split the data into energy intervals. Given the number of bins in which the data is desired to be divided, each bin contains an energy interval of the same size, calculated based on the energy range of all the data. Therefore, `analysis.binning()` is executed taking as parameters the ground truth and prediction arrays, as well as the number of bins. The number of bins is 10 by default, since it was considered appropriate for an energy range between 10 and 500 GeV.

The function returns relevant data to be analysed as arrays, such as absolute and relative mean values, absolute and relative standard deviations, and array sizes.

These analysis are plotted with `analysis.plotN()` by giving the desired data as parameters.

The illustration is found in `train_diagram.jpg`.

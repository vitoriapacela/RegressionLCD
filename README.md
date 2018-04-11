# RegressionLCD [![Build Status](https://travis-ci.org/vitoriapacela/RegressionLCD.svg?branch=master)](https://travis-ci.org/vitoriapacela/RegressionLCD)


Repository for my SURF Caltech-CMS project on deep learning for imaging calorimetry in the LCD (Linear Collider Detector) at CERN.

This work was published in the Deep Learning in Particle Physics workshop at NIPS 2017. [Calorimetry with Deep Learning: Particle Classification, Energy Regression, and Simulation for High-Energy Physics](https://dl4physicalsciences.github.io/files/nips_dlps_2017_15.pdf).

### Usage
This repository contains data pre-processing functions (`preprocessing.py`), model topologies (`model.py`) and post-processing functions (`analysis.py`) to analyze model performance.

You can find example notebooks in `examples`. Older notebooks (not updated) can be found in the [NotebooksLCD](https://github.com/vitoriapacela/NotebooksLCD) repository.

To submit a training job to a gpu, modify `train.py` adapting it to your data.

To get predictions of the test set, use `test.py`. 

### Environment
After cloning this repository and entering it, execute `pip install . --user`.
Notice that there are inconsistencies with how `tensorflow` should be installed.

Pre-processing dependencies:
Danny Weitecamp's [CMS_Deep_Learning package](https://github.com/DannyWeitekamp/CMS_Deep_Learning) for the data generator.

Machine Learning dependencies:
[Keras](https://keras.io/) with [Tensorflow](https://www.tensorflow.org/) in the backend.

Post-processing dependencies:
[Matplotlib](http://matplotlib.org/), [Scipy](https://www.scipy.org/).

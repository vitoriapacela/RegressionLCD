# RegressionLCD [![Build Status](https://travis-ci.org/vitoriapacela/RegressionLCD.svg?branch=master)](https://travis-ci.org/vitoriapacela/RegressionLCD)


Repository for my SURF Caltech-CMS project on deep learning for imaging calorimetry in the LCD (Linear Collider Detector) at CERN.

This work was published in the Deep Learning in Particle Physics ([DLPS](https://dl4physicalsciences.github.io/)) workshop at [NIPS 2017](https://nips.cc/Conferences/2017). [Calorimetry with Deep Learning: Particle Classification, Energy Regression, and Simulation for High-Energy Physics](https://dl4physicalsciences.github.io/files/nips_dlps_2017_15.pdf).

B. Hooberman, V. Barin Pacela, M. Zhang, W. Wei, G. Khattak, S. Vallecorsa, A. Farbin, J-R. Vlimant, F. Carminati, M. Spiropulu, M. Pierini. [**Calorimetry with Deep Learning: Particle Classification, Energy Regression, and Simulation for High-Energy Physics**](https://dl4physicalsciences.github.io/files/nips_dlps_2017_15.pdf). DLPS 2017, NIPS 2017, Long Beach, CA, USA.

### Usage
This repository contains data pre-processing functions (`preprocessing.py`), model topologies (`model.py`) and post-processing functions (`analysis.py`) to analyze model performance.

You can find example notebooks in `examples`. Older notebooks (not updated) can be found in the [NotebooksLCD](https://github.com/vitoriapacela/NotebooksLCD) repository.

To submit a training job to a gpu, modify `train.py` adapting it to your data.

To get predictions of the test set, use `test.py`. 

### Environment
After cloning this repository and entering it, execute `pip install . --user`.
Notice that there are inconsistencies with how `tensorflow` should be installed.

Code in Python 2.7.

Pre-processing dependencies:
Danny Weitecamp's [CMS_Deep_Learning package](https://github.com/DannyWeitekamp/CMS_Deep_Learning) for the data generator.

Machine Learning dependencies:
[Keras 1.2.2](https://keras.io/) with [Tensorflow 0.12.0](https://www.tensorflow.org/) in the backend.

Post-processing dependencies:
[Matplotlib 1.4.3](http://matplotlib.org/), [Scipy 1.0.0](https://www.scipy.org/).

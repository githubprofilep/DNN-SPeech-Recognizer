#Voice User Interfaces
#Project: Speech Recognition with Neural Networks
In this notebook, some template code has already been provided for you, and you will need to implement additional functionality to successfully complete this project. You will not need to modify the included code beyond what is requested. Sections that begin with '(IMPLEMENTATION)' in the header indicate that the following blocks of code will require additional functionality which you must provide. Please be sure to read the instructions carefully!

Note: Once you have completed all of the code implementations, you need to finalize your work by exporting the Jupyter Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to \n", "File -> Download as -> HTML (.html). Include the finished document along with this notebook as your submission.

In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a 'Question X' header. Carefully read each question and provide thorough answers in the following text boxes that begin with 'Answer:'. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.

Note: Code and Markdown cells can be executed using the Shift + Enter keyboard shortcut. Markdown cells can be edited by double-clicking the cell to enter edit mode.

The rubric contains optional "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. If you decide to pursue the "Stand Out Suggestions", you should include the code in this Jupyter notebook.

Introduction
In this notebook, you will build a deep neural network that functions as part of an end-to-end automatic speech recognition (ASR) pipeline! Your completed pipeline will accept raw audio as input and return a predicted transcription of the spoken language. The full pipeline is summarized in the figure below.


STEP 1 is a pre-processing step that converts raw audio to one of two feature representations that are commonly used for ASR.
STEP 2 is an acoustic model which accepts audio features as input and returns a probability distribution over all potential transcriptions. After learning about the basic types of neural networks that are often used for acoustic modeling, you will engage in your own investigations, to design your own acoustic model!
STEP 3 in the pipeline takes the output from the acoustic model and returns a predicted transcription.
Feel free to use the links below to navigate the notebook:

The Data
STEP 1: Acoustic Features for Speech Recognition
STEP 2: Deep Neural Networks for Acoustic Modeling
Model 0: RNN
Model 1: RNN + TimeDistributed Dense
Model 2: CNN + RNN + TimeDistributed Dense
Model 3: Deeper RNN + TimeDistributed Dense
Model 4: Bidirectional RNN + TimeDistributed Dense
Models 5+
Compare the Models
Final Model
STEP 3: Obtain Predictions

The Data
We begin by investigating the dataset that will be used to train and evaluate your pipeline. LibriSpeech is a large corpus of English-read speech, designed for training and evaluating models for ASR. The dataset contains 1000 hours of speech derived from audiobooks. We will work with a small subset in this project, since larger-scale data would take a long while to train. However, after completing this project, if you are interested in exploring further, you are encouraged to work with more of the data that is provided online.

In the code cells below, you will use the vis_train_features module to visualize a training example. The supplied argument index=0 tells the module to extract the first example in the training set. (You are welcome to change index=0 to point to a different training example, if you like, but please DO NOT amend any other code in the cell.) The returned variables are:

vis_text - transcribed text (label) for the training example.
vis_raw_audio - raw audio waveform for the training example.
vis_mfcc_feature - mel-frequency cepstral coefficients (MFCCs) for the training example.
vis_spectrogram_feature - spectrogram for the training example.
vis_audio_path - the file path to the training example.
from data_generator import vis_train_features
​
# extract label and audio features for a single training example
vis_text, vis_raw_audio, vis_mfcc_feature, vis_spectrogram_feature, vis_audio_path = vis_train_features()
There are 2023 total training examples.
The following code cell visualizes the audio waveform for your chosen example, along with the corresponding transcript. You also have the option to play the audio in the notebook!

from IPython.display import Markdown, display
from data_generator import vis_train_features, plot_raw_audio
from IPython.display import Audio
%matplotlib inline
​
# plot audio signal
plot_raw_audio(vis_raw_audio)
# print length of audio signal
display(Markdown('**Shape of Audio Signal** : ' + str(vis_raw_audio.shape)))
# print transcript corresponding to audio clip
display(Markdown('**Transcript** : ' + str(vis_text)))
# play the audio file
Audio(vis_audio_path)

<IPython.core.display.Markdown object>
<IPython.core.display.Markdown object>

STEP 1: Acoustic Features for Speech Recognition
For this project, you won't use the raw audio waveform as input to your model. Instead, we provide code that first performs a pre-processing step to convert the raw audio to a feature representation that has historically proven successful for ASR models. Your acoustic model will accept the feature representation as input.

In this project, you will explore two possible feature representations. After completing the project, if you'd like to read more about deep learning architectures that can accept raw audio input, you are encouraged to explore this research paper.

Spectrograms
The first option for an audio feature representation is the spectrogram. In order to complete this project, you will not need to dig deeply into the details of how a spectrogram is calculated; but, if you are curious, the code for calculating the spectrogram was borrowed from this repository. The implementation appears in the utils.py file in your repository.

The code that we give you returns the spectrogram as a 2D tensor, where the first (vertical) dimension indexes time, and the second (horizontal) dimension indexes frequency. To speed the convergence of your algorithm, we have also normalized the spectrogram. (You can see this quickly in the visualization below by noting that the mean value hovers around zero, and most entries in the tensor assume values close to zero.)

from data_generator import plot_spectrogram_feature
​
# plot normalized spectrogram
plot_spectrogram_feature(vis_spectrogram_feature)
# print shape of spectrogram
display(Markdown('**Shape of Spectrogram** : ' + str(vis_spectrogram_feature.shape)))

<IPython.core.display.Markdown object>
Mel-Frequency Cepstral Coefficients (MFCCs)
The second option for an audio feature representation is MFCCs. You do not need to dig deeply into the details of how MFCCs are calculated, but if you would like more information, you are welcome to peruse the documentation of the python_speech_features Python package. Just as with the spectrogram features, the MFCCs are normalized in the supplied code.

The main idea behind MFCC features is the same as spectrogram features: at each time window, the MFCC feature yields a feature vector that characterizes the sound within the window. Note that the MFCC feature is much lower-dimensional than the spectrogram feature, which could help an acoustic model to avoid overfitting to the training dataset.

from data_generator import plot_mfcc_feature
​
# plot normalized MFCC
plot_mfcc_feature(vis_mfcc_feature)
# print shape of MFCC
display(Markdown('**Shape of MFCC** : ' + str(vis_mfcc_feature.shape)))

<IPython.core.display.Markdown object>
When you construct your pipeline, you will be able to choose to use either spectrogram or MFCC features. If you would like to see different implementations that make use of MFCCs and/or spectrograms, please check out the links below:

This repository uses spectrograms.
This repository uses MFCCs.
This repository also uses MFCCs.
This repository experiments with raw audio, spectrograms, and MFCCs as features.

STEP 2: Deep Neural Networks for Acoustic Modeling
In this section, you will experiment with various neural network architectures for acoustic modeling.

You will begin by training five relatively simple architectures. Model 0 is provided for you. You will write code to implement Models 1, 2*, *3, and 4*. If you would like to experiment further, you are welcome to create and train more models under the *Models 5+ heading.

All models will be specified in the sample_models.py file. After importing the sample_models module, you will train your architectures in the notebook.

After experimenting with the five simple architectures, you will have the opportunity to compare their performance. Based on your findings, you will construct a deeper architecture that is designed to outperform all of the shallow models.

For your convenience, we have designed the notebook so that each model can be specified and trained on separate occasions. That is, say you decide to take a break from the notebook after training Model 1. Then, you need not re-execute all prior code cells in the notebook before training Model 2. You need only re-execute the code cell below, that is marked with RUN THIS CODE CELL IF YOU ARE RESUMING THE NOTEBOOK AFTER A BREAK, before transitioning to the code cells corresponding to Model 2.

#####################################################################
# RUN THIS CODE CELL IF YOU ARE RESUMING THE NOTEBOOK AFTER A BREAK #
#####################################################################
​
# allocate 50% of GPU memory (if you like, feel free to change this)
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf 
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))
​
# watch for any changes in the sample_models module, and reload it automatically
%load_ext autoreload
%autoreload 2
# import NN architectures for speech recognition
from sample_models import *
# import function for training acoustic model
from train_utils import train_model
Using TensorFlow backend.

Model 0: RNN
Given their effectiveness in modeling sequential data, the first acoustic model you will use is an RNN. As shown in the figure below, the RNN we supply to you will take the time sequence of audio features as input.


At each time step, the speaker pronounces one of 28 possible characters, including each of the 26 letters in the English alphabet, along with a space character (" "), and an apostrophe (').

The output of the RNN at each time step is a vector of probabilities with 29 entries, where the 𝑖-th entry encodes the probability that the 𝑖-th character is spoken in the time sequence. (The extra 29th character is an empty "character" used to pad training examples within batches containing uneven lengths.) If you would like to peek under the hood at how characters are mapped to indices in the probability vector, look at the char_map.py file in the repository. The figure below shows an equivalent, rolled depiction of the RNN that shows the output layer in greater detail.


The model has already been specified for you in Keras. To import it, you need only run the code cell below.

model_0 = simple_rnn_model(input_dim=161) # change to 13 if you would like to use MFCC features
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
the_input (InputLayer)       (None, None, 161)         0         
_________________________________________________________________
rnn (GRU)                    (None, None, 29)          16617     
_________________________________________________________________
softmax (Activation)         (None, None, 29)          0         
=================================================================
Total params: 16,617
Trainable params: 16,617
Non-trainable params: 0
_________________________________________________________________
None
As explored in the lesson, you will train the acoustic model with the CTC loss criterion. Custom loss functions take a bit of hacking in Keras, and so we have implemented the CTC loss function for you, so that you can focus on trying out as many deep learning architectures as possible :). If you'd like to peek at the implementation details, look at the add_ctc_loss function within the train_utils.py file in the repository.

To train your architecture, you will use the train_model function within the train_utils module; it has already been imported in one of the above code cells. The train_model function takes three required arguments:

input_to_softmax - a Keras model instance.
pickle_path - the name of the pickle file where the loss history will be saved.
save_model_path - the name of the HDF5 file where the model will be saved.
If we have already supplied values for input_to_softmax, pickle_path, and save_model_path, please DO NOT modify these values.

There are several optional arguments that allow you to have more control over the training process. You are welcome to, but not required to, supply your own values for these arguments.

minibatch_size - the size of the minibatches that are generated while training the model (default: 20).
spectrogram - Boolean value dictating whether spectrogram (True) or MFCC (False) features are used for training (default: True).
mfcc_dim - the size of the feature dimension to use when generating MFCC features (default: 13).
optimizer - the Keras optimizer used to train the model (default: SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)).
epochs - the number of epochs to use to train the model (default: 20). If you choose to modify this parameter, make sure that it is at least 20.
verbose - controls the verbosity of the training output in the model.fit_generator method (default: 1).
sort_by_duration - Boolean value dictating whether the training and validation sets are sorted by (increasing) duration before the start of the first epoch (default: False).
The train_model function defaults to using spectrogram features; if you choose to use these features, note that the acoustic model in simple_rnn_model should have input_dim=161. Otherwise, if you choose to use MFCC features, the acoustic model should have input_dim=13.

We have chosen to use GRU units in the supplied RNN. If you would like to experiment with LSTM or SimpleRNN cells, feel free to do so here. If you change the GRU units to SimpleRNN cells in simple_rnn_model, you may notice that the loss quickly becomes undefined (nan) - you are strongly encouraged to check this for yourself! This is due to the exploding gradients problem. We have already implemented gradient clipping in your optimizer to help you avoid this issue.

IMPORTANT NOTE: If you notice that your gradient has exploded in any of the models below, feel free to explore more with gradient clipping (the clipnorm argument in your optimizer) or swap out any SimpleRNN cells for LSTM or GRU cells. You can also try restarting the kernel to restart the training process.

train_model(input_to_softmax=model_0, 
            pickle_path='model_0.pickle', 
            save_model_path='model_0.h5',
            spectrogram=True) # change to False if you would like to use MFCC features
Epoch 1/20
101/101 [==============================] - 201s 2s/step - loss: 849.8264 - val_loss: 757.5640
Epoch 2/20
101/101 [==============================] - 200s 2s/step - loss: 779.7249 - val_loss: 765.7749
Epoch 3/20
101/101 [==============================] - 197s 2s/step - loss: 779.4234 - val_loss: 749.3697
Epoch 4/20
101/101 [==============================] - 198s 2s/step - loss: 778.0687 - val_loss: 754.8384
Epoch 5/20
101/101 [==============================] - 198s 2s/step - loss: 778.5715 - val_loss: 756.8771
Epoch 6/20
101/101 [==============================] - 198s 2s/step - loss: 777.9909 - val_loss: 757.9533
Epoch 7/20
101/101 [==============================] - 196s 2s/step - loss: 777.5092 - val_loss: 747.1889
Epoch 8/20
101/101 [==============================] - 196s 2s/step - loss: 777.8546 - val_loss: 763.3541
Epoch 9/20
101/101 [==============================] - 198s 2s/step - loss: 777.8735 - val_loss: 755.6761
Epoch 10/20
101/101 [==============================] - 198s 2s/step - loss: 777.8627 - val_loss: 757.3696
Epoch 11/20
101/101 [==============================] - 196s 2s/step - loss: 777.8858 - val_loss: 755.0960
Epoch 12/20
101/101 [==============================] - 196s 2s/step - loss: 777.9197 - val_loss: 756.5354
Epoch 13/20
101/101 [==============================] - 197s 2s/step - loss: 777.7165 - val_loss: 754.7956
Epoch 14/20
101/101 [==============================] - 196s 2s/step - loss: 777.6205 - val_loss: 759.9967
Epoch 15/20
101/101 [==============================] - 198s 2s/step - loss: 777.8210 - val_loss: 749.3608
Epoch 16/20
101/101 [==============================] - 196s 2s/step - loss: 777.2953 - val_loss: 761.7993
Epoch 17/20
101/101 [==============================] - 196s 2s/step - loss: 777.8510 - val_loss: 757.8568
Epoch 18/20
101/101 [==============================] - 197s 2s/step - loss: 777.5470 - val_loss: 755.6718
Epoch 19/20
101/101 [==============================] - 195s 2s/step - loss: 777.5611 - val_loss: 754.6449
Epoch 20/20
101/101 [==============================] - 196s 2s/step - loss: 777.6080 - val_loss: 756.7471

(IMPLEMENTATION) Model 1: RNN + TimeDistributed Dense
Read about the TimeDistributed wrapper and the BatchNormalization layer in the Keras documentation. For your next architecture, you will add batch normalization to the recurrent layer to reduce training times. The TimeDistributed layer will be used to find more complex patterns in the dataset. The unrolled snapshot of the architecture is depicted below.


The next figure shows an equivalent, rolled depiction of the RNN that shows the (TimeDistrbuted) dense and output layers in greater detail.


Use your research to complete the rnn_model function within the sample_models.py file. The function should specify an architecture that satisfies the following requirements:

The first layer of the neural network should be an RNN (SimpleRNN, LSTM, or GRU) that takes the time sequence of audio features as input. We have added GRU units for you, but feel free to change GRU to SimpleRNN or LSTM, if you like!
Whereas the architecture in simple_rnn_model treated the RNN output as the final layer of the model, you will use the output of your RNN as a hidden layer. Use TimeDistributed to apply a Dense layer to each of the time steps in the RNN output. Ensure that each Dense layer has output_dim units.
Use the code cell below to load your model into the model_1 variable. Use a value for input_dim that matches your chosen audio features, and feel free to change the values for units and activation to tweak the behavior of your recurrent layer.

model_1 = rnn_model(input_dim=161, # change to 13 if you would like to use MFCC features
                    units=200,
                    activation='relu')
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
the_input (InputLayer)       (None, None, 161)         0         
_________________________________________________________________
rnn (GRU)                    (None, None, 200)         217200    
_________________________________________________________________
batch_normalization_1 (Batch (None, None, 200)         800       
_________________________________________________________________
time_distributed_1 (TimeDist (None, None, 29)          5829      
_________________________________________________________________
softmax (Activation)         (None, None, 29)          0         
=================================================================
Total params: 223,829
Trainable params: 223,429
Non-trainable params: 400
_________________________________________________________________
None
Please execute the code cell below to train the neural network you specified in input_to_softmax. After the model has finished training, the model is saved in the HDF5 file model_1.h5. The loss history is saved in model_1.pickle. You are welcome to tweak any of the optional parameters while calling the train_model function, but this is not required.

train_model(input_to_softmax=model_1, 
            pickle_path='model_1.pickle', 
            save_model_path='model_1.h5',
            spectrogram=True) # change to False if you would like to use MFCC features
Epoch 1/20
101/101 [==============================] - 199s 2s/step - loss: 294.2625 - val_loss: 259.8630
Epoch 2/20
101/101 [==============================] - 199s 2s/step - loss: 211.3662 - val_loss: 202.7216
Epoch 3/20
101/101 [==============================] - 195s 2s/step - loss: 186.1604 - val_loss: 177.0332
Epoch 4/20
101/101 [==============================] - 196s 2s/step - loss: 169.5708 - val_loss: 165.8714
Epoch 5/20
101/101 [==============================] - 196s 2s/step - loss: 159.0815 - val_loss: 157.3539
Epoch 6/20
101/101 [==============================] - 194s 2s/step - loss: 151.6395 - val_loss: 156.4733
Epoch 7/20
101/101 [==============================] - 194s 2s/step - loss: 146.0513 - val_loss: 148.5378
Epoch 8/20
101/101 [==============================] - 195s 2s/step - loss: 141.9663 - val_loss: 147.5847
Epoch 9/20
101/101 [==============================] - 194s 2s/step - loss: 137.4338 - val_loss: 145.6031
Epoch 10/20
101/101 [==============================] - 195s 2s/step - loss: 134.1823 - val_loss: 145.9991
Epoch 11/20
101/101 [==============================] - 193s 2s/step - loss: 131.1519 - val_loss: 142.8736
Epoch 12/20
101/101 [==============================] - 194s 2s/step - loss: 128.7438 - val_loss: 143.3977
Epoch 13/20
101/101 [==============================] - 193s 2s/step - loss: 126.7920 - val_loss: 139.0594
Epoch 14/20
101/101 [==============================] - 192s 2s/step - loss: 124.1532 - val_loss: 139.0521
Epoch 15/20
101/101 [==============================] - 194s 2s/step - loss: 122.3026 - val_loss: 141.8338
Epoch 16/20
101/101 [==============================] - 193s 2s/step - loss: 120.8502 - val_loss: 136.9616
Epoch 17/20
101/101 [==============================] - 193s 2s/step - loss: 119.0904 - val_loss: 138.9010
Epoch 18/20
101/101 [==============================] - 194s 2s/step - loss: 117.7079 - val_loss: 137.1840
Epoch 19/20
101/101 [==============================] - 195s 2s/step - loss: 117.3683 - val_loss: 138.4030
Epoch 20/20
101/101 [==============================] - 195s 2s/step - loss: 116.1696 - val_loss: 137.3820

(IMPLEMENTATION) Model 2: CNN + RNN + TimeDistributed Dense
The architecture in cnn_rnn_model adds an additional level of complexity, by introducing a 1D convolution layer.


This layer incorporates many arguments that can be (optionally) tuned when calling the cnn_rnn_model module. We provide sample starting parameters, which you might find useful if you choose to use spectrogram audio features.

If you instead want to use MFCC features, these arguments will have to be tuned. Note that the current architecture only supports values of 'same' or 'valid' for the conv_border_mode argument.

When tuning the parameters, be careful not to choose settings that make the convolutional layer overly small. If the temporal length of the CNN layer is shorter than the length of the transcribed text label, your code will throw an error.

Before running the code cell below, you must modify the cnn_rnn_model function in sample_models.py. Please add batch normalization to the recurrent layer, and provide the same TimeDistributed layer as before.

model_2 = cnn_rnn_model(input_dim=161, # change to 13 if you would like to use MFCC features
                        filters=200,
                        kernel_size=11, 
                        conv_stride=2,
                        conv_border_mode='valid',
                        units=200)
/opt/conda/lib/python3.6/site-packages/keras/layers/recurrent.py:1004: UserWarning: The `implementation` argument in `SimpleRNN` has been deprecated. Please remove it from your layer call.
  warnings.warn('The `implementation` argument '
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
the_input (InputLayer)       (None, None, 161)         0         
_________________________________________________________________
conv1d (Conv1D)              (None, None, 200)         354400    
_________________________________________________________________
bn_conv_1d (BatchNormalizati (None, None, 200)         800       
_________________________________________________________________
rnn (SimpleRNN)              (None, None, 200)         80200     
_________________________________________________________________
bn_rnn (BatchNormalization)  (None, None, 200)         800       
_________________________________________________________________
time_distributed_2 (TimeDist (None, None, 29)          5829      
_________________________________________________________________
softmax (Activation)         (None, None, 29)          0         
=================================================================
Total params: 442,029
Trainable params: 441,229
Non-trainable params: 800
_________________________________________________________________
None
Please execute the code cell below to train the neural network you specified in input_to_softmax. After the model has finished training, the model is saved in the HDF5 file model_2.h5. The loss history is saved in model_2.pickle. You are welcome to tweak any of the optional parameters while calling the train_model function, but this is not required.

train_model(input_to_softmax=model_2, 
            pickle_path='model_2.pickle', 
            save_model_path='model_2.h5', 
            spectrogram=True) # change to False if you would like to use MFCC features
Epoch 1/20
101/101 [==============================] - 54s 533ms/step - loss: 243.4286 - val_loss: 215.0856
Epoch 2/20
101/101 [==============================] - 50s 495ms/step - loss: 183.6578 - val_loss: 173.2059
Epoch 3/20
101/101 [==============================] - 50s 493ms/step - loss: 158.7477 - val_loss: 156.7101
Epoch 4/20
101/101 [==============================] - 50s 493ms/step - loss: 145.7081 - val_loss: 148.2067
Epoch 5/20
101/101 [==============================] - 50s 496ms/step - loss: 137.0719 - val_loss: 141.8838
Epoch 6/20
101/101 [==============================] - 50s 492ms/step - loss: 130.0676 - val_loss: 139.0528
Epoch 7/20
101/101 [==============================] - 50s 496ms/step - loss: 124.1599 - val_loss: 140.8763
Epoch 8/20
101/101 [==============================] - 50s 498ms/step - loss: 119.9829 - val_loss: 139.1788
Epoch 9/20
101/101 [==============================] - 50s 495ms/step - loss: 115.5313 - val_loss: 137.7037
Epoch 10/20
101/101 [==============================] - 50s 496ms/step - loss: 111.7240 - val_loss: 136.3480
Epoch 11/20
101/101 [==============================] - 50s 490ms/step - loss: 108.1988 - val_loss: 135.3801
Epoch 12/20
101/101 [==============================] - 50s 493ms/step - loss: 105.3121 - val_loss: 136.2248
Epoch 13/20
101/101 [==============================] - 50s 491ms/step - loss: 102.0426 - val_loss: 136.4973
Epoch 14/20
101/101 [==============================] - 50s 494ms/step - loss: 99.1487 - val_loss: 139.7910
Epoch 15/20
101/101 [==============================] - 50s 495ms/step - loss: 96.5352 - val_loss: 137.4534
Epoch 16/20
101/101 [==============================] - 50s 493ms/step - loss: 94.0480 - val_loss: 140.8339
Epoch 17/20
101/101 [==============================] - 50s 492ms/step - loss: 91.5886 - val_loss: 140.5733
Epoch 18/20
101/101 [==============================] - 50s 494ms/step - loss: 88.8616 - val_loss: 140.6289
Epoch 19/20
101/101 [==============================] - 50s 497ms/step - loss: 86.4834 - val_loss: 143.1618
Epoch 20/20
101/101 [==============================] - 50s 499ms/step - loss: 84.4901 - val_loss: 146.0076

(IMPLEMENTATION) Model 3: Deeper RNN + TimeDistributed Dense
Review the code in rnn_model, which makes use of a single recurrent layer. Now, specify an architecture in deep_rnn_model that utilizes a variable number recur_layers of recurrent layers. The figure below shows the architecture that should be returned if recur_layers=2. In the figure, the output sequence of the first recurrent layer is used as input for the next recurrent layer.


Feel free to change the supplied values of units to whatever you think performs best. You can change the value of recur_layers, as long as your final value is greater than 1. (As a quick check that you have implemented the additional functionality in deep_rnn_model correctly, make sure that the architecture that you specify here is identical to rnn_model if recur_layers=1.)

model_3 = deep_rnn_model(input_dim=161, # change to 13 if you would like to use MFCC features
                         units=200,
                         recur_layers=2) 
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
the_input (InputLayer)       (None, None, 161)         0         
_________________________________________________________________
rnn_1 (GRU)                  (None, None, 200)         217200    
_________________________________________________________________
rnn_0 (GRU)                  (None, None, 200)         240600    
_________________________________________________________________
bn_0 (BatchNormalization)    (None, None, 200)         800       
_________________________________________________________________
time_distributed_1 (TimeDist (None, None, 29)          5829      
_________________________________________________________________
softmax (Activation)         (None, None, 29)          0         
=================================================================
Total params: 464,429
Trainable params: 464,029
Non-trainable params: 400
_________________________________________________________________
None
Please execute the code cell below to train the neural network you specified in input_to_softmax. After the model has finished training, the model is saved in the HDF5 file model_3.h5. The loss history is saved in model_3.pickle. You are welcome to tweak any of the optional parameters while calling the train_model function, but this is not required.

train_model(input_to_softmax=model_3, 
            pickle_path='model_3.pickle', 
            save_model_path='model_3.h5', 
            spectrogram=True) # change to False if you would like to use MFCC features
Epoch 1/20
101/101 [==============================] - 351s 3s/step - loss: 298.0278 - val_loss: 265.3843
Epoch 2/20
101/101 [==============================] - 357s 4s/step - loss: 202.3947 - val_loss: 191.6398
Epoch 3/20
101/101 [==============================] - 357s 4s/step - loss: 173.0987 - val_loss: 172.1038
Epoch 4/20
101/101 [==============================] - 358s 4s/step - loss: 157.8814 - val_loss: 157.7426
Epoch 5/20
101/101 [==============================] - 354s 4s/step - loss: 148.2932 - val_loss: 151.8765
Epoch 6/20
101/101 [==============================] - 356s 4s/step - loss: 139.9178 - val_loss: 148.2812
Epoch 7/20
101/101 [==============================] - 366s 4s/step - loss: 133.7577 - val_loss: 144.5698
Epoch 8/20
101/101 [==============================] - 357s 4s/step - loss: 128.5204 - val_loss: 141.4944
Epoch 9/20
101/101 [==============================] - 357s 4s/step - loss: 124.1663 - val_loss: 137.1718
Epoch 10/20
101/101 [==============================] - 356s 4s/step - loss: 119.8381 - val_loss: 138.2943
Epoch 11/20
101/101 [==============================] - 357s 4s/step - loss: 116.5912 - val_loss: 136.1385
Epoch 12/20
101/101 [==============================] - 358s 4s/step - loss: 114.4788 - val_loss: 135.0064
Epoch 13/20
101/101 [==============================] - 356s 4s/step - loss: 111.5543 - val_loss: 130.0465
Epoch 14/20
101/101 [==============================] - 355s 4s/step - loss: 109.1810 - val_loss: 134.0160
Epoch 15/20
101/101 [==============================] - 358s 4s/step - loss: 106.3814 - val_loss: 128.0234
Epoch 16/20
101/101 [==============================] - 357s 4s/step - loss: 104.6540 - val_loss: 128.3594
Epoch 17/20
101/101 [==============================] - 358s 4s/step - loss: 102.2506 - val_loss: 128.7814
Epoch 18/20
101/101 [==============================] - 357s 4s/step - loss: 101.7930 - val_loss: 132.6089
Epoch 19/20
101/101 [==============================] - 356s 4s/step - loss: 99.3755 - val_loss: 129.4195
Epoch 20/20
101/101 [==============================] - 357s 4s/step - loss: 97.4058 - val_loss: 132.1125

(IMPLEMENTATION) Model 4: Bidirectional RNN + TimeDistributed Dense
Read about the Bidirectional wrapper in the Keras documentation. For your next architecture, you will specify an architecture that uses a single bidirectional RNN layer, before a (TimeDistributed) dense layer. The added value of a bidirectional RNN is described well in this paper.

One shortcoming of conventional RNNs is that they are only able to make use of previous context. In speech recognition, where whole utterances are transcribed at once, there is no reason not to exploit future context as well. Bidirectional RNNs (BRNNs) do this by processing the data in both directions with two separate hidden layers which are then fed forwards to the same output layer.


Before running the code cell below, you must complete the bidirectional_rnn_model function in sample_models.py. Feel free to use SimpleRNN, LSTM, or GRU units. When specifying the Bidirectional wrapper, use merge_mode='concat'.

model_4 = bidirectional_rnn_model(input_dim=161, # change to 13 if you would like to use MFCC features
                                  units=200)
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
the_input (InputLayer)       (None, None, 161)         0         
_________________________________________________________________
bidirectional_1 (Bidirection (None, None, 400)         434400    
_________________________________________________________________
time_distributed_2 (TimeDist (None, None, 29)          11629     
_________________________________________________________________
softmax (Activation)         (None, None, 29)          0         
=================================================================
Total params: 446,029
Trainable params: 446,029
Non-trainable params: 0
_________________________________________________________________
None
Please execute the code cell below to train the neural network you specified in input_to_softmax. After the model has finished training, the model is saved in the HDF5 file model_4.h5. The loss history is saved in model_4.pickle. You are welcome to tweak any of the optional parameters while calling the train_model function, but this is not required.

train_model(input_to_softmax=model_4, 
            pickle_path='model_4.pickle', 
            save_model_path='model_4.h5', 
            spectrogram=True) # change to False if you would like to use MFCC features
Epoch 1/20
101/101 [==============================] - 354s 4s/step - loss: 374.5360 - val_loss: 334.5457
Epoch 2/20
101/101 [==============================] - 359s 4s/step - loss: 322.4271 - val_loss: 305.9504
Epoch 3/20
101/101 [==============================] - 360s 4s/step - loss: 312.6363 - val_loss: 309.4455
Epoch 4/20
101/101 [==============================] - 359s 4s/step - loss: 310.2969 - val_loss: 314.4826
Epoch 5/20
101/101 [==============================] - 358s 4s/step - loss: 304.2302 - val_loss: 250.2658
Epoch 6/20
101/101 [==============================] - 358s 4s/step - loss: 247.4622 - val_loss: 219.7860
Epoch 7/20
101/101 [==============================] - 357s 4s/step - loss: 219.5885 - val_loss: 205.1456
Epoch 8/20
101/101 [==============================] - 358s 4s/step - loss: 206.9923 - val_loss: 194.6797
Epoch 9/20
101/101 [==============================] - 361s 4s/step - loss: 196.8176 - val_loss: 191.9813
Epoch 10/20
101/101 [==============================] - 359s 4s/step - loss: 187.6223 - val_loss: 180.4179
Epoch 11/20
101/101 [==============================] - 360s 4s/step - loss: 179.6301 - val_loss: 175.9601
Epoch 12/20
101/101 [==============================] - 359s 4s/step - loss: 172.2103 - val_loss: 173.3003
Epoch 13/20
101/101 [==============================] - 359s 4s/step - loss: 165.5943 - val_loss: 167.1611
Epoch 14/20
101/101 [==============================] - 361s 4s/step - loss: 159.4949 - val_loss: 166.3472
Epoch 15/20
101/101 [==============================] - 361s 4s/step - loss: 154.2367 - val_loss: 160.2820
Epoch 16/20
101/101 [==============================] - 360s 4s/step - loss: 149.1224 - val_loss: 158.4822
Epoch 17/20
101/101 [==============================] - 358s 4s/step - loss: 144.6964 - val_loss: 155.2544
Epoch 18/20
101/101 [==============================] - 358s 4s/step - loss: 140.4290 - val_loss: 153.7408
Epoch 19/20
101/101 [==============================] - 358s 4s/step - loss: 136.7008 - val_loss: 151.6428
Epoch 20/20
101/101 [==============================] - 360s 4s/step - loss: 132.6509 - val_loss: 150.0365

(OPTIONAL IMPLEMENTATION) Models 5+
If you would like to try out more architectures than the ones above, please use the code cell below. Please continue to follow the same convention for saving the models; for the 𝑖-th sample model, please save the loss at model_i.pickle and saving the trained model at model_i.h5.

## (Optional) TODO: Try out some more models!
### Feel free to use as many code cells as needed.

Compare the Models
Execute the code cell below to evaluate the performance of the drafted deep learning models. The training and validation loss are plotted for each model.

from glob import glob
import numpy as np
import _pickle as pickle
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set_style(style='white')
​
# obtain the paths for the saved model history
all_pickles = sorted(glob("results/*.pickle"))
# extract the name of each model
model_names = [item[8:-7] for item in all_pickles]
# extract the loss history for each model
valid_loss = [pickle.load( open( i, "rb" ) )['val_loss'] for i in all_pickles]
train_loss = [pickle.load( open( i, "rb" ) )['loss'] for i in all_pickles]
# save the number of epochs used to train each model
num_epochs = [len(valid_loss[i]) for i in range(len(valid_loss))]
​
fig = plt.figure(figsize=(16,5))
​
# plot the training loss vs. epoch for each model
ax1 = fig.add_subplot(121)
for i in range(len(all_pickles)):
    ax1.plot(np.linspace(1, num_epochs[i], num_epochs[i]), 
            train_loss[i], label=model_names[i])
# clean up the plot
ax1.legend()  
ax1.set_xlim([1, max(num_epochs)])
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
​
# plot the validation loss vs. epoch for each model
ax2 = fig.add_subplot(122)
for i in range(len(all_pickles)):
    ax2.plot(np.linspace(1, num_epochs[i], num_epochs[i]), 
            valid_loss[i], label=model_names[i])
# clean up the plot
ax2.legend()  
ax2.set_xlim([1, max(num_epochs)])
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.show()

Question 1: Use the plot above to analyze the performance of each of the attempted architectures. Which performs best? Provide an explanation regarding why you think some models perform better than others.

Answer:

Model 0 RNN: We can see an overfitting during training. That means our training is bad. Our neural network can not learn. It is because this architecture is very simple and there are not enough parameters (only 16,617) to find patterns in that data

Model 1 RNN + TimeDistributed Dense: There is a big improvement if we compare it with the Model 0. It is because the Time Distributed Fully Connected layer improved the number of parameters (223,829). However, we can see that our model is clearly overfiting

Model 2 CNN + RNN + TimeDistributed Dense: This is clearly the model that has the best results (loss: 84.4901 - val_loss: 146.0076) mainly because adding a CNN helped our Neural Network to find the patterns in the Spectograms. However, we still have some overfitting problems.

Model 3 Deeper RNN + TimeDistributed Dense: This model is better than Model 1 because we added an RNN layer. However, it is not the best model. It is because of the fact that it doesn't have any CNN to analyse the spectogram and find the patterns in an easier way than directly analyse it with an RNN

Model 4 Bidirectional RNN + TimeDistributed Dense: This model has better results than Model 1 (because of the RNN bidirectional architecture), but it is still not the best model.

To conclude, Model 2 CNN + RNN + TimeDistributed Dense is the best model. However all of these models have some problems of overfitting. Even if we added for some BatchNormalization. Maybe it's a good idea to add some dropout layers. We'll study that for our final model.


(IMPLEMENTATION) Final Model
Now that you've tried out many sample models, use what you've learned to draft your own architecture! While your final acoustic model should not be identical to any of the architectures explored above, you are welcome to merely combine the explored layers above into a deeper architecture. It is NOT necessary to include new layer types that were not explored in the notebook.

However, if you would like some ideas for even more layer types, check out these ideas for some additional, optional extensions to your model:

If you notice your model is overfitting to the training dataset, consider adding dropout! To add dropout to recurrent layers, pay special attention to the dropout_W and dropout_U arguments. This paper may also provide some interesting theoretical background.
If you choose to include a convolutional layer in your model, you may get better results by working with dilated convolutions. If you choose to use dilated convolutions, make sure that you are able to accurately calculate the length of the acoustic model's output in the model.output_length lambda function. You can read more about dilated convolutions in Google's WaveNet paper. For an example of a speech-to-text system that makes use of dilated convolutions, check out this GitHub repository. You can work with dilated convolutions in Keras by paying special attention to the padding argument when you specify a convolutional layer.
If your model makes use of convolutional layers, why not also experiment with adding max pooling? Check out this paper for example architecture that makes use of max pooling in an acoustic model.
So far, you have experimented with a single bidirectional RNN layer. Consider stacking the bidirectional layers, to produce a deep bidirectional RNN!
All models that you specify in this repository should have output_length defined as an attribute. This attribute is a lambda function that maps the (temporal) length of the input acoustic features to the (temporal) length of the output softmax layer. This function is used in the computation of CTC loss; to see this, look at the add_ctc_loss function in train_utils.py. To see where the output_length attribute is defined for the models in the code, take a look at the sample_models.py file. You will notice this line of code within most models:

model.output_length = lambda x: x
The acoustic model that incorporates a convolutional layer (cnn_rnn_model) has a line that is a bit different:

model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
In the case of models that use purely recurrent layers, the lambda function is the identity function, as the recurrent layers do not modify the (temporal) length of their input tensors. However, convolutional layers are more complicated and require a specialized function (cnn_output_length in sample_models.py) to determine the temporal length of their output.

You will have to add the output_length attribute to your final model before running the code cell below. Feel free to use the cnn_output_length function, if it suits your model.

# specify the model
model_end = final_model(input_dim = 161,
                       filters = 220,
                       kernel_size = 11,
                       conv_stride = 2,
                       conv_border_mode='valid',units=200)
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
the_input (InputLayer)       (None, None, 161)         0         
_________________________________________________________________
conv1d (Conv1D)              (None, None, 220)         389840    
_________________________________________________________________
bn_conv_1d (BatchNormalizati (None, None, 220)         880       
_________________________________________________________________
bidirectional_2 (Bidirection (None, None, 400)         505200    
_________________________________________________________________
bn_rnn (BatchNormalization)  (None, None, 400)         1600      
_________________________________________________________________
time_distributed_3 (TimeDist (None, None, 29)          11629     
_________________________________________________________________
dropout_1 (Dropout)          (None, None, 29)          0         
_________________________________________________________________
time_distributed_4 (TimeDist (None, None, 29)          870       
_________________________________________________________________
softmax (Activation)         (None, None, 29)          0         
=================================================================
Total params: 910,019
Trainable params: 908,779
Non-trainable params: 1,240
_________________________________________________________________
None
Please execute the code cell below to train the neural network you specified in input_to_softmax. After the model has finished training, the model is saved in the HDF5 file model_end.h5. The loss history is saved in model_end.pickle. You are welcome to tweak any of the optional parameters while calling the train_model function, but this is not required.

train_model(input_to_softmax=model_end, 
            pickle_path='model_end.pickle', 
            save_model_path='model_end.h5', 
            spectrogram=True) # change to False if you would like to use MFCC features
Epoch 1/20
101/101 [==============================] - 186s 2s/step - loss: 265.2777 - val_loss: 239.8362
Epoch 2/20
101/101 [==============================] - 185s 2s/step - loss: 215.3410 - val_loss: 189.4265
Epoch 3/20
101/101 [==============================] - 184s 2s/step - loss: 188.1919 - val_loss: 173.8010
Epoch 4/20
101/101 [==============================] - 183s 2s/step - loss: 171.9946 - val_loss: 158.3863
Epoch 5/20
101/101 [==============================] - 184s 2s/step - loss: 160.7885 - val_loss: 162.7669
Epoch 6/20
101/101 [==============================] - 183s 2s/step - loss: 152.4714 - val_loss: 143.4888
Epoch 7/20
101/101 [==============================] - 182s 2s/step - loss: 145.1093 - val_loss: 140.4346
Epoch 8/20
101/101 [==============================] - 183s 2s/step - loss: 139.3487 - val_loss: 134.3605
Epoch 9/20
101/101 [==============================] - 183s 2s/step - loss: 133.8506 - val_loss: 133.9601
Epoch 10/20
101/101 [==============================] - 183s 2s/step - loss: 128.8915 - val_loss: 133.7576
Epoch 11/20
101/101 [==============================] - 183s 2s/step - loss: 124.5555 - val_loss: 131.8723
Epoch 12/20
101/101 [==============================] - 182s 2s/step - loss: 120.2604 - val_loss: 130.4166
Epoch 13/20
101/101 [==============================] - 184s 2s/step - loss: 116.5846 - val_loss: 128.7351
Epoch 14/20
101/101 [==============================] - 184s 2s/step - loss: 112.5825 - val_loss: 130.0187
Epoch 15/20
101/101 [==============================] - 183s 2s/step - loss: 109.2764 - val_loss: 131.1578
Epoch 16/20
101/101 [==============================] - 183s 2s/step - loss: 105.9382 - val_loss: 131.2532
Epoch 17/20
101/101 [==============================] - 182s 2s/step - loss: 103.0074 - val_loss: 129.8336
Epoch 18/20
101/101 [==============================] - 183s 2s/step - loss: 99.8909 - val_loss: 131.6434
Epoch 19/20
101/101 [==============================] - 182s 2s/step - loss: 96.6102 - val_loss: 134.9031
Epoch 20/20
101/101 [==============================] - 183s 2s/step - loss: 93.4834 - val_loss: 134.9697
Question 2: Describe your final model architecture and your reasoning at each step.

Answer:

For this final model, some of the best strategies were combined that were used in our former models

A convolution layer: It leads to great results because CNN are good to find patterns in images and Spectograms can be seen as an image A bidirectional RNN: A bidirectional RNN (for temporal dependencies) because we've seen that the results are better with them than simple RNN (like in Model 0, Model 3 and Model 4) Batch Normalization: Batch Normalization helps to make the learning faster by reducing the covariate shift TimeDistributed Fully Connected Layer Dropout: Used to avoid overfitting, it helps to find more complex patterns in the dataset Softmax activation function: Used for output probabilities

At the end we have 910,019 parameters and great results with a training loss of 93.4834 and a validation loss of 134.9697. We can improve this model (and prevent overfitting because we've still have this problem here) by adding more data, adding layers (2 convolution can be interesting but no maxpool or meanpool because we know that it leads to a too much loss of information).


STEP 3: Obtain Predictions
We have written a function for you to decode the predictions of your acoustic model. To use the function, please execute the code cell below.

import numpy as np
from data_generator import AudioGenerator
from keras import backend as K
from utils import int_sequence_to_text
from IPython.display import Audio
​
def get_predictions(index, partition, input_to_softmax, model_path):
    """ Print a model's decoded predictions
    Params:
        index (int): The example you would like to visualize
        partition (str): One of 'train' or 'validation'
        input_to_softmax (Model): The acoustic model
        model_path (str): Path to saved acoustic model's weights
    """
    # load the train and test data
    data_gen = AudioGenerator()
    data_gen.load_train_data()
    data_gen.load_validation_data()
    
    # obtain the true transcription and the audio features 
    if partition == 'validation':
        transcr = data_gen.valid_texts[index]
        audio_path = data_gen.valid_audio_paths[index]
        data_point = data_gen.normalize(data_gen.featurize(audio_path))
    elif partition == 'train':
        transcr = data_gen.train_texts[index]
        audio_path = data_gen.train_audio_paths[index]
        data_point = data_gen.normalize(data_gen.featurize(audio_path))
    else:
        raise Exception('Invalid partition!  Must be "train" or "validation"')
        
    # obtain and decode the acoustic model's predictions
    input_to_softmax.load_weights(model_path)
    prediction = input_to_softmax.predict(np.expand_dims(data_point, axis=0))
    output_length = [input_to_softmax.output_length(data_point.shape[0])] 
    pred_ints = (K.eval(K.ctc_decode(
                prediction, output_length)[0][0])+1).flatten().tolist()
    
    # play the audio file, and display the true and predicted transcriptions
    print('-'*80)
    Audio(audio_path)
    print('True transcription:\n' + '\n' + transcr)
    print('-'*80)
    print('Predicted transcription:\n' + '\n' + ''.join(int_sequence_to_text(pred_ints)))
    print('-'*80)
Use the code cell below to obtain the transcription predicted by your final model for the first example in the training dataset.

get_predictions(index=0,partition='train',
                input_to_softmax=final_model(input_dim = 161,
                       filters = 220,
                       kernel_size = 11,
                       conv_stride = 2,
                       conv_border_mode='valid',
                        units=200 ), 
                model_path='results/model_end.h5')
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
the_input (InputLayer)       (None, None, 161)         0         
_________________________________________________________________
conv1d (Conv1D)              (None, None, 220)         389840    
_________________________________________________________________
bn_conv_1d (BatchNormalizati (None, None, 220)         880       
_________________________________________________________________
bidirectional_4 (Bidirection (None, None, 400)         505200    
_________________________________________________________________
bn_rnn (BatchNormalization)  (None, None, 400)         1600      
_________________________________________________________________
time_distributed_7 (TimeDist (None, None, 29)          11629     
_________________________________________________________________
dropout_3 (Dropout)          (None, None, 29)          0         
_________________________________________________________________
time_distributed_8 (TimeDist (None, None, 29)          870       
_________________________________________________________________
softmax (Activation)         (None, None, 29)          0         
=================================================================
Total params: 910,019
Trainable params: 908,779
Non-trainable params: 1,240
_________________________________________________________________
None
--------------------------------------------------------------------------------
True transcription:

her father is a most remarkable person to say the least
--------------------------------------------------------------------------------
Predicted transcription:

hre fo ther as a mos re markabl personto sa tha least
--------------------------------------------------------------------------------
Use the next code cell to visualize the model's prediction for the first example in the validation dataset.

get_predictions(index=0, 
                partition='validation',
                input_to_softmax=final_model(input_dim = 161,
                       filters = 220,
                       kernel_size = 11,
                       conv_stride = 2,
                       conv_border_mode='valid',
                        units=200), 
                model_path='results/model_end.h5')
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
the_input (InputLayer)       (None, None, 161)         0         
_________________________________________________________________
conv1d (Conv1D)              (None, None, 220)         389840    
_________________________________________________________________
bn_conv_1d (BatchNormalizati (None, None, 220)         880       
_________________________________________________________________
bidirectional_5 (Bidirection (None, None, 400)         505200    
_________________________________________________________________
bn_rnn (BatchNormalization)  (None, None, 400)         1600      
_________________________________________________________________
time_distributed_9 (TimeDist (None, None, 29)          11629     
_________________________________________________________________
dropout_4 (Dropout)          (None, None, 29)          0         
_________________________________________________________________
time_distributed_10 (TimeDis (None, None, 29)          870       
_________________________________________________________________
softmax (Activation)         (None, None, 29)          0         
=================================================================
Total params: 910,019
Trainable params: 908,779
Non-trainable params: 1,240
_________________________________________________________________
None
--------------------------------------------------------------------------------
True transcription:

the bogus legislature numbered thirty six members
--------------------------------------------------------------------------------
Predicted transcription:

the bo tis lrdi slagor noberm therte six mevers
--------------------------------------------------------------------------------
One standard way to improve the results of the decoder is to incorporate a language model. We won't pursue this in the notebook, but you are welcome to do so as an optional extension.

If you are interested in creating models that provide improved transcriptions, you are encouraged to download more data and train bigger, deeper models. But beware - the model will likely take a long while to train. For instance, training this state-of-the-art model would take 3-6 weeks on a single GPU!

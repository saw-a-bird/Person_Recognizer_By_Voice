
from threading import Thread
import os

import librosa
import numpy as np
from pandas import DataFrame

from tqdm import tqdm
import matplotlib.pyplot as plt
import librosa.display
import tensorflow

from recognize.csv_service import get_csv

import seaborn as sn

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.metrics import confusion_matrix
from config import MODELS_DIR

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, LSTM, Input, LeakyReLU
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, MaxPool1D, GaussianNoise, GlobalMaxPooling1D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint


TRAIN_DIR = "training"

def get_models_dir():
    return MODELS_DIR

def get_spec_model_path():
    return os.path.join(MODELS_DIR, "model_spectrogram.h5")

def get_mel_model_path():
    return os.path.join(MODELS_DIR, "model_mel.h5")

threads_in_line = []
is_running = False

def get_is_running():
    return is_running
    
def train_model(why):
    new_thread = Thread(target=start_learning, args=(), daemon=True)
    threads_in_line.append({"why": why, "thread": new_thread})
    global is_running
    
    if (is_running == False):
        is_running = True
        new_thread.start()

def get_threads_in_line():
    return threads_in_line



def start_learning():
    df = get_csv()
    
    all_labels = df["person_name"].unique().tolist()
    label_count = len(all_labels)

    def label_person(label):
        return all_labels.index(label)


    ## Feature Extraction
    
    # Create empty arrays for the features
    ## FIRST COLUMN: ROW COUNT, SECOND ???, THIRD: MAX AUDIO DURATION (3min)
    dfX = len(df["person_name"]) # count
    dfZ = 130 # audioDuration
    # AllSpec = np.empty([dfX, 1025, dfZ])
    AllMel = np.empty([dfX, 128, dfZ])
    # AllMfcc = np.empty([dfX, 10, dfZ])
    # AllZcr = np.empty([dfX, dfZ])
    # AllCen = np.empty([dfX, dfZ])
    # AllChroma = np.empty([dfX, 12, dfZ])
    
    # Then we iterate through all rows in the csv file
    bad_index = []
    for i, row in tqdm(df.iterrows()):
        try:
            path = row['file_path']
            y, sr = librosa.load(path)
            
            # For Spectrogram
            # X = librosa.stft(y)
            # Xdb = librosa.amplitude_to_db(abs(X))
            # AllSpec[i] = Xdb
            
            # # Mel-Spectrogram 
            M = librosa.feature.melspectrogram(y=y)
            M_db = librosa.power_to_db(M)
            AllMel[i] = M_db
            
            # # MFCC
            # mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc= 10)
            # AllMfcc[i] = mfcc
            
            # # Zero-crossing rate
            # zcr = librosa.feature.zero_crossing_rate(y)[0]
            # AllZcr[i] = zcr
            
            # # Spectral centroid
            # sp_cen = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            # AllCen[i] = sp_cen
            
            # # Chromagram
            # chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12, n_fft=4096)
            # AllChroma[i] = chroma_stft
            

        except Exception as e:
            bad_index.append(i)

    # Delete the features at the corrupt indices
    # AllSpec = np.delete(AllSpec, bad_index, 0)
    AllMel = np.delete(AllMel, bad_index, 0)
    # AllMfcc = np.delete(AllMfcc, bad_index, 0)
    # AllZcr = np.delete(AllZcr, bad_index, 0)
    # AllCen = np.delete(AllCen, bad_index, 0)
    # AllChroma = np.delete(AllChroma, bad_index, 0)

    # Convert features to float32 
    # AllSpec = AllSpec.astype(np.float32)
    AllMel = AllMel.astype(np.float32)
    # AllMfcc = AllMfcc.astype(np.float32)
    # AllZcr = AllZcr.astype(np.float32)
    # AllCen = AllCen.astype(np.float32)
    # AllChroma = AllChroma.astype(np.float32)

        
    df["person_name"] =  df["person_name"].apply(label_person)

    # Convert numerical data into categorical data
    x = df.drop("person_name", axis = 1)
    y = tensorflow.keras.utils.to_categorical(df["person_name"], num_classes = label_count, dtype ="int32")
    
    # split train-test data
   # S_train, S_test, mfcc_train, mfcc_test, mel_train, mel_test, chroma_train, chroma_test, y_train, y_test = train_test_split(AllSpec, AllMfcc, AllMel, AllChroma, y, test_size= 0.2)
    mel_train, mel_test, y_train, y_test = train_test_split(AllMel, y, test_size= 0.2)


    y_true_train = np.argmax(y_train, axis= -1)
    y_true_test = np.argmax(y_test, axis= -1)

    #The scaling operation is applied only to the training dataset. During testing, the same maximum 
    # (from training data) is used to perform scaling on the testing data.


    # The original shape of Spectrogram is (944, 1025, 1295).

    # So, we find out the maximum of S_train and divide S_train by the maximum. During testing, we are dividing the 
    #S _test also by the maximum of S_train.

    # After that, we reshape the data in the form (N, row, col, 1) because CNN requires the input to be in this form. 
    # It indicates that there is only one channel in the image.

    # maximum = np.amax(S_train)
    # S_train = S_train/np.amax(maximum)
    # S_test = S_test/np.amax(maximum)

    # S_train = S_train.astype(np.float32)
    # S_test = S_test.astype(np.float32)

    # N, row, col = S_train.shape
    # S_train = S_train.reshape((N, row, col, 1))

    # N, row, col = S_test.shape
    # S_test = S_test.reshape((N, row, col, 1))


    ##  The original shape of MFCC is (944, 10, 1293). We first resize both the MFCC train and test data to 
    ## (944, 120, 600). After that, we reshape the data into (N, row, col, 1) for CNN. Then we standardize the data.

    # maximum = np.amax(mfcc_train)
    # mfcc_train = mfcc_train/np.amax(maximum)
    # mfcc_test = mfcc_test/np.amax(maximum)

    # mfcc_train = mfcc_train.astype(np.float32)
    # mfcc_test = mfcc_test.astype(np.float32)

    # N, row, col = mfcc_train.shape
    # mfcc_train = mfcc_train.reshape((N, row, col, 1))

    # N, row, col = mfcc_test.shape
    # mfcc_test = mfcc_test.reshape((N, row, col, 1))

    ## The original shape of the Mel-Spectrogram is (944, 128, 1293). We first scale the train and test data using 
    ## the maximum of train data. Then we reshape the data to (N, row, col, 1) for CNN.

    maximum = np.amax(mel_train)
    mel_train = mel_train/np.amax(maximum)
    mel_test = mel_test/np.amax(maximum)

    mel_train = mel_train.astype(np.float32)
    mel_test = mel_test.astype(np.float32)

    N, row, col = mel_train.shape
    mel_train = mel_train.reshape((N, row, col, 1))

    N, row, col = mel_test.shape
    mel_test = mel_test.reshape((N, row, col, 1))

    # Classification

    def get_confusion_matrix(pred_test_y):
        conf_mat = confusion_matrix(y_true_test, pred_test_y, normalize= 'true')
        conf_mat = np.round(conf_mat, 2)

        conf_mat_df = DataFrame(conf_mat, columns = all_labels, index= label_count)
        plt.figure(figsize = (10,7), dpi = 200)
        sn.set(font_scale=1.4)
        sn.heatmap(conf_mat_df, annot=True, annot_kws={"size": 16}) # font size
        plt.tight_layout()

    def get_majority(pred) :
        N = len(pred[0])
        vote = []
        for i in range(N) :
            candidates = [x[i] for x in pred]
            candidates = np.array(candidates)
            uniq, freq = np.unique(candidates, return_counts= True)
            vote.append(uniq[np.argmax(freq)])
            
        vote = np.array(vote)
        return vote
    
    
    # def train_spec_model(): ## Using Spectrogram
    #     # 1. First, We define a CNN model for classification. 
    #     def get_spec_model():
    #         model = Sequential()
    #         model.add(Conv2D(8, (3,3), activation= 'relu', input_shape= S_train[0].shape, padding= 'same'))

    #         ## Pooling:
    #         model.add(MaxPooling2D((4,4), padding= 'same'))
    #         model.add(Conv2D(16, (3,3), activation= 'relu', padding= 'same'))
    #         model.add(MaxPooling2D((4,4), padding= 'same'))
    #         model.add(Conv2D(32, (3,3), activation= 'relu', padding= 'same'))
    #         model.add(MaxPooling2D((4,4), padding= 'same'))
    #         model.add(Conv2D(64, (3,3), activation= 'relu', padding= 'same'))
    #         model.add(MaxPooling2D((4,4), padding= 'same'))
    #         model.add(Conv2D(64, (3,3), activation= 'relu', padding= 'same'))
    #         model.add(MaxPooling2D((4,4), padding= 'same'))

    #         ## Flattening:
    #         model.add(Flatten())

    #         ## ReLu as activation function
    #         model.add(Dense(128, activation= 'relu'))
    #         model.add(Dense(64, activation= 'relu'))

    #         ## For the last layer, we use softmax
    #         model.add(Dense(label_count, activation= 'softmax'))

    #         # 2. We then used the Adam optimizer.
    #         model.compile(optimizer= 'Adam', loss= 'categorical_crossentropy')
    #         model.summary()
    #         return model

    #     # 3. We have a checkpoint after every 5 epochs, which we can use if in any case our model is interrupted during training
    #     checkpoint = ModelCheckpoint(os.path.join(MODELS_DIR, TRAIN_DIR, "new_spec_model_{epoch:03d}.h5"), period= 5)

    #     # 5. Creating our model
    #     spec_cnn_model = get_spec_model()

    #     # 4. Training our model
    #     spec_cnn_model.fit(S_train, y_train, epochs= 100, callbacks= [checkpoint], batch_size= 32, verbose= 1)

    #     # 5. Save the trained model
    #     spec_cnn_model.save(get_spec_model_path())

    #     print("Spec model successfully trained.")
        
        
    #     # 6. Evaluation:
        
    #     ## Training Accuracy
    #     y_pred_spec_train = spec_cnn_model.predict(S_train)
    #     y_pred_spec_train = np.argmax(y_pred_spec_train, axis= -1)

    #     correct = len(y_pred_spec_train) - np.count_nonzero(y_pred_spec_train - y_true_train)
    #     acc = correct/ len(y_pred_spec_train)
    #     acc = np.round(acc, 4) * 100

    #     print("Train Accuracy: ", correct, "/", len(y_pred_spec_train), " = ", acc, "%")

    #     ## Testing Accuracy
    #     y_pred_spec_test = spec_cnn_model.predict(S_test)
    #     y_pred_spec_test = np.argmax(y_pred_spec_test, axis= -1)

    #     correct = len(y_pred_spec_test) - np.count_nonzero(y_pred_spec_test - y_true_test)
    #     acc = correct/ len(y_pred_spec_test)
    #     acc = np.round(acc, 4) * 100

    #     print("Test Accuracy: ", correct, "/", len(y_pred_spec_test), " = ", acc, "%")

    #     # Confusion matrix
    #     #get_confusion_matrix(y_pred_spec_test)
    #     return y_pred_spec_train, y_pred_spec_test
    
    

    # def train_mfcc_model():
    #     # For MFCC, we trained two models and used an ensemble of the two models to report accuracy. 
        
    #     def get_mfcc_cnn_model() :
    #         model = Sequential()
    #         model.add(Conv2D(16, (3,3), input_shape= mfcc_train[0].shape, activation= 'tanh', padding= 'same'))
    #         model.add(BatchNormalization())
    #         model.add(MaxPooling2D((4,6), padding= 'same'))
    #         model.add(Conv2D(32, (3,3), input_shape= mfcc_train[0].shape, activation= 'tanh', padding= 'same'))
    #         model.add(BatchNormalization())
    #         model.add(MaxPooling2D((4,6), padding= 'same'))
    #         model.add(Conv2D(64, (3,3), input_shape= mfcc_train[0].shape, activation= 'tanh', padding= 'same'))
    #         model.add(BatchNormalization())
    #         model.add(MaxPooling2D((4,6), padding= 'same'))
    #         model.add(Flatten())
    #         # model.add(Dense(256, activation= 'tanh'))
    #         model.add(Dense(256, activation= 'tanh'))
    #         model.add(Dense(64, activation= 'tanh'))
    #         model.add(Dense(label_count, activation= 'softmax'))

    #         model.compile(optimizer= 'Adam', loss= 'categorical_crossentropy')
    #         model.summary()
    #         return model

    #     #checkpoint = ModelCheckpoint(os.getcwd()+"/models_dir/new_mfcc_model_{epoch:03d}.h5", period= 5)

    #     mfcc_cnn_models = []
    #     mfcc_cnn_models.append(get_mfcc_cnn_model())
    #     mfcc_cnn_models.append(get_mfcc_cnn_model())
    #     mfcc_cnn_models.append(get_mfcc_cnn_model())


    #     # We have used k-fold cross-validation with k = 10.
    #     kf = KFold(n_splits = 10)
    #     counter = 1
    #     for model in mfcc_cnn_models:  
    #         for train_index, val_index in kf.split(mfcc_train, np.argmax(y_train, axis= -1)):
    #             kf_mfcc_train = mfcc_train[train_index]
    #             kf_X_val = mfcc_train[val_index]
    #             kf_y_train = y_train[train_index]
    #             kf_y_val = y_train[val_index]

    #             model.fit(kf_mfcc_train, kf_y_train, validation_data= (kf_X_val, kf_y_val), epochs= 30, batch_size= 30, verbose= 1)
                
    #         model.save(os.getcwd() + "/models_dir/model_mfcc_"+ str(counter) + ".h5")
    #         counter += 1
            
    #     print("MFCC Model trained.")

    #     # Evaluation:

    #     # Training Accuracy
    #     y_preds = []
    #     for model in mfcc_cnn_models:   
    #         y_pred = model.predict(mfcc_train)
    #         y_pred = np.argmax(y_pred, axis= -1)
    #         y_preds.append(y_pred)

    #     y_pred_mfcc_train = get_majority(y_preds)

    #     correct = len(y_pred_mfcc_train) - np.count_nonzero(y_pred_mfcc_train - y_true_train)
    #     acc = correct/ len(y_pred_mfcc_train)
    #     acc = np.round(acc, 4) * 100

    #     print("Train Accuracy: ", correct, "/", len(y_pred_mfcc_train), " = ", acc, "%")

    #     ## Testing Accuracy
    #     y_preds = []
    #     for model in mfcc_cnn_models:   
    #         y_pred = model.predict(mfcc_test)
    #         y_pred = np.argmax(y_pred, axis= -1)
    #         y_preds.append(y_pred)

    #     y_pred_mfcc_test = get_majority(y_preds)

    #     correct = len(y_pred_mfcc_test) - np.count_nonzero(y_pred_mfcc_test - y_true_test)
    #     acc = correct/ len(y_pred_mfcc_test)
    #     acc = np.round(acc, 4) * 100

    #     print("Testing Accuracy: ", correct, "/", len(y_pred_mfcc_test), " = ", acc, "%")

    #     # Confusion Matrix
    #     #get_confusion_matrix(y_pred_mfcc_test)
    #     return y_pred_mfcc_train, y_pred_mfcc_test

    

    def train_mel_model(): ## Using Mel-Spectrogram
        def get_mel_cnn_model() :
            
            model = Sequential()
            model.add(Conv2D(8, (3,3), activation= 'relu', input_shape= mel_train[0].shape, padding= 'same'))
            model.add(MaxPooling2D((4,4), padding= 'same'))
            model.add(Conv2D(16, (3,3), activation= 'relu', padding= 'same'))
            model.add(MaxPooling2D((4,4), padding= 'same'))
            model.add(Conv2D(32, (3,3), activation= 'relu', padding= 'same'))
            model.add(MaxPooling2D((4,4), padding= 'same'))
            model.add(Conv2D(64, (3,3), activation= 'relu', padding= 'same'))
            model.add(MaxPooling2D((4,4), padding= 'same'))
            model.add(Conv2D(64, (3,3), activation= 'relu', padding= 'same'))
            model.add(MaxPooling2D((4,4), padding= 'same'))
            model.add(Flatten())
            model.add(Dense(64, activation= 'relu'))
            model.add(Dense(label_count, activation= 'softmax'))

            model.compile(optimizer= 'Adam', loss= 'categorical_crossentropy')
            model.summary()

            return model


        checkpoint = ModelCheckpoint(os.path.join(MODELS_DIR, TRAIN_DIR, "new_mel_model_{epoch:03d}.h5"), period= 5)

        mel_cnn_model = get_mel_cnn_model()
        mel_cnn_model.fit(mel_train, y_train, epochs= 200, callbacks= [checkpoint], batch_size= 32, verbose= 1)
        mel_cnn_model.save(get_mel_model_path())
                
        # Evaluation:

        ## Training Accuracy
        y_pred_mel_train = mel_cnn_model.predict(mel_train)
        y_pred_mel_train = np.argmax(y_pred_mel_train, axis= -1)

        correct = len(y_pred_mel_train) - np.count_nonzero(y_pred_mel_train - y_true_train)
        acc = correct/ len(y_pred_mel_train)
        acc = np.round(acc, 4) * 100

        print("Train Accuracy: ", correct, "/", len(y_pred_mel_train), " = ", acc, "%")

        ## Testing Accuracy
        y_pred_mel_test = mel_cnn_model.predict(mel_test)
        y_pred_mel_test = np.argmax(y_pred_mel_test, axis= -1)

        correct = len(y_pred_mel_test) - np.count_nonzero(y_pred_mel_test - y_true_test)
        acc = correct/ len(y_pred_mel_test)
        acc = np.round(acc, 4) * 100

        print("Testing Accuracy", correct, "/", len(y_pred_mel_test), " = ", acc, "%")

        # Confusion Matrix
        #get_confusion_matrix(y_pred_mel_test)
        return y_pred_mel_train, y_pred_mel_test



    # # # Training our models
    # # y_pred_spec_train, y_pred_spec_test = train_spec_model()
    # # y_pred_mfcc_train, y_pred_mfcc_test = train_mfcc_model()
    # # y_pred_mel_train, y_pred_mel_test = train_mel_model()
    
    # # # Rounding up all measurements
    
    # # ## Best training accuracy:
    
    # # y_pred = [y_pred_spec_train, y_pred_mfcc_train, y_pred_mel_train]
    # # y_pred = get_majority(y_pred)

    # # correct = len(y_pred) - np.count_nonzero(y_pred - y_true_train)
    # # acc = correct/ len(y_pred)
    # # acc = np.round(acc, 4) * 100

    # # print("Training Accuracy: ", correct, "/", len(y_pred), " = ", acc, "%")

    # # ## Best testing accuracy:
    # # y_pred = [y_pred_spec_test, y_pred_mfcc_test, y_pred_mel_test]
    # # y_pred = get_majority(y_pred)

    # # correct = len(y_pred) - np.count_nonzero(y_pred - y_true_test)
    # # acc = correct/ len(y_pred)
    # # acc = np.round(acc, 4) * 100
    # # print("Testing Accuracy: ", correct, "/", len(y_pred), " = ", acc, "%")

    # # # Confusion matrix
    # # #get_confusion_matrix(y_pred)
    
    # print("training by spectogram")
    # train_spec_model()
    print("training by mel-spec")
    train_mel_model()
    
    global threads_in_line
    threads_in_line.pop(0) 
    l = len(threads_in_line)
    if l > 0:
        if (l > 1):
            threads_in_line = [threads_in_line[l-1]]
            
        threads_in_line[0].get("thread").start()
    else:
        global is_running
        is_running = False
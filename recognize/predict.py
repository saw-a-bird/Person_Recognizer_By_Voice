import numpy as np
import librosa
from keras.models import load_model
from .csv_service import get_label
from .train import get_mel_model_path
from .audio_butcher import split_audio, add_silence

#import IPython.display as display

def find_majority(array):
    confidence = {}
    
    for label in array:
        if label in confidence: confidence[label] += 1
        else: confidence[label] = 1

    best_label, best_score = (None, None)
    for label, value in confidence.items():
        score = value/len(array)
        confidence[label] = score
        
        if best_score == None or score > best_score:
            best_label = label
            best_score = score
            
    return best_label, confidence


def predict(file):
    try:
        mel_cnn_model = load_model(get_mel_model_path())
        #TEST_FILE = "/home/mahdi/Desktop/Python Notebook/speech_recognizer/recognize/audio_dir/salma/0-2022-11-26 16:38:15.716951.wav"
        #TEST_FILE = "/home/mahdi/Desktop/Python Notebook/speech_recognizer/recognize/audio_dir/mahdi/0-2022-11-26 18:41:04.249768.wav"
        #TEST_FILE = "/home/mahdi/Desktop/Python Notebook/speech_recognizer/temp_records/2022-11-28 20:46:22.041438.wav"
        wave, sr = librosa.load(file)

        duration = librosa.get_duration(y = wave)
        print("duration:", duration)
        
        if (duration > 3):
            print("predicting by majority")
            splits = split_audio(wave, sr, mercy = True)
            predictions = []

            for wave_portion in splits:
                # display.display(display.Audio(wave, rate=sr))
                duration = librosa.get_duration(y = wave_portion)
                print("split", duration)
                if (duration >= 2*2 and duration < 3*2):
                    print("adding silence to split...")
                    wave_portion = add_silence(libr = (wave_portion, sr))
                    
                predictions.append(mel_predict(wave_portion, mel_cnn_model))

            label, confidences = find_majority(predictions)
            print(confidences)
            return (get_label(label), "(confidence: "+str(round(confidences[label]*100))+"%)")
        
        elif (duration >= 1):
            wave = add_silence(audiofile=file)
            print("predicted easy way", duration, "to", librosa.get_duration(wave))
            return (get_label(mel_predict(wave, mel_cnn_model)), "")
        
        else:
            return (False, False)
        
    except:
        return (None, None)


def mel_predict(wave, model):
    M = librosa.feature.melspectrogram(y=wave)
    spec_x = librosa.power_to_db(M)

    row, col = spec_x.shape
    spec_x = spec_x.reshape((row, col, 1))
    spec_x = spec_x[np.newaxis, ...]

    x_predict = model.predict(spec_x)
    predicted_label=np.argmax(x_predict,axis=1)
    
    return predicted_label[0]



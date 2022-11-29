import os
import librosa
import soundfile as sf
import numpy as np
from . import csv_service
from  .audio_butcher import remove_silence, split_audio
import shutil
from .train import train_model, get_models_dir
from config import BACKUP_DIR, TEMP_DIR, AUDIO_DIR

#MAX_FILES_PER_DIR = 20

def get_dataset_fullness(): 
    dataset = {}# dataset = { "mahdi": "20%", }
    
    all_files_count = 0
    for person_name in os.listdir(AUDIO_DIR):
        path = os.path.join(AUDIO_DIR, person_name)
        if os.path.isdir(path):
            print("found", person_name)
            c = len(os.listdir(path))
            dataset[person_name] = c
            all_files_count += c
    
    data = []
    for k, v in dataset.items():
        formatted_val = str(round((v / all_files_count)*100, 2))
        data.append(k + ' '+formatted_val+'%')
    # for person_name in os.listdir(TEMP_DIR):
    #     if os.path.isdir(os.path.join(TEMP_DIR, person_name)):
    #         c = (len(os.listdir(os.path.join(TEMP_DIR, person_name)))/MAX_FILES_PER_DIR)*100
    #         dataset.append(person_name + ' '+str(c)+'%')
    #         print("temp", person_name)
        
    return data
        
    
def add_record(person_name, audio_file_name):
    _wav_path = os.path.join(TEMP_DIR, audio_file_name)
    
    # remove silence
    wave, sr = librosa.load(_wav_path, sr=None)
    trimmed_y, _ = remove_silence(wave, top_db= 15)

    # split audio
    splits = split_audio(trimmed_y, sr, mercy=False)
    
    if len(splits) > 0:
        keyLabel = "UPDATING"
        
        _classe_folder = os.path.join(AUDIO_DIR, person_name)
        if (os.path.exists(_classe_folder) == False):
            os.mkdir(_classe_folder)
            keyLabel = "ADDING"
            
        files = []
        for i, segment in enumerate(splits):
            out_file = os.path.join(_classe_folder, str(i) +"-"+audio_file_name)
            sf.write(out_file, segment, sr)
            files.append(out_file)

        # check if surpassed limit
        #numberFiles = len(os.listdir(_classe_folder))
        #if (numberFiles >= MAX_FILES_PER_DIR):
        add_to_person(person_name, files, keyLabel)
        print(keyLabel)
        return True
    else:
        return False
             
             
train_thread = None
def add_to_person(person_name, files = None, keyLabel = None):
    def getting_necessary_data(file):
        wave, sr = librosa.load(file, sr=None)
        silence_, _ = librosa.effects.trim(wave)
        silence_trimmed_duration = librosa.get_duration(y=silence_, sr=sr)
        return {"filename": wav, "sound_rate": sr, "silence_duration": silence_trimmed_duration}
    
    # VERSION 1
    # _temp_classe_loc = os.path.join(TEMP_DIR, person_name)    
    # _main_classe_folder = os.path.join(AUDIO_DIR, person_name)
    
    # if (os.path.exists(_main_classe_folder) == False):
    #     os.mkdir(_main_classe_folder)
        
    # allAudios = []
    # for wav in os.listdir(_temp_classe_loc):
    #     in_wav = os.path.join(_temp_classe_loc, wav)
    #     wave, sr = librosa.load(in_wav, sr=None)
        
    #     silence_, _ = librosa.effects.trim(wave)
    #     silence_trimmed_duration = librosa.get_duration(y=silence_, sr=sr)
        
    #     out_wav = os.path.join(_main_classe_folder, wav)
    #     allAudios.append({"filename": out_wav, "sound_rate": sr, "silence_duration": silence_trimmed_duration})
        
    #     # Move audio file
    #     shutil.move(in_wav, out_wav)
    
    # # remove dir
    # os.rmdir(_temp_classe_loc)
    
    # # edit csv file
    
    # allAudios = sorted(allAudios, key=lambda d: d['silence_duration']) # compare also by sound_rate later
    # allAudios = allAudios[: MAX_FILES_PER_DIR]

    # VERSION 2
    
    allAudios = []
    for wav in files:
        allAudios.append(getting_necessary_data(wav))
        
    csv_service.csv_add_new(person_name, allAudios)
    train_model(keyLabel+" '"+person_name+"'")
    

def remove_person(person_name):
    
    # removing all old stuff
    temp_classe_dir = os.path.join(TEMP_DIR, person_name)
    audio_classe_dir = os.path.join(AUDIO_DIR, person_name)
    
    if (os.path.exists(audio_classe_dir) and len(os.listdir(AUDIO_DIR)) > 1):
        shutil.rmtree(audio_classe_dir)
        csv_service.csv_remove(person_name)
        train_model("REMOVING '"+person_name+"'")
        return True
    
    elif (os.path.exists(temp_classe_dir)):
        shutil.rmtree(temp_classe_dir)
        return True
    
    return False
        
        
        
        
        
def reset_everything():
    # removing all old stuff
    print("removing csvs..")
    if os.path.exists(csv_service.get_location_csv()):
        os.remove(csv_service.get_location_csv())
        
    print("removing temps..")
    for root, dirs, files in os.walk(TEMP_DIR):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))
            
    print("removing audios..")
    for root, dirs, files in os.walk(AUDIO_DIR):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))
    
    print("removing models..")
    for root, dirs, files in os.walk(get_models_dir()):
        for f in files:
            os.unlink(os.path.join(root, f))
          
    # and adding anew  
    def add_classe_files(dir_classe_name):
        in_classe_loc = os.path.join(BACKUP_DIR, dir_classe_name)
        out_classe_folder = os.path.join(AUDIO_DIR, dir_classe_name)
        
        print(dir_classe_name, "- found", len(os.listdir(in_classe_loc)), "items")
        
        allAudios = []
        for wav in os.listdir(in_classe_loc):
            in_wav_loc = os.path.join(in_classe_loc, wav)
            in_wav_name = os.path.basename(in_wav_loc)
            
            # remove silence
            wave, sr = librosa.load(in_wav_loc, sr=None)
            trimmed_y, _ = remove_silence(wave, top_db= 15)

            # split audio
            split = split_audio(trimmed_y, sr, mercy=False)
            
            for i, segment in enumerate(split):
                out_wav_name = os.path.join(out_classe_folder, str(i) +"-"+in_wav_name)
                
                # sf.write(out_wav_name, segment, sr)
                silence_, _ = librosa.effects.trim(segment)
                silence_trimmed_duration = librosa.get_duration(y=silence_, sr=sr)
                allAudios.append({"filename": out_wav_name, "segment": segment, "sound_rate": sr, "silence_duration": silence_trimmed_duration})
        
        
        # if (len(allAudios) >= MAX_FILES_PER_DIR):
        #     allAudios = sorted(allAudios, key=lambda d: d['silence_duration']) # compare also by sound_rate later
        #     allAudios = allAudios[: MAX_FILES_PER_DIR]
            
        if (os.path.exists(out_classe_folder) == False):
            os.mkdir(out_classe_folder)

        for audio in allAudios:
            sf.write(audio.get("filename"), audio.get("segment"), audio.get("sound_rate"))
        
        print(dir_classe_name, "successfully recognized and added on queue for model.")
        csv_service.csv_add_new(dir_classe_name, allAudios)
        #else:
            #print(dir_classe_name, "still unable to cross the threshold", len(allAudios), "/", MAX_FILES_PER_DIR,"location: ", in_classe_loc)
            
    print("adding new audio files from backup..")
    
    for dir_classe_name in os.listdir(BACKUP_DIR):
        add_classe_files(dir_classe_name)
        
                    
    print("training model..")
    train_model("DEBUG - RESTORE BACKUP")
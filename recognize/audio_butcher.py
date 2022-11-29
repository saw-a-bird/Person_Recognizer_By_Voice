import librosa
from config import SPLIT_BY_SEC, TEMP_FILE
from pydub import AudioSegment
from soundfile import write

def split_audio(y, sr, mercy = False, f = True):
    segment_length =  SPLIT_BY_SEC * sr
    split = []
    for s in range(0, len(y), segment_length):
        t = y[s: s + segment_length]
        if (mercy == False):
            duration = librosa.get_duration(y = t)
            if (f == True and duration != SPLIT_BY_SEC * 2) or (f==False and duration != SPLIT_BY_SEC):
                continue
             
        split.append(t)        
        
    return split

def remove_silence(y, top_db):
    return librosa.effects.trim(y, top_db = top_db)

def add_silence(audiofile = None, libr = (None, None)):
    original_segment = None
    
    if audiofile == None:
        write(TEMP_FILE, libr[0], libr[1])
        original_segment = AudioSegment.from_wav(TEMP_FILE)
    else:
        original_segment = AudioSegment.from_wav(audiofile)
    
    silence_duration = (SPLIT_BY_SEC * 1000) - len(original_segment)
    silenced_segment = AudioSegment.silent(duration=silence_duration)
    combined_segment = original_segment + silenced_segment
    combined_segment.export(TEMP_FILE, format="wav")

    # audio + silence
    wave, sr = librosa.load(TEMP_FILE)
    return wave
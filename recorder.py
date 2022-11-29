import wave
from dataclasses import dataclass, asdict
import threading
import pyaudio
import datetime
from os import path
from config import TEMP_DIR

@dataclass
class StreamParams:
    format: int = pyaudio.paInt16
    channels: int = 2
    rate: int = 44100
    frames_per_buffer: int = 1024
    input: bool = True
    output: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

class Recorder:
    _writing_loop : threading.Thread
    """Recorder uses the blocking I/O facility from pyaudio to record sound
    from mic.
    Attributes:
        - stream_params: StreamParams object with values for pyaudio Stream
            object
    """
    def __init__(self, stream_params: StreamParams) -> None:
        self.stream_params = stream_params
        self._pyaudio = None
        self._stream = None
        self._last_audio_name = None
        self._wav_file_writer = None
        self._is_started = False
        self._time_tracker = 0
        
    def start_recording(self, timeTracker):
        if self._is_started == False:
            self._is_started = True
            self._last_audio_name = str(datetime.datetime.now()) + ".wav"
            self._create_recording_resources()
            
            print("Recording started...")
            
            self._writing_loop = threading.Thread(target=_write_wav_file_reading_from_stream, args=(self,), daemon=True)
            self._writing_loop.start()
            
            self._time_tracker = 0

            timeTracker.update('(Recording: ...)', None, None, None, True)
            threading.Thread(target=_track_time, args=(self, timeTracker), daemon=True).start()

    def stop_recording(self):
        self._is_started = False
        print("Recording stopped.")
    
    def _create_recording_resources(self):
        self._pyaudio = pyaudio.PyAudio()
        self._stream = self._pyaudio.open(**self.stream_params.to_dict())
        self._create_wav_file()

    def _create_wav_file(self):
        self._wav_file_writer = wave.open(path.join(TEMP_DIR, self._last_audio_name), "wb")
        self._wav_file_writer.setnchannels(self.stream_params.channels)
        self._wav_file_writer.setsampwidth(self._pyaudio.get_sample_size(self.stream_params.format))
        self._wav_file_writer.setframerate(self.stream_params.rate)

    def _close_recording_resources(self) -> None:
        self._wav_file_writer.close()
        self._stream.close()
        self._pyaudio.terminate()
        

def _write_wav_file_reading_from_stream(recorder : Recorder):
    while True:
        if recorder._is_started == False: 
            recorder._close_recording_resources()
            break
        
        audio_data = recorder._stream.read(recorder.stream_params.frames_per_buffer)
        recorder._wav_file_writer.writeframes(audio_data)
        

def _track_time(recorder : Recorder, timeTracker):
    recorder._time_tracker += 0.1
    timeTracker.update('(Recording: '+str(round(recorder._time_tracker, 1))+'s)',)
    if recorder._is_started == True: 
        threading.Timer(0.1, _track_time, args=(recorder,timeTracker)).start()
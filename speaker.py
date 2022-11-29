from gtts import gTTS
from io import BytesIO
from pygame import mixer

LINES = ["Hello stranger. What's your name?"
         "Hello {name}!"]


class Speaker:

    @classmethod
    def speak(self, text):
        mp3_file_object = BytesIO()
        tts = gTTS(text, lang = "en-us")
        tts.write_to_fp(mp3_file_object)
     
        mixer.init()
        mixer.music.load(mp3_file_object, 'mp3')
        mixer.music.play()
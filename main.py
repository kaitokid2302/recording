import pyaudio
import numpy as np
import wave
from sklearn.preprocessing import StandardScaler
import time
from getfeature import get_features
from tensorflow.keras.models import load_model
import os
from prediction import emotion

def record():
    ok = False
    try:
        while True:
            audio = pyaudio.PyAudio()
            stream = audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=4096)
            frames = []
            for _ in range(int(44100 / 4096 * 3)):
                if ok == False:
                    print("Talk now")
                    ok = True
                try:
                    data = stream.read(4096)
                    frames.append(data)
                except IOError as e:
                    if e.errno == pyaudio.paInputOverflowed:
                        continue

            sound_file = wave.open('myrecording.wav', 'wb')
            sound_file.setnchannels(1)
            sound_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
            sound_file.setframerate(44100)
            sound_file.writeframes(b''.join(frames))
            sound_file.close()


            features = get_features('myrecording.wav')
            print(features.shape) # should be (2376,)
            print(emotion(features))
            

    except KeyboardInterrupt:
        print("Đã ngắt ghi âm.")
 


def main():
    record()

if __name__ == '__main__':
    main()
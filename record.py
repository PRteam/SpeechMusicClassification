import wave
import alsaaudio 
import time

# type sudo apt-get install python-alsaaudio if required

#DO NOT CHANGE
RATE = 44100
NO_CHANNELS = 1
SAMPLE_WIDTH = 2
CHUNK_SIZE = 1024
inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE)

#User defineds
FILENAME = "test.wav"
RECORD_TIME = 7 # this is in seconds

def record():

    global RATE, NO_CHANNELS, SAMPLE_WIDTH, CHUNK_SIZE, RECORD_TIME, FILENAME, inp
    
    print "recording ..."
    w = wave.open(FILENAME, 'wb')
    w.setnchannels(NO_CHANNELS)
    w.setsampwidth(SAMPLE_WIDTH)
    w.setframerate(RATE)
    
    record_limit = RECORD_TIME
    
    past_time = time.time()
    delta_time = 0
    while delta_time <= record_limit:
        delta_time = int(time.time() - past_time)
        l, data = inp.read()
        w.writeframes(data)
    w.close()

record()

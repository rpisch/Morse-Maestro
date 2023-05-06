# Decoding a morse signal - Ryan Pischinger, Braden Karley

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import pyttsx3 as pyttsx
import pyaudio
import wave
import scipy
from scipy.signal import filtfilt

FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

pa = pyaudio.PyAudio()

stream = pa.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=FRAMES_PER_BUFFER
)

print('start recording')

seconds = 8
frames = []
second_tracking = 0
second_count = 0
for i in range(0, int(RATE/FRAMES_PER_BUFFER*seconds)):
    data1 = stream.read(FRAMES_PER_BUFFER)
    frames.append(data1)
    second_tracking += 1
    if second_tracking == RATE/FRAMES_PER_BUFFER:
        second_count += 1
        second_tracking = 0
        print(f'Time left: {seconds - second_count} seconds')

stream.stop_stream()
stream.close()
pa.terminate()

obj = wave.open('recording.wav', 'wb')
obj.setnchannels(CHANNELS)
obj.setsampwidth(pa.get_sample_size(FORMAT))
obj.setframerate(RATE)
obj.writeframes(b''.join(frames))
obj.close()

file = wave.open('recording.wav', 'rb')

sameple_freq = file.getframerate()
frames = file.getnframes()
signal_wave = file.readframes(-1)

file.close()


#reads wave file and saves the sample rate and the data of the file
sample_rate, data = wavfile.read("recording.wav")

# Calculating Sample rate, # of samples, time in seconds, and units in the data



#creates a list that stores time in seconds by calculating the current sample number / the sample rate
time_s = []
for i in range (0, len(data)):
    time_s.append(i/sample_rate)

plt.plot(time_s,data)
plt.title("Raw Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()

reffreq = 1000
range_freq = 100
def bandpass(signal):
    fs = 44100.0
    lowcut = reffreq-range_freq
    hicut = reffreq+range_freq
    nyq = 0.5*fs
    low = lowcut / nyq
    high = hicut / nyq
    order = 4
    b,a = scipy.signal.butter(order, [low, high], 'bandpass', analog=False)
    y = scipy.signal.filtfilt(b, a, signal, axis=0)
    return (y)

testtrimdata = bandpass(data)
plt.plot(time_s, testtrimdata)
plt.title("Filtered Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()

def trim_data():
    trim1 = bandpass(data)
    index_trim = np.where(trim1>400)[0]
    if(len(index_trim)<2):
        print("Please Increase Volume!")
        trimmed_data = []
    else:
        index_trim_head = index_trim[0]
        index_trim_tail = index_trim[-1]
        trimmed_data = trim1[index_trim_head:index_trim_tail]
        time_s_plt = time_s[index_trim_head:index_trim_tail]
    return trimmed_data, time_s_plt

data_trimmed, plottime = trim_data()
plt.plot(plottime, data_trimmed)
plt.title("Trimmed Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()

sampletime = (len(data_trimmed))/sample_rate
WPM = 15    #select words per minute
UPW = 50    #50 units per word
units = round(((UPW/60)*WPM*sampletime),0) 
print(units)

time_s = []
for i in range (0, len(data_trimmed)):
    time_s.append(i/sample_rate)



# Shows the specified letter in the time domain and the frequency domain to evaluate the fft
def evaluate_letter(s,e):
    snipdata = []
    
    # plots a snippet of the original waveform
    start = int((s/units)*(len(data_trimmed)))
    snip = int((e/units)*(len(data_trimmed)))
    snipdata = data_trimmed[start:snip]
        
    #fig,axs = plt.subplots(2,1)
    #plt.sca(axs[0])
    #plt.plot(snipdata)

    # plots the fft
    sample = len(snipdata)
    fhat = np.fft.fft(snipdata, sample)
    PSD = fhat * np.conj(fhat) / sample
    freq = (1/(.001*sample))*np.arange(sample)
    L = np.arange(1,np.floor(sample/2),dtype='int')

    max_val = np.amax(PSD[L])
    return(max_val)
    #plt.sca(axs[1])
    #plt.plot(freq[L], PSD[L])

    #plt.show()

# function to take a specified start and end and determine whether the PSD of the fft meets a ref frequency
def unit_fft(start, end):
    snipdata = []
   
    # creates new list of shortened data
    startsnip = int((start/units)*(len(data_trimmed)))
    endsnip = int((end/units)*(len(data_trimmed)))
    snipdata = data_trimmed[startsnip:endsnip]
   
    # performs fft on data snippet
    sample = len(snipdata)
    fhat = np.fft.fft(snipdata, sample)
    PSD = fhat * np.conj(fhat) / sample
    freq = (1/(.001*sample))*np.arange(sample)
    L = np.arange(1,np.floor(sample/2),dtype='int')
    # if a single value of > 5e5 is found in PSD, return one
    for j in range (0,len(PSD[L])):
        if (PSD[L][j] > 1e8): #ref freq = 1e8
            return 1
    return 0 # if the for loop is completed without returning a 1
    
def calibrate():
    max_low = evaluate_letter(1,2)
    max_high = evaluate_letter(2,3)
    avg_psd = (max_low+max_high) / 2
    return max_low, max_high, avg_psd
    

# run above fn. as many times as there are morse units in the data and plot as a digital signal
digital_array = []
def digitalfft():
    x_axs = []
    digital_array.append(0)
    for r in range(0, int(units)):
        digital_array.append(unit_fft(r,r+1))
        x_axs.append(r)
    digital_array.append(0)
    digiunits = [i for i in range(len(digital_array))]
    plt.step(digiunits,digital_array)
    plt.title("Digital Signal")
    plt.xlabel("units")
    plt.ylabel("Digital Value")
    plt.show()

morse_input = []
decoded_string = ''

def toMorse():
    x = 0
    digital_array.append(0)
    while(x<=len(digital_array)):
        if (x+4 <= len(digital_array) and digital_array[x] == digital_array[x+1] == digital_array[x+2] == 1 and digital_array[x+3] == 0):
            morse_input.append('-')
            x+=4
        elif (x+2 <= len(digital_array) and digital_array[x] == 1 and digital_array[x+1] == 0):
            morse_input.append('.')
            x+=2
        elif (x+6 <= len(digital_array) and digital_array[x] == digital_array[x+1] == digital_array[x+2] == digital_array[x+3] == digital_array[x+4] == digital_array[x+5] == 0):
            morse_input.append('/ /')
            x+=6
        elif (x+2 <= len(digital_array) and digital_array[x] == digital_array[x+1] == 0):
            morse_input.append('/')
            x+=2
        else:
            x+=1


    join = ''
    morse_code = join.join(morse_input)
    #print(morse_code)
    #print("done")
    return (morse_code)
            

morse_table = { '.-': 'A',  
                '-...': 'B',
                '-.-.': 'C', 
                '-..': 'D',  
                '.': 'E',    
                '..-.': 'F', 
                '--.': 'G',  
                '....': 'H', 
                '..': 'I',   
                '.---': 'J', 
                '-.-': 'K',  
                '.-..': 'L', 
                '--': 'M',   
                '-.': 'N',   
                '---': 'O',  
                '.--.': 'P', 
                '--.-': 'Q', 
                '.-.': 'R', 
                '...': 'S',  
                '-': 'T',    
                '..-': 'U',  
                '...-': 'V', 
                '.--': 'W',  
                '-..-': 'X', 
                '-.--': 'Y', 
                '--..': 'Z', 
                '-----': '0',
                '.----': '1',
                '..---': '2',
                '...--': '3',
                '....-': '4', 
                '.....': '5',
                '-....': '6',
                '--...': '7',
                '---..': '8',
                '----.': '9',
                '/': ' ',
                '--..--': ',',
                '.-.-.-': '.',
                '..--..': '?',
                '-..-.': '/',
                '-....-': '-',
                '-.--.': '(',
                '-.--.-': ')',
                ' ': ' '}

def morseToASCII(morse):
    decoded_string = ''
    ascii = morse.split('/')
    for letter in ascii:
        if letter in morse_table:
            decoded_string += morse_table[letter]
        else:
            decoded_string += ''
    return(decoded_string)
    


digitalfft()
print(morseToASCII(toMorse()))

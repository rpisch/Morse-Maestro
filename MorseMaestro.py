from tkinter import *
import customtkinter
import pyaudio
import scipy.io.wavfile as wavf
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import sounddevice as sd
import wave
import datetime
import threading
import mysql.connector
import time
import math
import winsound
import argparse
import queue
import sys
from pydub import AudioSegment
from pydub.playback import play
from scipy.signal import filtfilt
import scipy
import wave
import pyttsx3 as pyttsx


from matplotlib.animation import FuncAnimation
#import matplotlib.pyplot as plt
#import numpy as np
#import sounddevice as sd
# from Note import Note
# from play_tone import Tone

customtkinter.set_appearance_mode("dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"
customtkinter.set_widget_scaling(1.2)

#db = mysql.connector.connect (
#    host="localhost",   # can also use IP address
#    user="root",
#    passwd="root",
#    database="morsemaestro"
#)


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()


        # start/stop boolean
        self.recording = False
        
        # configure MySQL Database
        #self.mycursor = db.cursor()


        # NATO call names: 
        # Alfa, Bravo, Charlie, Delta, Echo, Foxtrot, 
        # Golf, Hotel, India, Juliett, Kilo, Lima, Mike, 
        # November, Oscar, Papa, Quebec, Romeo, Sierra, Tango, 
        # Uniform, Victor, Whiskey, X-ray, Yankee, Zulu

        self.deviceCallName = "Quebec"
        # Libre -> Whiskey
        # Desktop -> Quebec
        # Laptop -> Foxtrot

        # configure window
        self.title("Morse Maestro")
        self.geometry(f"{1100}x{880}")
        #self.geometry(f"{1100}x{580}")

        # configure grid layout (4x4)
        #self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(3, weight=1)
        self.grid_rowconfigure(3, weight=1)
        #self.grid_columnconfigure((2, 3), weight=1)
        #self.grid_rowconfigure((0, 1, 2), weight=1)


        # create textbox
        self.textbox_frame = customtkinter.CTkFrame(self, fg_color="transparent")
        self.textbox_frame.grid(row=0, column=1, columnspan=2, rowspan=2, padx=(20, 20), pady=(20, 20), sticky="nw")
        self.textbox_button = customtkinter.CTkButton(master=self.textbox_frame, text="Translate 📤", command=self.ASCIItoMorse)
        self.textbox_button.grid(row=1, column=1, padx=(20, 20), pady=(20, 20), sticky="ns")
        self.record_button = customtkinter.CTkButton(master=self.textbox_frame, text="Record ⏺️", command=self.MorsetoASCII)   # record button

        self.ASCII_textbox = customtkinter.CTkTextbox(master=self.textbox_frame)     
        self.ASCII_textbox.grid(row=0, column=1, padx=(20, 20), pady=(20, 20), sticky="")
        self.Morse_textbox = customtkinter.CTkTextbox(master=self.textbox_frame)
        self.Morse_textbox.grid(row=0, column=2, padx=(20, 20), pady=(20, 20), sticky="")

        # create audio plot
        self.audio_frame = customtkinter.CTkFrame(self, fg_color="transparent")
        self.audio_frame.grid(row=3, column=1, padx=(5, 5), pady=(5, 5), sticky="we")
        self.audio_frame.grid_forget()  # by default don't show audio plot frame 
 

        # create slider and progressbar frame
        self.slider_progressbar_frame = customtkinter.CTkFrame(self, fg_color="transparent")
        self.slider_progressbar_frame.grid(row=0, column=0, columnspan=1, padx=(20, 0), pady=(20, 0), sticky="nwe")   # frame
        self.slider_progressbar_frame.grid_columnconfigure(0, weight=1)
        self.slider_progressbar_frame.grid_rowconfigure(4, weight=1)
        self.seg_button_1 = customtkinter.CTkSegmentedButton(self.slider_progressbar_frame, command=self.changeButton)
        self.seg_button_1.grid(row=0, column=0, columnspan=1, padx=(20, 10), pady=(10, 10), sticky="news") # A-M / M-A button

        self.slider_1 = customtkinter.CTkSlider(self.slider_progressbar_frame, number_of_steps=30, from_=0.001, to=0.030 , command=self.getWPM)
        self.slider_1.grid(row=1, column=0, columnspan=1, padx=(20, 10), pady=(10, 10), sticky="ews") # WPM slider
        self.label_slider = customtkinter.CTkLabel(master=self.slider_progressbar_frame, text="WPM: " + str(int(self.slider_1.get()*1000)))
        self.label_slider.grid(row=2, column=0, columnspan=1, padx=10, pady=10, sticky="new") # WPM value

        self.ASCII_textbox.insert("0.0", "ASCII Text\n\n" + "abcdefghijklmnopqrstuvwxyz1234567890")
        self.Morse_textbox.insert("0.0", "Morse Text\n\n" + ".- -... -.-. -.. . ..-. --. .... .. .--- -.- .-.. -- -. --- .--. --.- .-. ... - ..- ...- .-- -..- -.-- --.. .---- ..--- ...-- ....- ..... -.... --... ---.. ----. -----")
        self.Morse_textbox.configure(state="disabled")  # by default read only

        self.seg_button_1.configure(values=["ASCII to Morse Code", "Morse Code to ASCII"])
        self.seg_button_1.set("ASCII to Morse Code")


        # create sender/receiver textboxes
        self.callnames_textbox_frame = customtkinter.CTkFrame(self, fg_color="transparent")
        self.callnames_textbox_frame.grid(row=1, column=0, padx=(20,0), sticky="wen")          # frame  
        self.label_sender = customtkinter.CTkLabel(master=self.callnames_textbox_frame, text="Sender Name:")
        self.label_sender.grid(row=1, column=0, padx=(20,10), pady=(10, 0), sticky="w")         # sender label
        self.sender_textbox = customtkinter.CTkTextbox(master=self.callnames_textbox_frame, height=30)     
        self.sender_textbox.grid(row=2, column=0, padx=(20,10), pady=(0,5), sticky="news")      # sender textbox 
        self.sender_textbox.insert("0.0", self.deviceCallName)
        self.label_receiver = customtkinter.CTkLabel(master=self.callnames_textbox_frame, text="Receiver Name:")
        self.label_receiver.grid(row=3, column=0, padx=(20,10), pady=(10, 0), sticky="w")       # receiver label
        self.receiver_textbox = customtkinter.CTkTextbox(master=self.callnames_textbox_frame, height=30)     
        self.receiver_textbox.grid(row=4, column=0, padx=(20,10), pady=(0,5), sticky="news")    # receiver textbox
        self.receiver_textbox.insert("0.0", "Foxtrot")


    def changeButton(self, value):
        if value == "ASCII to Morse Code":
            self.ASCII_textbox.configure(state="normal")
            self.record_button.grid_forget()
            self.audio_frame.grid_forget()
            self.textbox_button.grid(row=1, column=1, padx=(20, 20), pady=(20, 20), sticky="ns")
        elif value == "Morse Code to ASCII":
            self.ASCII_textbox.configure(state="disabled")
            self.textbox_button.grid_forget()
            self.audio_frame.grid(row=3, column=1, padx=(5, 5), pady=(5, 5), sticky="we")
            self.record_button.grid(row=1, column=2, padx=(20, 20), pady=(20, 20), sticky="ns")
    
    def record(self):
            FRAMES_PER_BUFFER = 3200
            FORMAT = pyaudio.paInt16
            CHANNELS = 1
            RATE = 44100
            audio = pyaudio.PyAudio()
            stream = audio.open(format=pyaudio.paInt16, channels=1, rate=44100,
                                input=True, frames_per_buffer=1024)
            frames = []
            start = time.time()
            while self.recording:
                data = stream.read(1024)
                frames.append(data)         ########### THIS IS THE AUDIO DATA TO BE PROCESSED ###########
                passed = time.time() - start
                secs = passed % 60
                mins = passed // 60
                hours = mins // 60
                self.record_button.configure(text=f"Recording...\n{int(hours):02d}:{int(mins):02d}:{int(secs):02d}")
                # self.plot_audio(frames) 
            self.record_button.configure(text="Record ⏺️")
            stream.stop_stream()
            stream.close()
            audio.terminate()
            obj = wave.open('recording.wav', 'wb')
            obj.setnchannels(CHANNELS)
            obj.setsampwidth(audio.get_sample_size(FORMAT))
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
            WPM = int(self.slider_1.get()*1000)
            words, codes = self.Decode(data, sample_rate, WPM, 50, 1000)

            self.Morse_textbox.configure(state="normal")
            self.Morse_textbox.delete("0.0", "end-1c")
            print(codes)
            self.Morse_textbox.insert("0.0", codes)
            self.Morse_textbox.configure(state="disable")

            self.ASCII_textbox.configure(state="normal")
            self.ASCII_textbox.delete("0.0", "end-1c")
            print(words)
            self.ASCII_textbox.insert("0.0", words)
            self.ASCII_textbox.configure(state="disable")


    def play_message(self, bitstream):
        bitstream = '000' + bitstream   # buffer time to begin
        WPM = int(self.slider_1.get()*1000)
        # System Output Sample Frequency
        fs = 48000  # Sampling Frequency (samples/sec)
        dt = 1/fs   # sec/sample
        # Carrier Frequency Parameters
        Freq = 1000                             # Sine wave freq (Hz)
        Unit_time = 1.2/WPM                  # cycles per unit 
        Unit_samples = int(Unit_time*fs)     # Number of samples per unit
        A = np.iinfo(np.int16).max           # Amplitude
        # Input Stream
        Num_Input_Vals = len(bitstream)
        # Total number of generated wav samples
        Total_Samples = int(Num_Input_Vals*Unit_time*fs)
        F = [Freq if i == '1' else 0 for i in bitstream]
        fullF = []
        for i in F:
            if i == Freq:
                for j in range(Unit_samples):
                    fullF.append(Freq)
            else:
                for j in range(Unit_samples):
                    fullF.append(0)
        fullF = np.array(fullF)
        t = np.linspace(0., dt*Total_Samples, len(fullF))
        data = A*np.sin(2. * np.pi * fullF * t)
        sd.play(data, fs)
        #wavf.write('encodedAudio.wav', fs, data.astype(np.int16))
        #audio = AudioSegment.from_wav("encodedAudio.wav")
        #play(audio)
        

    def ASCIItoMorse(self):
        self.Morse_textbox.configure(state="normal")
        self.Morse_textbox.delete("0.0", "end-1c")
        message = self.ASCII_textbox.get("0.0", "end-1c")
        reversed_message = message[::-1]
        a2m = {
            "A":".-",
            "B":"-...",
            "C":"-.-.",
            "D":"-..",
            "E":".",
            "F":"..-.",
            "G":"--.",
            "H":"....",
            "I":"..",
            "J":".---",
            "K":"-.-",
            "L":".-..",
            "M":"--",
            "N":"-.",
            "O":"---",
            "P":".--.",
            "Q":"--.-",
            "R":".-.",
            "S":"...",
            "T":"-",
            "U":"..-",
            "V":"...-",
            "W":".--",
            "X":"-..-",
            "Y":"-.--",
            "Z":"--..",
            "a":".-",
            "b":"-...",
            "c":"-.-.",
            "d":"-..",
            "e":".",
            "f":"..-.",
            "g":"--.",
            "h":"....",
            "i":"..",
            "j":".---",
            "k":"-.-",
            "l":".-..",
            "m":"--",
            "n":"-.",
            "o":"---",
            "p":".--.",
            "q":"--.-",
            "r":".-.",
            "s":"...",
            "t":"-",
            "u":"..-",
            "v":"...-",
            "w":".--",
            "x":"-..-",
            "y":"-.--",
            "z":"--..",
            "1":".----",
            "2":"..---",
            "3":"...--",
            "4":"....-",
            "5":".....",
            "6":"-....",
            "7":"--...",
            "8":"---..",
            "9":"----.",
            "0":"-----",
            " ":" / "
        }
        for letter in reversed_message:
            if letter in a2m:
                self.Morse_textbox.insert("0.0", a2m[letter] + " ")
            else:
                self.Morse_textbox.insert("0.0", letter + " ")
        bitstream = ''
        temp = ''
        for unit in self.Morse_textbox.get("0.0", "end-1c"):
            if (temp == '.' or temp == '-') and (unit == '.' or unit == '-'):
                bitstream += '0'
            if unit == '.':
                bitstream += '1'
            elif unit == '-':
                bitstream += '111'
            elif (temp == '.' or temp == '-' or temp == '/') and unit == ' ':
                bitstream += '000'
            elif temp == ' ' and unit == '/':
                bitstream += '0'
            temp = unit
        # for database:
            # Sender            (data_sender)
            # Receiver          (data_receiver)
            # ASCII Message     (data_message)
            # WPM               (data_WPM)
            # Bitstream         (data_bitstream)
            # Total Time        (data_total_time)
            # Date of Message   (data_date)
            # Time of message   (data_timestamp)
        spu = (6/5) / (float(self.slider_1.get())*1000)
        spu = round(spu, 3)
        data_sender = self.deviceCallName   # Quebec on desktop
        data_receiver = "Receiver_Name"
        data_message = self.ASCII_textbox.get("0.0", "end-1c")
        data_WPM = int(float(self.slider_1.get())*1000)
        data_bitstream = bitstream
        data_total_time = round(spu*len(data_bitstream), 3)
        data_timestamp = datetime.datetime.now()
        data_timestamp = str(data_timestamp)
        data_timestamp = data_timestamp.split()
        data_date = data_timestamp[0]
        data_timestamp = data_timestamp[1]
        data_timestamp = data_timestamp.split(":")
        hour = int(data_timestamp[0])
        minute = int(data_timestamp[1])
        if hour >= 12:
            if hour > 12:
                hour -= 12
            data_timestamp = str(hour) + ":" + str(minute) + "pm"
        else:
            data_timestamp = str(hour) + ":" + str(minute) + "am"
        
        print(f"From: %s \nTo: %s" % (data_sender, data_receiver))
        print(f"Message: %s" % data_message)
        print(f"WPM: %s" % data_WPM)
        print(data_bitstream)
        
        print(f"Rate: %s seconds per unit" % spu)
        print(f"%s units long" % len(bitstream))
        print(f"%s seconds total" % data_total_time)
        print(f"Date: %s" % data_date)
        print(f"Time: %s" % data_timestamp)
        print()
        self.Morse_textbox.configure(state="disabled")


        #self.mycursor.execute(
        #    "INSERT INTO messages (sender, receiver, message, WPM, bitstream, total_time, sent_date, sent_time) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
        #    (data_sender, data_receiver, data_message, data_WPM, data_bitstream, data_total_time, data_date, data_timestamp)
        #)

        #db.commit()

        self.play_message(bitstream)

    def MorsetoASCII(self):
        self.ASCII_textbox.configure(state="normal")
        self.ASCII_textbox.delete("0.0", "end-1c")
        message = self.Morse_textbox.get("0.0", "end-1c")
        words = message.split()
        reverse_words = words[::-1]
        a2m = {
            "A":".-",
            "B":"-...",
            "C":"-.-.",
            "D":"-..",
            "E":".",
            "F":"..-.",
            "G":"--.",
            "H":"....",
            "I":"..",
            "J":".---",
            "K":"-.-",
            "L":".-..",
            "M":"--",
            "N":"-.",
            "O":"---",
            "P":".--.",
            "Q":"--.-",
            "R":".-.",
            "S":"...",
            "T":"-",
            "U":"..-",
            "V":"...-",
            "W":".--",
            "X":"-..-",
            "Y":"-.--",
            "Z":"--..",
            "a":".-",
            "b":"-...",
            "c":"-.-.",
            "d":"-..",
            "e":".",
            "f":"..-.",
            "g":"--.",
            "h":"....",
            "i":"..",
            "j":".---",
            "k":"-.-",
            "l":".-..",
            "m":"--",
            "n":"-.",
            "o":"---",
            "p":".--.",
            "q":"--.-",
            "r":".-.",
            "s":"...",
            "t":"-",
            "u":"..-",
            "v":"...-",
            "w":".--",
            "x":"-..-",
            "y":"-.--",
            "z":"--..",
            "1":".----",
            "2":"..---",
            "3":"...--",
            "4":"....-",
            "5":".....",
            "6":"-....",
            "7":"--...",
            "8":"---..",
            "9":"----.",
            "0":"-----",
            " ":"/"
        }
        m2a = dict([(value, key) for key, value in a2m.items()])
        for word in reverse_words:
            if word in m2a:
                self.ASCII_textbox.insert("0.0", m2a[word] + "")
            else:
                self.ASCII_textbox.insert("0.0", word + " ")

        if self.recording:
            self.recording = False
            self.seg_button_1.configure(state="normal")
            self.slider_1.configure(state="normal")
            self.receiver_textbox.configure(state="normal")
            self.sender_textbox.configure(state="normal")
            self.record_button.configure(fg_color="#1f6aa5")
        else:
            self.recording = True
            self.record_button.configure(fg_color="red")
            self.seg_button_1.configure(state="disabled")
            self.slider_1.configure(state="disabled")
            self.receiver_textbox.configure(state="disabled")
            self.sender_textbox.configure(state="disabled")
            threading.Thread(target=self.record).start()

        # write translation here
        self.ASCII_textbox.configure(state="disabled")
        

        """
        CHUNK = 1024
        FRAMES_PER_BUFFER = 3200
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000

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
            data = stream.read(FRAMES_PER_BUFFER)
            frames.append(data)
            second_tracking += 1
            if second_tracking == RATE/FRAMES_PER_BUFFER:
                second_count += 1
                second_tracking = 0
                print(f'Time left: {seconds - second_count} seconds')

        stream.stop_stream()
        stream.close()
        pa.terminate()

        obj = wave.open('recordedAudio.wav', 'wb')
        obj.setnchannels(CHANNELS)
        obj.setsampwidth(pa.get_sample_size(FORMAT))
        obj.setframerate(RATE)
        obj.writeframes(b''.join(frames))
        obj.close()

        file = wave.open('recordedAudio.wav', 'rb')

        sameple_freq = file.getframerate()
        frames = file.getnframes()
        signal_wave = file.readframes(-1)

        file.close()

        time = frames / sameple_freq

        audio_array = np.frombuffer(signal_wave, dtype=np.int16)

        times = np.linspace(0, time, num=frames)


        fig = Figure(figsize=(12,3))
        ax = fig.add_subplot(111)
        ax.plot(times, abs(audio_array))
        canvas = FigureCanvasTkAgg(fig, master=self.audio_frame)
        canvas.draw()
        #canvas.get_tk_widget().pack()
        canvas.get_tk_widget().grid(row=2, column=0)
        ax.set_title('Audio Plot')
        ax.set_ylabel('Signal Wave')
        ax.set_xlabel('Time (s)')"""

    def getWPM(self, value):
        WPM = round(value, 3)
        self.label_slider.configure(text = "WPM: " + str(int(WPM*1000)))


    def Decode(self, data, sample_rate, words_per_minute, units_per_word, morse_freq):

        #creates a list that stores time in seconds by calculating the current sample number / the sample rate
        time_s = []
        for i in range (0, len(data)):
            time_s.append(i/sample_rate)

        #plt.plot(time_s,data)
        #plt.show()

        reffreq = morse_freq
        range_freq = reffreq*0.1
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

        def trim_data():
            trim1 = bandpass(data)
            index_trim = np.where(trim1>500)[0]
            if(len(index_trim)<2):
                trimmed_data = []
            else:
                index_trim_head = index_trim[0]
                index_trim_tail = index_trim[-1]
                trimmed_data = trim1[index_trim_head:index_trim_tail]
            return trimmed_data

        if (len(trim_data()) <2):
            return("Please Increase Volume! ","Please Increase Volume! ")
        else:
            data_trimmed = trim_data()
        #plt.plot(data_trimmed)
        #plt.show()

        sampletime = (len(data_trimmed))/sample_rate
        WPM = words_per_minute    #select words per minute
        UPW = units_per_word   #50 units per word
        units = round(((UPW/60)*WPM*sampletime),0)
        print(units)

        time_s = []
        for i in range (0, len(data_trimmed)):
            time_s.append(i/sample_rate)

        # Shows the specified letter in the time domain and the frequency domain to evaluate the fft
        def evaluate_letter():
            snipdata = []
            
            # plots a snippet of the original waveform
            start = int((30/units)*(len(data_trimmed)))
            snip = int((31/units)*(len(data_trimmed)))
            snipdata = data_trimmed[start:snip]
                
            fig,axs = plt.subplots(2,1)
            plt.sca(axs[0])
            plt.plot(snipdata)

            # plots the fft
            sample = len(snipdata)
            fhat = np.fft.fft(snipdata, sample)
            PSD = fhat * np.conj(fhat) / sample
            freq = (1/(.001*sample))*np.arange(sample)
            L = np.arange(1,np.floor(sample/2),dtype='int')

            plt.sca(axs[1])
            plt.plot(freq[L], PSD[L])

            plt.show()

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
            #freq = (1/(.001*sample))*np.arange(sample)
            L = np.arange(1,np.floor(sample/2),dtype='int')
            # if a single value of > 5e5 is found in PSD, return one
            for j in range (0,len(PSD[L])):
                if (PSD[L][j] > 1e8): #ref psd 6e7 - 9e7
                    #0 when its supposed to be 1: lower value
                    return 1
            return 0 # if the for loop is completed without returning a 1
            

        # run above fn. as many times as there are morse units in the data and plot as a digital signal
        digital_array = []
        def digitalfft():
            x_axs = []
            for r in range(0, int(units)):
                digital_array.append(unit_fft(r,r+1))
                x_axs.append(r)
            #plt.step(x_axs,digital_array)
            #plt.show()

        morse_input = []

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
                    morse_input.append(' / ')
                    x+=6
                elif (x+2 <= len(digital_array) and digital_array[x] == digital_array[x+1] == 0):
                    morse_input.append(' ')
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
                        ' ': ''}

        def morseToASCII(morse):
            decoded_string = ''
            ascii = morse.split(' ')
            for letter in ascii:
                if letter in morse_table:
                    decoded_string += morse_table[letter]
                else:
                    decoded_string += ''
            return(decoded_string)
            

        # Run fft for all units
        digitalfft()
        #call Morse to ASCII function
        morse_out = toMorse()
        ascii_out = morseToASCII(morse_out)
        #return ASCII output
        return(ascii_out, morse_out)


if __name__ == "__main__":
    app = App()
    app.mainloop()

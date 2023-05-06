Code Created for the Morse Maestro device, meant to run Libre Computer Board system or on a laptop with accurate Morse code audio signals played. Created by Ryan Pischinger and Braden Karley.

This respository has 2 code segments: the full code with interface, encoder, and decoder, and the code to test the decoder which shows plots of the different stages of signal processing.

How to use:
Text to Morse: Enter your message, select a speed, and press translate.
Morse to Text: Select the Morse to ASCII page. Play a Morse code signal at a certain words per minute that the interface slider is set to. Make sure the signal is 1000 Hz. Press the record button before playing the signal and again when the signal is completed. 

This code consists of 4 parts:
	Interface
	Encoder 
	Decoder
	Database
  
	The Interface is run on Custom Tkinter commands and displays a . 
 
 	The Encoder takes a message, converts it into Morse characters via a lookup table, creates an accurate bitstream, and plays the bitstream as a sine wave at the user    set speed.
 
 	The Decoder uses signal processing to convert an audio signal into decoded text. Its steps are:
		Record the audio
		Filter frequencies not equal to 1000 using a bandpass filter
		Trim leading and trailing silence to remove timing offset
		Split signal into single units 
		Convert to frequency domain with FFT
		Create a digitized list with dense frequencies
		Convert to Morse characters
		Convert Morse characters to ASCII characters
		Return text to the interface
  
	The database saves conversation of the encoder and decoder into a string of messages.
  
For more information about Morse code timing visit this link: https://morsecode.world/international/timing.html 
  

from pylab import*
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import math

TYPE = "song"

for j in xrange(1,18):
	
	#sampFreq, snd = wavfile.read('/home/srinidhi/Downloads/speech/'+str(j)+'.wav')
	sampFreq, snd = wavfile.read('../data/speech/'+str(j) +'.wav')
	#sampFreq, snd = wavfile.read('../data/speech/'+str(j) +'.wav')
	#sampFreq, snd = wavfile.read('../Dataset/song/song_ring5.wav')
	#sampFreq, snd = wavfile.read('../Dataset/speech/speech_vasanth.wav')
	#, snd = wavfile.read('../data/speech/6.wav')
	#This part of the code calculates zero detection error
	
	flag = 0
	try :
		snd = snd[:, 0]
	except :
		flag = 1
		
	aver = np.mean(snd)
	mini = np.min(snd)
	maxi = np.max(snd)
		
	cnt = 0
	for i in xrange(1, len(snd),2):
		if (snd[i] < aver  and snd[i-1] > aver) or (snd[i] > aver and snd[i-1] < aver):
			cnt += 1		
	print cnt*1000 / len(snd)
	
	'''
	cnt = [0]
	k = -1
	print mini,maxi,aver
	for i in xrange(1, len(snd), 2):
	    if (snd[i] < aver  and snd[i-1] > aver) or (snd[i] > aver and snd[i-1] < aver):
		        cnt[k] += 1
	    if i % 1000 == 1:
	        k += 1
        	cnt.append(0)
			sum_sq = sum([(i**2) for i in cnt])
	std = sum_sq / (k*1.0)
	std = math.sqrt(std)
	
	plt.hist(cnt)
	plt.show()
	'''
	

#~ Plot to obtain freq domain

#~ snd = snd / (2.**15)
#~ s1 = snd
#~ timeArray = arange(0, 5060.0, 1)
#~ timeArray = timeArray / sampFreq
#~ timeArray = timeArray * 1000  #scale to milliseconds

#~ n = len(s1) 
#~ p = fft(s1) # take the fourier transform 

#~ print p

#~ nUniquePts = ceil((n+1)/2.0)

#~ p = p[0:nUniquePts]
#~ p = abs(p)
#~ p = p / float(n) # scale by the number of points so that
                 #~ # the magnitude does not depend on the length 
                 #~ # of the signal or on its sampling frequency  
#~ p = p**2  # square it to get the power 

#~ print p
#~ # multiply by two (see technical document for details)
#~ # odd nfft excludes Nyquist point
#~ if n % 2 > 0: # we've got odd number of points fft
    #~ p[1:len(p)] = p[1:len(p)] * 2
#~ else:
    #~ p[1:len(p) -1] = p[1:len(p) - 1] * 2 # we've got even number of points fft

#~ freqArray = arange(0, nUniquePts, 1.0) * (sampFreq / n);
#~ #plt.plot(freqArray/1000, 10*log10(p), color='k')
#~ plt.specgram(snd)
#~ xlabel('Frequency (kHz)')
#~ ylabel('Power (dB)')
#~ plt.show()

import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks


def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):

    """ short time fourier transform of audio signal """
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))
    
    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(np.floor(frameSize/2.0)), sig)    
    # cols for windowing
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))
    
    frames = stride_tricks.as_strided(samples, shape=(cols, frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win
    
    return np.fft.rfft(frames)    
    

def logscale_spec(spec, sr=44100, factor=20.):

    """ scale frequency axis logarithmically """    
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))
    
    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            newspec[:,i] = np.sum(spec[:,scale[i]:], axis=1)
        else:        
            newspec[:,i] = np.sum(spec[:,scale[i]:scale[i+1]], axis=1)
    
    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[scale[i]:])]
        else:
            freqs += [np.mean(allfreqs[scale[i]:scale[i+1]])]
    
    return newspec, freqs


def plotstft(audiopath, binsize=2**10, plotpath=None, colormap="jet"):
    
    """ plot spectrogram"""
    samplerate, samples = wav.read(audiopath)
    s = stft(samples, binsize)
    
    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)
    ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel
    
    ims = ims[1:2000,:]
#    print np.shape(ims)
    
    timebins, freqbins = np.shape(ims)
    
    new_ims = []
    
    for i in range(len(ims)):
        maximum = 0
        maximum_index = 0
        #print ims[i]
        val1 = [(j+1)*ims[i][j] for j in range(len(ims[i]))]
        val1 = sum(val1) / sum(ims[i])
        for j in range(len(ims[i])):
            if ims[i][j] > maximum:
                maximum = ims[i][j]
                maximum_index = j
        new_ims.append((j*9 + val1) / 10)
    new_ims = np.array(new_ims)
    return [new_ims.max(), new_ims.min(), new_ims.mean()]  
#    plt.figure(figsize=(15, 7.5))
#    plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
#    plt.colorbar()

#    plt.xlabel("time (s)")
#    plt.ylabel("frequency (hz)")
#    plt.xlim([0, timebins-1])
#    plt.ylim([0, freqbins])

#    xlocs = np.float32(np.linspace(0, timebins-1, 5))
#    plt.xticks(xlocs, ["%.02f" % l for l in ((xlocs*len(samples)/timebins)+(0.5*binsize))/samplerate])
#    ylocs = np.int16(np.round(np.linspace(0, freqbins-1, 10)))
#    plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])
#    
#    if plotpath:
#        plt.savefig(plotpath, bbox_inches="tight")
#    else:
#        plt.show()
#    plt.clf()
for name in range(1,9):
    try:
        value = plotstft("data/song/" + str(name) + ".wav")
        print value[0], value[1]
        value = value[0] - value[1]
        print value
        if value >= 1 and value <= 2.5:
            print "speech"
        else:
            print "song"
    except:
        print "skipping badly"

# calculate mfcc features.

import numpy
from scipy.fftpack import dct

def mfccx(signal,samplerate=16000,winlen=0.025,winstep=0.01,numcep=13,
         nfilt=26,nfft=None,lowfreq=0,highfreq=None,preemph=0.97,ceplifter=22,appendEnergy=True,
         winfunc=lambda x:numpy.ones((x,))):
    """Compute MFCC features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the sample rate of the signal we are working with, in Hz.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param numcep: the number of cepstrum to return, default 13
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is None, which uses the calculate_nfft function to choose the smallest size that does not drop sample data.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param ceplifter: apply a lifter to final cepstral coefficients. 0 is no lifter. Default is 22.
    :param appendEnergy: if this is true, the zeroth cepstral coefficient is replaced with the log of the total frame energy.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy window functions here e.g. winfunc=numpy.hamming
    :returns: A numpy array of size (NUMFRAMES by numcep) containing features. Each row holds 1 feature vector.
    """
    def mfcc(padsignal, frame_len, frame_step, nfft, fb, numcep, ceplifter, appendEnergy, winfunc):
        #Compute MFCC features from an audio signal.
        
        shape = padsignal.shape[:-1] + (padsignal.shape[-1] - frame_len + 1, frame_len)
        strides = padsignal.strides + (padsignal.strides[-1],)
        frames = numpy.lib.stride_tricks.as_strided(padsignal, shape=shape, strides=strides)[::frame_step] * winfunc(frame_len)

        if frames.shape[1] > nfft:
            print('frame length {} is greater than FFT size {}, frame will be truncated. Increase NFFT to avoid.'.format(frames.shape[1], nfft))

        pspec = 1.0 / nfft * numpy.square(numpy.absolute(numpy.fft.rfft(frames, nfft)))
        energy = numpy.sum(pspec,1) # this stores the total energy in each frame
        energy = numpy.where(energy == 0,numpy.finfo(float).eps,energy) # if energy is zero, we get problems with log

        feat = numpy.dot(pspec,fb.T) # compute the filterbank energies
        feat = numpy.where(feat == 0,numpy.finfo(float).eps,feat) # if feat is zero, we get problems with log

        feat = numpy.log(feat)
        feat = dct(feat, type=2, axis=1, norm='ortho')[:,:numcep]

        if ceplifter > 0:
            n = numpy.arange(feat.shape[1])
            feat = (1 + (ceplifter/2.)*numpy.sin(numpy.pi*n/ceplifter)) * feat

        if appendEnergy: feat[:,0] = numpy.log(energy) # replace first cepstral coefficient with log of frame energy
        return feat
    
    if not nfft:
        nfft = 1
        while nfft < winlen * samplerate:
            nfft *= 2
    
    highfreq= highfreq or samplerate/2
    assert highfreq <= samplerate/2, "highfreq is greater than samplerate/2"
    
    # compute points evenly spaced in mels
    lowmel = 2595 * numpy.log10(1+lowfreq/700)
    highmel = 2595 * numpy.log10(1+highfreq/700)

    melpoints = numpy.linspace(lowmel,highmel,nfilt+2)
    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    bin = numpy.floor((nfft+1)* (700*(10**(melpoints/2595)-1))/samplerate)

    fb = numpy.zeros([nfilt,nfft//2+1])
    for j in range(0,nfilt):
        for i in range(int(bin[j]), int(bin[j+1])):
            fb[j,i] = (i - bin[j]) / (bin[j+1]-bin[j])
        for i in range(int(bin[j+1]), int(bin[j+2])):
            fb[j,i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])

    frame_len = numpy.round(winlen*samplerate).astype(int)
    frame_step = numpy.round(winstep*samplerate).astype(int)

    slen = len(signal)

    if slen <= frame_len:
        numframes = 1
    else:
        numframes = 1 + numpy.ceil((slen - frame_len) / frame_step).astype(int)

    padlen = ((numframes - 1) * frame_step + frame_len).astype(int)
    padsignal = numpy.pad(signal, (0, padlen - slen), 'constant', constant_values=(0, 0))
    padsignal = numpy.append(padsignal[0], padsignal[1:] - preemph * padsignal[:-1])
    padsignal[slen] = 0

    return mfcc(padsignal, frame_len, frame_step, nfft, fb, numcep, ceplifter, appendEnergy, winfunc)
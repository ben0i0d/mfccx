# calculate mfcc features.

import numpy

import soundfile as sf

import jax
import jax.numpy as jnp

def mfccx(wav_files,mmap=None,winlen=0.025,winstep=0.01,numcep=13,
         nfilt=26,nfft=None,lowfreq=0,highfreq=None,preemph=0.97,ceplifter=22,appendEnergy=True,
         winfunc=lambda x:jnp.ones((x,))):
    """Compute MFCC features from an audio signal.
    :param wav_files: the audio files list
    :param mmap: use mmap mode to save memory
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
        
        starts = jax.numpy.arange(0, padsignal.shape[-1] - frame_len + 1, frame_step)
        frames = jax.vmap(lambda start: jax.lax.dynamic_slice(padsignal, (start,), (frame_len,)))(starts) * winfunc(frame_len)

        if frames.shape[1] > nfft:
            print('frame length {} is greater than FFT size {}, frame will be truncated. Increase NFFT to avoid.'.format(frames.shape[1], nfft))

        pspec = 1.0 / nfft * jnp.square(jnp.absolute(jnp.fft.rfft(frames, nfft)))
        energy = jnp.sum(pspec,1) # this stores the total energy in each frame
        energy = jnp.where(energy == 0,jnp.finfo(float).eps,energy) # if energy is zero, we get problems with log

        feat = jnp.dot(pspec,fb) # compute the filterbank energies
        feat = jnp.where(feat == 0,jnp.finfo(float).eps,feat) # if feat is zero, we get problems with log

        feat = jnp.log(feat)
        feat = jax.scipy.fft.dct(feat, type=2, axis=1, norm='ortho')[:,:numcep]

        feat = jax.lax.cond(ceplifter > 0, lambda _: (1 + (ceplifter/2.)*jnp.sin(jnp.pi* jnp.arange(feat.shape[1]) /ceplifter)) * feat, lambda _: feat, None) 
        
        # replace first cepstral coefficient with log of frame energy
        return jax.lax.cond(appendEnergy, lambda _: feat.at[:,0].set(jnp.log(energy)), lambda _: feat, None) 
    
    signal,samplerate = sf.read(wav_files[0])
    nfft = nfft or jax.lax.while_loop(lambda nfft: nfft < winlen * samplerate, lambda nfft: nfft * 2, 1)
    highfreq= highfreq or samplerate/2
    assert highfreq <= samplerate/2, "highfreq is greater than samplerate/2"
    
    # compute points evenly spaced in mels
    lowmel = 2595 * jnp.log10(1+lowfreq/700)
    highmel = 2595 * jnp.log10(1+highfreq/700)

    melpoints = jnp.linspace(lowmel,highmel,nfilt+2)
    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    bin = jnp.floor((nfft+1)* (700*(10**(melpoints/2595)-1))/samplerate)

    fb = numpy.zeros([nfilt,nfft//2+1])
    for j in range(0,nfilt):
        for i in range(int(bin[j]), int(bin[j+1])):
            fb[j,i] = (i - bin[j]) / (bin[j+1]-bin[j])
        for i in range(int(bin[j+1]), int(bin[j+2])):
            fb[j,i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])
    fb = jnp.array(fb.T)

    frame_len = jnp.round(winlen*samplerate).astype(int)
    frame_step = jnp.round(winstep*samplerate).astype(int)

    slen = len(signal)

    numframes = jax.lax.cond(slen <= frame_len, lambda _: 1, lambda _: 1 + jnp.ceil((slen - frame_len) / frame_step).astype(int), None) 

    padlen = ((numframes - 1) * frame_step + frame_len).astype(int)
    padsignal = jnp.pad(signal, (0, padlen - slen), 'constant', constant_values=(0, 0))
    padsignal = jnp.append(padsignal[0], padsignal[1:] - preemph * padsignal[:-1]).at[slen].set(0).__array__()

    #if mmap:
    #    cache = numpy.memmap(signal, dtype='float32', mode='r')
    #else:
    #    cache = numpy.zeros((len(wav_files),padlen))

    return mfcc(padsignal, frame_len, frame_step, nfft, fb, numcep, ceplifter, appendEnergy, winfunc)
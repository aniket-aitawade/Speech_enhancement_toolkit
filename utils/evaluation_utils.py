import tensorflow as tf
import librosa
import numpy as np
from omegaconf import OmegaConf

config=OmegaConf.load('/home/skdm/Aniket/Speech Enhancement/config.yaml')

def padding_and_reshape(noisy):
    pad_count=noisy.shape[0]%(config.preprocessing.duration*config.preprocessing.target_sr)
    padding=np.zeros([(config.preprocessing.duration*config.preprocessing.target_sr)-pad_count])
    noisy_=np.concatenate([noisy,padding])
    return np.reshape(noisy_,(-1,(config.preprocessing.duration*config.preprocessing.target_sr)))
        
def feature_extraction(audio_path):
    noisy_waveform,sr=librosa.load(audio_path)
    noisy_waveform=librosa.resample(noisy_waveform,orig_sr=sr,target_sr=config.preprocessing.target_sr)
    audio=padding_and_reshape(noisy_waveform)
    noisy_stft=librosa.stft(audio,n_fft=512,win_length=512,hop_length=256)
    noisy_stft=np.transpose(noisy_stft,axes=[0,2,1])

    if config.preprocessing.abs:
        amp=np.abs(noisy_stft)
        phase=np.angle(noisy_stft)
        amp=np.expand_dims(amp,axis=0)
        return amp,phase,noisy_waveform.shape[0]
    else:
        noisy_stft=np.stack([np.real(noisy_stft),np.imag(noisy_stft)],axis=-1)
        noisy_stft=np.expand_dims(noisy_stft,axis=0)
        return noisy_stft,noisy_waveform.shape[0]

def audio_recontruction(amp,phase,length):
    enhanced=np.squeeze(amp,axis=0)
    enhanced_stft=np.transpose(enhanced*(np.exp(phase*1j)),axes=[0,2,1])
    enhanced=librosa.istft(enhanced_stft,n_fft=512,win_length=512,hop_length=256)
    enhanced=np.reshape(enhanced,(-1))
    enhanced=enhanced[:length]
    return enhanced

def audio_recontruction_complex(stft,length):
    enhanced=np.squeeze(stft,axis=0)
    enhanced=enhanced[:,:,:,0]+1j*enhanced[:,:,:,1]
    enhanced_stft=np.transpose(enhanced,axes=[0,2,1])
    enhanced=librosa.istft(enhanced_stft,n_fft=512,win_length=512,hop_length=256)
    enhanced=np.reshape(enhanced,(-1))
    enhanced=enhanced[:length]
    return enhanced

if __name__=="__main__":
   pass
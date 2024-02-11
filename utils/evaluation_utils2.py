import tensorflow as tf
import librosa
import numpy as np

# def padding_and_reshape(config,noisy):
#     pad_count=noisy.shape[0]%(config.data_module.train.duration*config.data_module.train.target_sr)
#     padding=np.zeros([(config.data_module.train.duration*config.data_module.train.target_sr)-pad_count])
#     noisy_=np.concatenate([noisy,padding])
#     return noisy_

def padd_stft(config,noisy):
    pad_count=noisy.shape[0]%(config.data_module.train.frame_input)
    padding=np.zeros([config.data_module.train.frame_input-pad_count,1+(config.data_module.train.n_fft//2)])
    noisy_=np.concatenate([noisy,padding],axis=0)
    return noisy_

def feature_extraction(config,audio_path):
    noisy_waveform,sr=librosa.load(audio_path)
    noisy_waveform=librosa.resample(noisy_waveform,orig_sr=sr,target_sr=config.data_module.train.target_sr)
    audio=noisy_waveform
    noisy_stft=librosa.stft(audio,
                            n_fft=config.data_module.train.n_fft,
                            win_length=config.data_module.train.frame_length,
                            hop_length=config.data_module.train.hop_length,
                            center=config.data_module.train.center)
    noisy_stft=np.transpose(noisy_stft)
    noisy_stft=padd_stft(config,noisy_stft)
    noisy_stft=np.reshape(noisy_stft,(-1,config.data_module.train.frame_input,1+(config.data_module.train.n_fft//2)))
    # print(noisy_stft.shape)

    if config.data_module.train.abs:
        amp=np.abs(noisy_stft)
        phase=np.angle(noisy_stft)
        amp=np.expand_dims(amp,axis=0)
        return amp,phase,noisy_waveform.shape[0]
    else:
        noisy_stft=np.stack([np.real(noisy_stft),np.imag(noisy_stft)],axis=-1)
        noisy_stft=np.expand_dims(noisy_stft,axis=0)
        return noisy_stft,noisy_waveform.shape[0]

def audio_recontruction(config,amp,phase,length):
    enhanced=np.squeeze(amp,axis=0)
    enhanced_stft=enhanced*(np.exp(phase*1j))
    enhanced_stft=np.reshape(enhanced_stft,(-1,1+(config.data_module.train.n_fft//2)))
    enhanced_stft=np.transpose(enhanced_stft)
    enhanced=librosa.istft(enhanced_stft,
                            n_fft=config.data_module.train.n_fft,
                            win_length=config.data_module.train.frame_length,
                            hop_length=config.data_module.train.hop_length,
                            center=config.data_module.train.center)
    enhanced=enhanced[:length]
    return enhanced

def audio_recontruction_complex(config,stft,length):
    enhanced=np.squeeze(stft,axis=0)
    enhanced_stft=enhanced[:,:,:,0]+1j*enhanced[:,:,:,1]
    enhanced_stft=np.reshape(enhanced_stft,(-1,1+(config.data_module.train.n_fft//2)))
    enhanced_stft=np.transpose(enhanced_stft)
    enhanced=librosa.istft(enhanced_stft,
                            n_fft=config.data_module.train.n_fft,
                            win_length=config.data_module.train.frame_length,
                            hop_length=config.data_module.train.hop_length,
                            center=config.data_module.train.center)
    enhanced=enhanced[:length]
    return enhanced

if __name__=="__main__":
   pass
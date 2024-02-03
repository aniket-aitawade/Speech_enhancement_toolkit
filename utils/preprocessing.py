import tensorflow as tf
from omegaconf import OmegaConf
import librosa
import numpy as np

config=OmegaConf.load('./config.yaml')
class Custom_dataloader(tf.keras.utils.Sequence):
    def __init__(self,noisy,clean):
        self.noisy=noisy
        self.clean=clean
        self.batch=config.preprocessing.batch_size
        self.frame_length=config.preprocessing.frame_length
        self.hop_length=config.preprocessing.frame_step
        self.n_fft=config.preprocessing.n_fft
        self.target_sr=config.preprocessing.target_sr
        self.duration=config.preprocessing.duration
    
    def __len__(self):
        return len(self.noisy)
        
    def audio_loader(self,noisy,clean):
        noisy_audio,sr1=librosa.load(noisy)
        noisy_audio=librosa.resample(noisy_audio,orig_sr=sr1,target_sr=self.target_sr)
        clean_audio,sr2=librosa.load(clean)
        clean_audio=librosa.resample(clean_audio,orig_sr=sr2,target_sr=self.target_sr)
        return noisy_audio,clean_audio
    
    def padding_and_reshape(self,noisy,clean):
        pad_count=noisy.shape[0]%(self.target_sr*self.duration)
        padding=np.zeros([(self.target_sr*self.duration)-pad_count])
        noisy_=np.concatenate([noisy,padding])
        clean_=np.concatenate([clean,padding])
        return np.reshape(noisy_,(-1,(self.target_sr*self.duration))),np.reshape(clean_,(-1,(self.target_sr*self.duration)))
    
    def get_stft(self,noisy,clean):
        noisy_stft=librosa.stft(noisy,n_fft=self.n_fft,win_length=self.frame_length,hop_length=self.hop_length)
        clean_stft=librosa.stft(clean,n_fft=self.n_fft,win_length=self.frame_length,hop_length=self.hop_length)
        noisy_stft=np.transpose(noisy_stft,axes=[0,2,1])
        clean_stft=np.transpose(clean_stft,axes=[0,2,1])

        if config.preprocessing.abs:
          noisy_stft=np.abs(noisy_stft)
          clean_stft=np.abs(clean_stft)
        else:
          noisy_stft=np.stack([np.real(noisy_stft),np.imag(noisy_stft)],axis=-1)
          clean_stft=np.stack([np.real(clean_stft),np.imag(clean_stft)],axis=-1)

        return noisy_stft,clean_stft
    
    # Use for Loading Signle item and use padding batch in tf.data.dataset for batches
    def __getitem__(self,idx):
        noisy_=self.noisy[idx]
        clean_=self.clean[idx]
        noisy_audio,clean_audio=self.audio_loader(noisy_,clean_)       
        noisy_audio,clean_audio=self.padding_and_reshape(noisy_audio,clean_audio) 
        noisy_stft,clean_stft=self.get_stft(noisy_audio,clean_audio)
        return tf.constant(noisy_stft,dtype=tf.float32),tf.constant(clean_stft,dtype=tf.float32)
    
    # def __getitem__(self,idx):
    #     noisy_=self.noisy[idx*self.batch:idx*self.batch+self.batch]
    #     clean_=self.clean[idx*self.batch:idx*self.batch+self.batch]
    #     # noisy_=self.noisy[idx]
    #     # clean_=self.clean[idx]
    #     if config.preprocessing.abs:
    #       shape=[1,int(np.floor(2*self.duration*self.target_sr/self.n_fft)+1),int(self.n_fft/2)+1]
    #     else:
    #       shape=[1,int(np.floor(2*self.duration*self.target_sr/self.n_fft)+1),int(self.n_fft/2)+1,2]
    #     noisy_batch=np.zeros(shape=shape)
    #     clean_batch=np.zeros(shape=shape)
    #     for i in range(self.batch):
    #         noisy_audio,clean_audio=self.audio_loader(noisy_[i],clean_[i])       
    #         noisy_audio,clean_audio=self.padding_and_reshape(noisy_audio,clean_audio) 
    #         noisy_audio,clean_audio=self.get_stft(noisy_audio,clean_audio)
            
    #         noisy_batch=np.concatenate([noisy_batch,noisy_audio],axis=0)
    #         clean_batch=np.concatenate([clean_batch,clean_audio],axis=0)

    #     return tf.constant(noisy_batch,dtype=tf.float32),tf.constant(clean_batch,dtype=tf.float32)
    

if __name__=="__main__":
  pass
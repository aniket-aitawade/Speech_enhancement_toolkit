import tensorflow as tf
import librosa
import numpy as np
import pandas as pd

class Custom_dataloader(tf.keras.utils.Sequence):
    def __init__(self,
                 frame_length:int,
                 hop_length:int,
                 n_fft:int,
                 abs:bool,
                 target_sr:int,
                 duration:int,
                 frame_input:int,
                 center:bool,
                 data_dir:str = "data/train.txt"):
        data=pd.read_csv(data_dir,sep=' ',names=['noisy','clean'])
        self.noisy=data['noisy']
        self.clean=data['clean']
        self.frame_length=frame_length
        self.hop_length=hop_length
        self.n_fft=n_fft
        self.abs=abs
        self.target_sr=target_sr
        self.duration=duration
        self.frame_input=frame_input
        self.center=center
    
    def __len__(self):
        return len(self.noisy)
        
    def audio_loader(self,noisy,clean):
        noisy_audio,sr1=librosa.load(noisy)
        noisy_audio=librosa.resample(noisy_audio,orig_sr=sr1,target_sr=self.target_sr)
        clean_audio,sr2=librosa.load(clean)
        clean_audio=librosa.resample(clean_audio,orig_sr=sr2,target_sr=self.target_sr)
        return noisy_audio,clean_audio
    
    def padd_stft(self,noisy):
        pad_count=noisy.shape[0]%(self.frame_input)
        padding=np.zeros([self.frame_input-pad_count,1+self.n_fft//2])
        noisy_=np.concatenate([noisy,padding],axis=0)
        return noisy_
    
    def get_stft(self,noisy,clean):
        noisy_stft=librosa.stft(noisy,n_fft=self.n_fft,win_length=self.frame_length,hop_length=self.hop_length,center=self.center)
        clean_stft=librosa.stft(clean,n_fft=self.n_fft,win_length=self.frame_length,hop_length=self.hop_length,center=self.center)
        
        noisy_stft=np.transpose(noisy_stft)
        noisy_stft=self.padd_stft(noisy_stft)
        noisy_stft=np.reshape(noisy_stft,(-1,self.frame_input,1+self.n_fft//2))

        clean_stft=np.transpose(clean_stft)
        clean_stft=self.padd_stft(clean_stft)
        clean_stft=np.reshape(clean_stft,(-1,self.frame_input,1+self.n_fft//2))

        if self.abs:
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
        noisy_stft,clean_stft=self.get_stft(noisy_audio,clean_audio)
        return tf.constant(noisy_stft,dtype=tf.float32),tf.constant(clean_stft,dtype=tf.float32)
    
class dataloader():
    def __init__(self,
                 batch_size:int,
                 frame_length:int,
                 hop_length:int,
                 n_fft:int,
                 abs:bool,
                 target_sr:int,
                 duration:int,
                 frame_input:int,
                 center:bool,
                 data_dir:str = "data/train.txt"):
        data=Custom_dataloader(frame_length,hop_length,n_fft,abs,target_sr,duration,frame_input,center,data_dir)
        shape=data[0][0].shape
        if abs:
            shape=[None,shape[1],shape[2]]
        else:
            shape=[None,shape[1],shape[2],shape[3]]

        self.dataset=tf.data.Dataset.from_generator(generator=lambda: (data[i] for i in range(len(data))),
                                                output_signature=((tf.TensorSpec(shape=shape, dtype=tf.float32),
                                                                    tf.TensorSpec(shape=shape, dtype=tf.float32))))
        self.dataset=self.dataset.padded_batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

if __name__=="__main__":
  pass
import tensorflow as tf
import numpy as np
from hydra import compose, initialize


with initialize(version_base=None, config_path="../conf"):
    config = compose(config_name="config")

dft_mat=np.fft.fft(np.eye(config.data_module.train.n_fft))
idft_mat=np.fft.ifft(np.eye(config.data_module.train.n_fft))

class TF_loss(tf.keras.losses.Loss):
  def call(self,y_true,y_pred):
    true=tf.complex(y_true[:,:,:,:,0],y_true[:,:,:,:,1])
    pred=tf.complex(y_pred[:,:,:,:,0],y_pred[:,:,:,:,1])
    
    true_flipped=tf.math.conj(tf.experimental.numpy.flip(true,axis=-1))

    true_full=tf.concat([true,true_flipped[:,:,:,1:-1]],axis=-1)
    true_time=tf.matmul(true_full,idft_mat)
    
    pred_flipped=tf.math.conj(tf.experimental.numpy.flip(pred,axis=-1))
    pred_full=tf.concat([pred,pred_flipped[:,:,:,1:-1]],axis=-1)
    pred_time=tf.matmul(pred_full,idft_mat)

    time_loss=tf.reduce_mean(tf.math.squared_difference(tf.math.real(true_time),tf.math.real(pred_time)))
    freq_loss=tf.reduce_mean(tf.abs(tf.math.subtract(y_true,y_pred)))
    return tf.math.add(4*time_loss,6*freq_loss)
  
class TF_loss_abs_angle(tf.keras.losses.Loss):
  def call(self,y_true,y_pred):
    true=tf.complex(y_true[:,:,:,:,0]*tf.cos(y_true[:,:,:,:,1]),y_true[:,:,:,:,0]*tf.sin(y_true[:,:,:,:,1]))
    pred=tf.complex(y_pred[:,:,:,:,0]*tf.cos(y_pred[:,:,:,:,1]),y_pred[:,:,:,:,0]*tf.sin(y_pred[:,:,:,:,1]))
    
    true_flipped=tf.math.conj(tf.experimental.numpy.flip(true,axis=-1))

    true_full=tf.concat([true,true_flipped[:,:,:,1:-1]],axis=-1)
    true_time=tf.matmul(true_full,idft_mat)
    
    pred_flipped=tf.math.conj(tf.experimental.numpy.flip(pred,axis=-1))
    pred_full=tf.concat([pred,pred_flipped[:,:,:,1:-1]],axis=-1)
    pred_time=tf.matmul(pred_full,idft_mat)

    time_loss=tf.reduce_mean(tf.math.squared_difference(tf.math.real(true_time),tf.math.real(pred_time)))
    freq_loss=tf.reduce_mean(tf.abs(tf.math.subtract(y_true,y_pred)))
    return tf.math.add(4*time_loss,6*freq_loss)
  
class TF_loss_abs(tf.keras.losses.Loss):
  def call(self,y_true,y_pred):
    true=tf.complex(y_true,tf.zeros_like(y_true,dtype=tf.float32))
    pred=tf.complex(y_pred,tf.zeros_like(y_pred,dtype=tf.float32))
    
    true_flipped=tf.math.conj(tf.experimental.numpy.flip(true,axis=-1))

    true_full=tf.concat([true,true_flipped[:,:,:,1:-1]],axis=-1)
    true_time=tf.matmul(true_full,idft_mat)
    
    pred_flipped=tf.math.conj(tf.experimental.numpy.flip(pred,axis=-1))
    pred_full=tf.concat([pred,pred_flipped[:,:,:,1:-1]],axis=-1)
    pred_time=tf.matmul(pred_full,idft_mat)

    time_loss=tf.reduce_mean(tf.math.squared_difference(tf.math.real(true_time),tf.math.real(pred_time)))
    # freq_loss=tf.reduce_mean(tf.abs(tf.math.subtract(y_true,y_pred)))
    freq_loss=tf.reduce_mean(tf.abs(tf.math.subtract(y_true,y_pred)))+tf.reduce_mean(tf.math.squared_difference(y_true,y_pred))
    return tf.math.add(4*time_loss,6*freq_loss)

def scheduler(epoch, lr):
  if epoch < 10:
    return lr
  else:
    return lr * tf.math.exp(-0.1)

def LR_scheduler_callback():
   return tf.keras.callbacks.LearningRateScheduler( scheduler, verbose=1)

class LearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self,lr_decay_factor):
       super().__init__()
       self.lr_decay_factor=lr_decay_factor

    def on_epoch_end(self, epoch, logs=None):
        current_lr = tf.keras.backend.get_value(self.model.optimizer.lr)
        if epoch > 0 and logs['loss'] > self.loss:
            new_lr = current_lr * self.lr_decay_factor
            tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
        else:
           tf.keras.backend.set_value(self.model.optimizer.lr, current_lr*0.98)
        self.loss = logs['loss']

def early_stop_callback():
    return tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        min_delta=0,
        patience=config.training.patience,
        verbose=1,
        mode='auto',
        baseline=None,
        restore_best_weights=False,
    )

def save_latest_model():
   return tf.keras.callbacks.ModelCheckpoint(
    filepath='checkpoints',
    verbose=0,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=False)

def save_best_model():
   return tf.keras.callbacks.ModelCheckpoint(
    filepath='best_model',
    verbose=1,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

class save_best_latest_model_ATT(tf.keras.callbacks.Callback):
    def __init__(self):
       super().__init__()
       self.min_loss=10000

    def on_epoch_end(self, epoch, logs=None):
        if logs['val_g_loss'] <= self.min_loss:
            self.model.generator.save('best_model')
            print(f"\n Loss improved from {self.min_loss} to {logs['val_g_loss']} Model saved in best model")
            self.min_loss = logs['val_g_loss']
        self.model.generator.save('checkpoints')

def tensorboard_callback():
   return tf.keras.callbacks.TensorBoard(log_dir="./tensorboard")

class Warmup_LR(tf.keras.callbacks.Callback):
    def __init__(self, k1=0.2, k2=4e-4, warmup=4000, d_model=32):
        super(Warmup_LR, self).__init__()
        self.k1 = k1
        self.k2 = k2
        self.warmup = warmup
        self.global_step = 0
        self.d_model = d_model

    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        if self.global_step < self.warmup:
           new_lr=self.k1*(self.d_model**-5)*self.global_step*(self.warmup**-1.5)
           tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)

    def on_epoch_end(self, epoch, logs=None):
       if self.global_step > self.warmup:
          new_lr=self.k2*(0.98**(epoch/2))
          tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)

# class validation(tf.keras.callbacks.Callback):
#     def __init__(self):
#         super(validation, self).__init__()
#         file=open('data/val.txt','r')
#         self.val=file.readlines()
#         file.close()
#         # for data in dataset:
#         #     print(data.strip().split(' '))
        
#     def on_epoch_end(self, epoch, logs=None):
#         print("Model Prediction and evaluation is in progress")
#         results=[[],[],[],[],[]]
#         metric=['SSNR','PESQ','CSIG','CBAK','COVL']
#         with alive_bar(len(self.val)) as bar:
#             for audio_path in self.val:
#                 if config.data_module.train.abs:
#                     amp,phase,length=evaluation_utils.feature_extraction(config,audio_path[0])
#                     features=amp
#                 else:
#                     features,length=evaluation_utils.feature_extraction(config,audio_path[0])
#                 model_output=self.model.predict(features,verbose=0)
#                 # model_output=features

#                 if config.data_module.train.abs:
#                     enhanced=evaluation_utils.audio_recontruction(config,model_output,phase,length)
#                 else:
#                     enhanced=evaluation_utils.audio_recontruction_complex(config,model_output,length)

#                 enhanced=enhanced.astype('float32')

#                 clean,sr=librosa.load(audio_path[1])
#                 clean=librosa.resample(clean,orig_sr=sr,target_sr=config.data_module.train.target_sr)
#                 result=sepm.composite(clean_speech=clean,processed_speech=enhanced,fs=config.data_module.train.target_sr)
#                 for i in range(len(result)):
#                     results[i].append(result[i])
#                 bar()

#         mean_results=[]
#         for i in results:
#             mean_results.append(mean(i))

#         for i in range(len(mean_results)):
#             print(f"Mean {metric[i]} ",mean_results[i])

if __name__=="__main__":
  pass
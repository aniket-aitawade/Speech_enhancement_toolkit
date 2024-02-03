import tensorflow as tf
from omegaconf import OmegaConf

config=OmegaConf.load('./config.yaml')

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
        if epoch > 0 and logs['loss'] > self.loss:
            current_lr = tf.keras.backend.get_value(self.model.optimizer.lr)
            new_lr = current_lr * self.lr_decay_factor
            tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
        self.loss = logs['loss']

def early_stop_callback():
    return tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=config.training.patience,
        verbose=1,
        mode='auto',
        baseline=None,
        restore_best_weights=False,
    )

def model_checkpoints_callback():
   return tf.keras.callbacks.ModelCheckpoint(
    filepath='checkpoints/',
    verbose=1,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

def tensorboard_callback():
   return tf.keras.callbacks.TensorBoard(log_dir="./tensorboard")

if __name__=="__main__":
  pass
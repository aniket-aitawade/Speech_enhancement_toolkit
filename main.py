import sys
from omegaconf import OmegaConf
import pandas as pd
import tensorflow as tf
from models import DC_SEtransformer as model
from utils import preprocessing
from utils import utils


print("Python version is",sys.version)
devices=tf.config.list_physical_devices('GPU')
print(devices)
for i in range(len(devices)):
  tf.config.experimental.set_memory_growth(devices[i],True)

config=OmegaConf.load('config.yaml')

# Load Dataset
train=pd.read_csv('data/train.txt',sep=' ',names=['noisy','clean'])
val=pd.read_csv('data/val.txt',sep=' ',names=['noisy','clean'])

dataset_train=preprocessing.Custom_dataloader(train['noisy'],train['clean'])
dataset_val=preprocessing.Custom_dataloader(val['noisy'],val['clean'])

shape=dataset_train[0][0].shape
if config.preprocessing.abs:
  shape=[None,shape[1],shape[2]]
else:
  shape=[None,shape[1],shape[2],shape[3]]

print("dataset shape",shape)

dataloader_train=tf.data.Dataset.from_generator(generator=lambda: (dataset_train[i] for i in range(len(dataset_train))),
                                          output_signature=((tf.TensorSpec(shape=shape, dtype=tf.float32),
                                                             tf.TensorSpec(shape=shape, dtype=tf.float32))))

dataloader_val=tf.data.Dataset.from_generator(generator=lambda: (dataset_val[i] for i in range(len(dataset_val))),
                                          output_signature=((tf.TensorSpec(shape=shape, dtype=tf.float32),
                                                             tf.TensorSpec(shape=shape, dtype=tf.float32))))

processed_dataset_train = dataloader_train.padded_batch(config.preprocessing.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE).cache()
processed_dataset_val = dataloader_val.padded_batch(config.preprocessing.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE).cache()
# processed_dataset_train = dataloader_train.prefetch(buffer_size=tf.data.AUTOTUNE).cache()
# processed_dataset_val = dataloader_val.prefetch(buffer_size=tf.data.AUTOTUNE).cache()

shape=iter(processed_dataset_train).next()[0].shape
print("input shape of model",shape)

#Model Building
SEmodel=model.trainer(config.preprocessing.n_fft)

# processed_dataset_train=processed_dataset_train.take(10)
# processed_dataset_val=processed_dataset_val.take(3)

history=SEmodel.fit(processed_dataset_train,
                      validation_data=processed_dataset_val,
                      batch_size=config.training.batch_size,
                      epochs=config.training.epochs,
                      callbacks=[ utils.model_checkpoints_callback(),
                                  utils.tensorboard_callback(),
                                  utils.early_stop_callback(),
                                  utils.LearningRateScheduler(0.5)])
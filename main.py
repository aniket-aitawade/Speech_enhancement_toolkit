import sys
import hydra
import pandas as pd
from hydra.utils import instantiate
import tensorflow as tf
from utils import utils
import mlflow
from datetime import datetime
import inference

print("Python version is",sys.version)
print("Tensorflow version is",tf.__version__)
devices=tf.config.list_physical_devices('GPU')
print(devices)
for i in range(len(devices)):
  tf.config.experimental.set_memory_growth(devices[i],True)

@hydra.main(config_path="conf",config_name="config",version_base=None)
def main(config):
  dataset=instantiate(config.data_module)
  shape=iter(dataset.train.dataset).next()[0].shape
  print("input shape of model",shape)
  trainer=instantiate(config.model)
  optimizer=instantiate(config.optimizers)
  loss=instantiate(config.loss)
  SEmodel=trainer.pack_model(input_shape=shape,optimizer=optimizer,loss=loss,metrics='mse')

  if config.experiment_name =="SETransformer":
    callbacks=[utils.save_best_model(),utils.save_latest_model(),utils.tensorboard_callback(),utils.early_stop_callback(),utils.LearningRateScheduler(0.5)]
  
  elif config.experiment_name =="ATT":
    callbacks=[utils.save_best_latest_model_ATT(),utils.tensorboard_callback()]
    
  elif config.experiment_name =="DPTFSNET":
    callbacks=[utils.save_best_model(),utils.save_latest_model(),utils.tensorboard_callback(),utils.early_stop_callback(),utils.Warmup_LR()]

  processed_dataset_train=dataset.train.dataset
  processed_dataset_val=dataset.val.dataset
  # processed_dataset_train=processed_dataset_train.take(5)
  # processed_dataset_val=processed_dataset_val.take(1)
  history=SEmodel.fit(processed_dataset_train,
                        validation_data=processed_dataset_val,
                        batch_size=config.training.batch_size,
                        epochs=config.training.epochs,
                        callbacks=callbacks)
  
  mean_results_ENG=inference.test_model(path='data/test.txt',header='English')
  mean_results_Hindi=inference.test_model(path='data/test_hindi.txt',header='Hindi')
  metric=['SSNR','PESQ','CSIG','CBAK','COVL']
  mlflow.set_experiment(config.experiment_name)
  mlflow.set_tag('mlflow.runName', config.experiment_name+'_'+datetime.now().strftime("%d%m%Y_%H%M"))
  mlflow.log_param("n_fft",config.data_module.train.n_fft)
  mlflow.log_param("frame_length",config.data_module.train.frame_length)
  mlflow.log_param("hop_length",config.data_module.train.hop_length)
  mlflow.log_param("abs",config.data_module.train.abs)
  mlflow.log_param("frame_input",config.data_module.train.frame_input)
  mlflow.log_param("epochs",config.training.epochs)
  mlflow.log_param("Optimizer",config.optimizers)
  mlflow.log_param("Loss Function",config.loss)
  mlflow.log_param("Model details",config.model)
  mlflow.log_artifact('best_model')
  for i in range(len(metric)): 
    mlflow.log_metric(f'English_{metric[i]}', mean_results_ENG[i])
    mlflow.log_metric(f'Hindi_{metric[i]}', mean_results_Hindi[i])

if __name__=="__main__":
  main()
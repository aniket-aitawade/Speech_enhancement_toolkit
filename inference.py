import argparse
import pandas as pd
import tensorflow as tf
from utils import evaluation_utils2 as evaluation_utils
import soundfile as sf
from alive_progress import alive_bar
from hydra import compose, initialize
from utils import sepm
from statistics import mean
import librosa
import numpy as np


with initialize(version_base=None, config_path="./conf"):
    config = compose(config_name="config")

parser=argparse.ArgumentParser()
parser.add_argument("--path",default="data/test.txt")
parser.add_argument("--header",default="English")
args=parser.parse_args()

devices=tf.config.list_physical_devices('GPU')
for i in range(len(devices)):
  tf.config.experimental.set_memory_growth(devices[i],True)

def test_model(path=args.path,header=args.header):
  test=pd.read_csv(path,sep=' ',names=['noisy','clean']).to_numpy()
  # test=test[:2]
  
  #Load Model
  model=tf.keras.models.load_model('./best_model',compile=False)

  print("Model Prediction and evaluation is in progress")
  results=[[],[],[],[],[]]
  metric=['SSNR','PESQ','CSIG','CBAK','COVL']
  with alive_bar(len(test)) as bar:
    for audio_path in test:
      if config.data_module.train.abs:
        amp,phase,length=evaluation_utils.feature_extraction(config,audio_path[0])
        features=amp
      else:
        features,length=evaluation_utils.feature_extraction(config,audio_path[0])
      model_output=model.predict(features,verbose=0)
      # model_output=features

      if config.data_module.train.abs:
        enhanced=evaluation_utils.audio_recontruction(config,model_output,phase,length)
      else:
        enhanced=evaluation_utils.audio_recontruction_complex(config,model_output,length)

      enhanced=enhanced.astype('float32')

      clean,sr=librosa.load(audio_path[1])
      clean=librosa.resample(clean,orig_sr=sr,target_sr=config.data_module.train.target_sr)
      if header=='Hindi':
        enhanced=enhanced[:clean.shape[0]]

      result=sepm.composite(clean_speech=clean,processed_speech=enhanced,fs=config.data_module.train.target_sr)
      for i in range(len(result)):
        results[i].append(result[i])
      output=audio_path[0].split('/')[-1]
      sf.write(f'results/{output}',enhanced,config.data_module.train.target_sr)
      bar()

  mean_results=[]
  for i in results:
    mean_results.append(mean(i))

  for i in range(len(mean_results)):
    print(f"Mean {header}_{metric[i]} Score is",mean_results[i])
  
  return mean_results

if __name__=="__main__":
  test_model()


from omegaconf import OmegaConf
import pandas as pd
import librosa
import tensorflow as tf
import evaluation_utils
import soundfile as sf
import sepm
import pesq
from statistics import mean
from alive_progress import alive_bar

devices=tf.config.list_physical_devices('GPU')
for i in range(len(devices)):
  tf.config.experimental.set_memory_growth(devices[i],True)

config=OmegaConf.load('./config.yaml')

# Load Dataset
test=pd.read_csv('data/test.txt',sep=' ',names=['noisy','clean']).to_numpy()

#Load Model
# model=tf.keras.models.load_model('./pretrained_model')
model=tf.keras.models.load_model('./checkpoints')
print(model.summary())

print("Model Prediction is in progress")
with alive_bar(len(test)) as bar:
  for audio_path in test:
    if config.preprocessing.abs:
      amp,phase,length=evaluation_utils.feature_extraction(audio_path[0])
      features=amp
    else:
      features,length=evaluation_utils.feature_extraction(audio_path[0])

    model_output=model.predict(features,verbose=0)

    if config.preprocessing.abs:
      enhanced=evaluation_utils.audio_recontruction(model_output,phase,length)
    else:
      enhanced=evaluation_utils.audio_recontruction_complex(model_output,length)
    output=audio_path[0].split('/')[-1]
    sf.write(f'results/{output}',enhanced,config.preprocessing.target_sr)
    bar()

pesq_wb=[]
print("Metric Evaluation is in progress")
with alive_bar(len(test)) as bar:
  for audio_path in test:
    clean,sr=librosa.load(audio_path[1])
    clean=librosa.resample(clean,orig_sr=sr,target_sr=config.preprocessing.target_sr)
    enhanced_path=audio_path[1].split('/')[-1]
    enhanced,sr=librosa.load(f'./results/{enhanced_path}')
    enhanced=librosa.resample(enhanced,orig_sr=sr,target_sr=config.preprocessing.target_sr)
    results=pesq.pesq(config.preprocessing.target_sr,clean,enhanced)
    pesq_wb.append(results)
    bar()
print("Mean PESQ score is",mean(pesq_wb))


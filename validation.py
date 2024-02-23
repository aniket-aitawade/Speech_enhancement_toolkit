import tensorflow as tf
from hydra import compose, initialize
from alive_progress import alive_bar
from utils import evaluation_utils2 as evaluation_utils
import librosa
from utils import sepm
from statistics import mean



with initialize(version_base=None, config_path="conf"):
    config = compose(config_name="config")
class validation(tf.keras.callbacks.Callback):
    def __init__(self):
        super(validation, self).__init__()
        file=open('data/val.txt','r')
        data=file.readlines()
        file.close()
        self.val=[line.strip().split(' ') for line in data]
        
    def on_epoch_end(self, epoch, logs=None):
        if epoch%5 != 0:
            return
        print("Model Prediction and evaluation is in progress")
        results=[[],[],[],[],[]]
        metric=['SSNR','PESQ','CSIG','CBAK','COVL']
        with alive_bar(len(self.val)) as bar:
            for audio_path in self.val:
                if config.data_module.train.abs:
                    amp,phase,length=evaluation_utils.feature_extraction(config,audio_path[0])
                    features=amp
                else:
                    features,length=evaluation_utils.feature_extraction(config,audio_path[0])
                model_output=self.model.predict(features,verbose=0)
                # model_output=features

                if config.data_module.train.abs:
                    enhanced=evaluation_utils.audio_recontruction(config,model_output,phase,length)
                else:
                    enhanced=evaluation_utils.audio_recontruction_complex(config,model_output,length)

                enhanced=enhanced.astype('float32')

                clean,sr=librosa.load(audio_path[1])
                clean=librosa.resample(clean,orig_sr=sr,target_sr=config.data_module.train.target_sr)
                result=sepm.composite(clean_speech=clean,processed_speech=enhanced,fs=config.data_module.train.target_sr)
                for i in range(len(result)):
                    results[i].append(result[i])
                bar()

        mean_results=[]
        for i in results:
            mean_results.append(mean(i))

        result_string=''
        for i in range(len(mean_results)):
            result_string+=f"{metric[i]} : {mean_results[i]} "
        print(result_string)
        with open('logs/log.txt','a') as f:
            f.write('epoch '+str(epoch)+' '+result_string+'\n')
import os
from omegaconf import OmegaConf
import pandas as pd

config=OmegaConf.load('./config.yaml')

print("Data Preparation Stage Started")
audios_train=os.listdir(os.path.join(config.directory,'train','noisy'))
audios_test=os.listdir(os.path.join(config.directory,'test','noisy'))

train_set=audios_train[:int(len(audios_train)*config.train_val_split)]
val_set=audios_train[int(len(audios_train)*config.train_val_split):]

train_set={'noisy':[os.path.join(config.directory,'train','noisy',i) for i in train_set],
            'clean':[os.path.join(config.directory,'train','clean',i) for i in train_set]}

val_set={'noisy':[os.path.join(config.directory,'train','noisy',i) for i in val_set],
            'clean':[os.path.join(config.directory,'train','clean',i) for i in val_set],}

test_set={'noisy':[os.path.join(config.directory,'test','noisy',i) for i in audios_test],
            'clean':[os.path.join(config.directory,'test','clean',i) for i in audios_test],}

train=pd.DataFrame(train_set)
val=pd.DataFrame(val_set)
test=pd.DataFrame(test_set)
train.to_csv('data/train.txt',sep=' ',index=False,header=False)
val.to_csv('data/val.txt',sep=' ',index=False,header=False)
test.to_csv('data/test.txt',sep=' ',index=False,header=False)
print("Data Preparation Stage Completed")

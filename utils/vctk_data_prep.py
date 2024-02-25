import os
import pandas as pd
import argparse

parser=argparse.ArgumentParser()
parser.add_argument("--vctk_path",default="/home/aniket/ANIKET/Datasets/vctk")
parser.add_argument("--output_dir",default="data")

args=parser.parse_args()
directory=args.vctk_path
output_dir=args.output_dir
val_spk=[]#["p226","p287"]

print("Data Preparation for VCTK Started")
audios_train=os.listdir(os.path.join(directory,'noisy_trainset_28spk_wav'))
test_set=os.listdir(os.path.join(directory,'noisy_testset_wav'))

val_set=[audio for audio in audios_train if audio.split('_')[0] in val_spk]
train_set=[audio for audio in audios_train if audio.split('_')[0] not in val_spk]

w1=open(os.path.join(output_dir,'train.txt'),'w+')
w2=open(os.path.join(output_dir,'val.txt'),'w+')
w3=open(os.path.join(output_dir,'test.txt'),'w+')

for audios in train_set:
    w1.write(os.path.join(directory,'noisy_trainset_28spk_wav',audios)+' '+os.path.join(directory,'clean_trainset_28spk_wav',audios)+'\n')
for audios in val_set:
    w2.write(os.path.join(directory,'noisy_trainset_28spk_wav',audios)+' '+os.path.join(directory,'clean_trainset_28spk_wav',audios)+'\n')
for audios in test_set:
    w3.write(os.path.join(directory,'noisy_testset_wav',audios)+' '+os.path.join(directory,'clean_testset_wav',audios)+'\n')

w1.close()
w2.close()
w3.close()

print("Data Preparation for VCTK Completed")

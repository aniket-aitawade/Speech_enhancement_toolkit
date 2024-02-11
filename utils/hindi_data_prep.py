import os
import argparse

parser=argparse.ArgumentParser()

parser.add_argument("--path",default='/home/aniket/ANIKET/Aniket/Datasets/hindi_IIT_madras')
parser.add_argument("--output_path",default='data/test_hindi.txt')

args=parser.parse_args()
directory=args.path
output_path=args.output_path

print("Data Preparation for Hindi Started")
test_set=os.listdir(os.path.join(directory,'noisy'))

w1=open(output_path,'w+')

for audios in test_set:
    w1.write(os.path.join(directory,'noisy',audios)+' '+os.path.join(directory,'clean',audios)+'\n')
w1.close()
print("Data Preparation for Hindi Completed")
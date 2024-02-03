mkdir data results
python3 utils/data_prep.py 2> log.txt
python3 main.py 2>> log.txt
cp -r checkpoints $(date +%d_%m_%y_%H_%M)_trained
python3 utils/evaluation.py 2>> log.txt
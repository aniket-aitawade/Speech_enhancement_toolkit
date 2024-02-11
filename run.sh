vctk_path=/home/aniket/ANIKET/Datasets/vctk
hindi_path=/home/aniket/ANIKET/Datasets/IndicTTS_DEMAND
test_hindi_path=data/test_hindi.txt
test_eng_path=data/test.txt

mkdir data results
python3 utils/vctk_data_prep.py --vctk_path $vctk_path --output_dir data
python3 utils/hindi_data_prep.py --path $hindi_path --output_path $test_hindi_path
python3 main.py 2> log.txt
cp -r best_model $(date +%d_%m_%y_%H_%M)_SETransformer


# python3 inference.py --path $test_eng_path --header English 2>> log.txt
# python3 inference.py --path $test_hindi_path --header Hindi 2>> log.txt
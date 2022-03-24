# Readme

This repo is the source code for paper **[Biomedical Concept Normalization by Leveraging Hypernyms](https://aclanthology.org/2021.emnlp-main.284.pdf)**.

## Preprocessing

### Dictionary Preprocessing

- Download the CTD vocabulary files from [CTD](http://ctdbase.org/downloads), we used the xml file `CTD_diseases.xml.gz` (v2021.02.01).
- Preprocess the CTD vocabulary files and  format to JSON.

```bash
python ./datasets/data_preprocess/preprocess_CDT.py \
    --disease_xml ./datasets/raw_data/CTD_diseases_MEDIC_2021.02.01.xml \
    --output_dir ./datasets/NCBI_Disease
```

### Raw Dataset Preprocessing

- Download the dataset files from [NCBI Disease Corpus (Complete Train / Development / Test set)](https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/), then unzip the files and get the `NCBItrainset_corpus.txt`, `NCBIdevelopset_corpus.txt`, `NCBItestset_corpus.txt`.
- Preprocess the dataset files to JSON files.

```bash
python ./datasets/data_preprocess/preprocess_NCBI.py \
    --train_txt ./datasets/raw_data/NCBItrainset_corpus.txt \
    --dev_txt ./datasets/raw_data/NCBIdevelopset_corpus.txt \
    --test_txt ./datasets/raw_data/NCBItestset_corpus.txt \
    --output_dir ./datasets/raw_data
```

- Preprocess JSON files with lowercasing, abbreviation etc.

```bash
python ./datasets/data_preprocess/preprocess_dataset.py \
    --input_file ./datasets/raw_data/NCBItrainset_corpus.json \
    --output_dir ./datasets/NCBI_Disease/processed_train \
    --ab3p_path ./Ab3P/identify_abbr \
    --dictionary_path ./datasets/NCBI_Disease/CTD_diseases_MEDIC_2021.02.01.json \
    --lowercase true \
    --remove_punctuation true

python ./datasets/data_preprocess/preprocess_dataset.py \
    --input_file ./datasets/raw_data/NCBIdevelopset_corpus.json \
    --output_dir ./datasets/NCBI_Disease/processed_dev \
    --ab3p_path ./Ab3P/identify_abbr \
    --dictionary_path ./datasets/NCBI_Disease/CTD_diseases_MEDIC_2021.02.01.json \
    --lowercase true \
    --remove_punctuation true

python ./datasets/data_preprocess/preprocess_dataset.py \
    --input_file ./datasets/raw_data//NCBItestset_corpus.json \
    --output_dir ./datasets/NCBI_Disease/processed_test \
    --ab3p_path ./Ab3P/identify_abbr \
    --dictionary_path ./datasets/NCBI_Disease/CTD_diseases_MEDIC_2021.02.01.json \
    --lowercase true \
    --remove_punctuation true
```

- Preprocess the extended dictionary.

```bash
# Note that the only difference between the dictionaries is that test_dictionary includes train mentions to increase the coverage.

python ./datasets/data_preprocess/preprocess_dictionary.py \
    --input_dictionary_path ./datasets/NCBI_Disease/CTD_diseases_MEDIC_2021.02.01.json \
    --output_dictionary_path ./datasets/NCBI_Disease/train_dictionary.txt \
    --lowercase true \
    --remove_punctuation true

python ./datasets/data_preprocess/preprocess_dictionary.py \
    --input_dictionary_path ./datasets/NCBI_Disease/CTD_diseases_MEDIC_2021.02.01.json \
    --additional_data_dir ./datasets/NCBI_Disease/processed_train \
    --output_dictionary_path ./datasets/NCBI_Disease/dev_dictionary.txt \
    --lowercase true \
    --remove_punctuation true

python ./datasets/data_preprocess/preprocess_dictionary.py \
    --input_dictionary_path ./datasets/NCBI_Disease/CTD_diseases_MEDIC_2021.02.01.json \
    --additional_data_dir ./datasets/NCBI_Disease/processed_train \
                          ./datasets/NCBI_Disease/processed_dev \
    --output_dictionary_path ./datasets/NCBI_Disease/test_dictionary.txt \
    --lowercase true \
    --remove_punctuation true

# mkdir
mkdir ./datasets/NCBI_Disease/processed_train_dev
cp ./datasets/NCBI_Disease/processed_train/* ./datasets/NCBI_Disease/processed_train_dev
cp ./datasets/NCBI_Disease/processed_dev/* ./datasets/NCBI_Disease/processed_train_dev
```

## Train Model

Use the following command to train the model.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --bert_dir ./pretrained/pt_biobert1.1/ \
    --model_dir exp/BCNH \
    --train_dictionary_path ./datasets/NCBI_Disease/train_dictionary.txt \
    --train_dir ./datasets/NCBI_Disease/processed_train_dev \
    --dev_dictionary_path ./datasets/NCBI_Disease/dev_dictionary.txt \
    --dev_dir ./datasets/NCBI_Disease/processed_dev \
    --test_dictionary_path ./datasets/NCBI_Disease/test_dictionary.txt \
    --test_dir ./datasets/NCBI_Disease/processed_test \
    --epoch 10 \
    --hyper_num 10 \
    --hyper_norm_scale 1 \
    --taxonomy ./datasets/NCBI_Disease/CTD_diseases_MEDIC_2021.02.01.json
```

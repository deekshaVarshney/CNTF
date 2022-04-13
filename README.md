# CNTF: Commonsense, Named Entity and Topical Knowledge Fused Neural Network

This is an implementation of our paper:

Commonsense and Named Entity Aware Knowledge Grounded Dialogue Generation (NAACL 2022).
*Deeksha Varshney, Akshara Prabhakar, Asif Ekbal*

## Overview 
The code and sample data for our work is organized as:
- ```source/``` contains the main model scripts
- ```sample_processed_data/``` has a small sample of the Wizard of Wikipedia dataset files after preprocessing
- ```tools/``` has the evaluation scripts

## Requirements
1. The implementation is based on Python 3.x. To install the dependencies used, run:
```
$ pip install -r requirements.txt
```
2. Save the pretrained Glove embeddings (glove.840B.300d.txt).

## Training
For training CNTF, run the following command. All available parameters and flags have been described in main.py. Please refer to paper appendix for further details.
```
$ CUDA_VISIBLE_DEVICES=x python main.py --data_dir ./sample_processed_data/wiz/ --save_dir ./sample_model/wiz --embed_size 300 --embed_file <path to Glove> --batch_size 8 --vocab_type dlg-seen --bert_config ./source/model/wiz_config.json --num_epochs 30 --log_steps 2
```

## Testing
For testing CNTF, run the following command:
```
$ CUDA_VISIBLE_DEVICES=x python main.py --data_dir ./sample_processed_data/wiz/ --save_dir ./sample_model/wiz --vocab_type dlg-seen --bert_config ./source/model/wiz_config.json --batch_size 32 --output_dir ./outputs/wiz --test
```

## Evaluation
To evaluate CNTF, run the following command:
```
$ cd tools/
$ python eval.py --eval_dir ../outputs/wiz --pred_file pred --ref_file tgt
```
This will generate **result.txt** which will have all the stated metrics.

To get the predicted responses of the model and gold responses in csv format, run:
```
$ python read.py --dir ../outputs/wiz
```
This will generate the required **output.csv**
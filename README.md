# CTKMN
The code and sample data for our work is organized as:
- ```source/``` contains the main model scripts
- ```sample_processed_data/``` has a small sample of the dataset files after preprocessing
- ```tools/``` has the evaluation script
- ```sample_model/``` contains some training logs and model parameters file for the model built on the **sample data**

## Requirements
The implementation is based on Python 3.x. To install the dependencies used, please run:
```
pip install -r requirements.txt
```

## Training
For training CTKMN, please run the following command. All available parameters and flags have been described in main.py.
```
CUDA_VISIBLE_DEVICES=x python main.py --data_dir ./sample_processed_data/wizard/ --save_dir ./sample_model/wiz --embed_size 300 --batch_size 8 --share_vocab False --vocab_type dlg-shared-seen --num_epochs 30 --bidirectional True --log_steps 2
```

## Testing
For testing CTKMN, please run the following command:
```
CUDA_VISIBLE_DEVICES=x python main.py --data_dir ./sample_processed_data/wizard/ --save_dir ./sample_model/wiz --vocab_type dlg-shared-seen --share_vocab False --batch_size 32 --output_dir ./outputs/wiz --test --bidirectional True --ckpt best.model
```

## Evaluation
To evaluate CTKMN, please run the following command inside ```tools/```:
```
cd tools/
python eval.py --eval_dir ../outputs/wiz --pred_file pred --ref_file tgt
```
This will generate **result.txt** which will have all the stated metrics.

To get the predicted responses of the model and gold responses, please run:
```
python read.py --dir ../outputs/wiz
```
This will generate the required **output.csv**.

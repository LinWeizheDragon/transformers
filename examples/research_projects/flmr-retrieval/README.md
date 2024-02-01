Example Use of FLMR

## Environment

Create virtualenv:
```
conda create -n FLMR python=3.8
conda activate FLMR
```
Install Pytorch:
```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```
Install transformers from this folder
```
cd ../../..
pip install -e .
```
Install ColBERT engine
```
cd third_party/ColBERT
pip install -e .
```

Install other dependencies
```
pip install ujson gitpython easydict ninja
```

## Use PreFLMR
```
cd transformers/examples/research_projects/flmr-retrieval/
```
```
python example_use_preflmr.py \
            --use_gpu \
            --index_root_path "." \
            --index_name EVQA_PreFLMR_ViT-G \
            --experiment_name EVQA \
            --indexing_batch_size 64 \
            --image_root_dir /rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/EVQA/eval_image/ \
            --dataset_path LinWeizheDragon/EVQA_PreFLMR_preprocessed_data \
            --passage_dataset_path LinWeizheDragon/EVQA_PreFLMR_preprocessed_passages \
            --use_split test \
            --nbits 8 \
            --Ks 1 5 10 20 50 100 \
            --checkpoint_path LinWeizheDragon/PreFLMR_ViT-G \
            --image_processor_name laion/CLIP-ViT-bigG-14-laion2B-39B-b160k \
            --query_batch_size 8 \
```
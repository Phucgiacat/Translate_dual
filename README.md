# docNMT
- docs: cac tai lieu tham khao
- papers: cac paper dang viet
- codes: code dang su dung
### Dataset
The IWLST'15 English-Vietnamese, which is from [Standford NLP Group](https://nlp.stanford.edu/projects/nmt/). \
Devset and testset is _tst2012_ and _tst2013_, respectively.

|Dataset | Sentences 
| ---------- | ---------- |
| Training | 131,263 |
| Development | 1528 |
| Testing | 1253 |
### AMR Dataset
We use [NeuralAMR](https://github.com/sinantie/NeuralAmr) to create AMR(anonymized format) from datasets and filtered out sentences which is incorrect. \
We also clean AMR by removing number after nodes and edges to reduce vocabulary. For examples: \
``story :arg1-of ( finish-01 :polarity - ) :domain this`` \
will be cleaned as:\
``story :arg-of ( finish :polarity not ) :domain this`` \
You can find the cleaning script [here](./codes/fairseq_g/fairseq_cli/clean_amr.py). \
For reading linearized amr, we modified the reading code of [this paper](https://github.com/freesunshine0316/neural-graph-to-seq-mp).
### Main results:
**uni-L**STM + **A**MREncoder: uni-LA \
**bi-L**STM + **A**MREncoder: bi-LA \
MaxPooling: MP \
MeanAggregator: MA \
GCN-Aggregator: GCN-A 

Use MaxPooling as default setting for AMREncoder. \
Baseline: uni-LSTM: 26.37 [Google Drive](https://drive.google.com/drive/folders/1Utnd6zgodgFOEsmScjNwON30SN2Q5e6C?usp=sharing) 

| Model   | # graph layers| with highway (#layer=1) | w/o highway layer (#layer=0)
| ----------- | ----------- | --------- | ------- |  
| uni-LA + MP  | 1 |   26.59  | 26.63 |
| **uni-LA + MP**  | 2|  **27.38** | **27.35** | 
| uni-LA + MP  | 3| 26.72  | 26.53 | 
| uni-LA + MP  | 4| 26.99  | 24.98 | 
| uni-LA + MP  | 5| 24.62  | 26.88 |
Models of uni-LA are available [here](https://drive.google.com/drive/folders/1qIFAqm-F-dwJOhnsbrQv1ves-9ivQFLE?usp=sharing). 

Further experiments with bi-LSTM:
| bi-LSTM ([baseline](https://drive.google.com/drive/folders/117vbWXh9RBM9GfNMJet9dFE4SxZGHzYH?usp=sharing)): 28.49 | 


| Model   | # graph layers| with highway (#layer=1) | w/o highway layer (#layer=0) 
| ----------- | ----------- | --------- | ------- | 
| bi-LA - MP | 1| 28.31 | 28.10 |
| bi-LA - MP | 2 | **28.94** | 28.40 
| bi-LA - MP | 3 | 28.16 | 28.10 
| bi-LA + MP | 4| 28.37  | 28.58 
| bi-LA + MP | 5| 28.19  | **28.66** 
Models of bi-LA are available [here](https://drive.google.com/drive/folders/1q3SnpYZhGvOGORJRYxsMKiDVRyt8EvYa?usp=sharing). 

Investigating the impact of aggregators ([models](https://drive.google.com/drive/folders/1dSJBNE1PPPHNEeFTcDod_AJMTs6LAZ9z?usp=sharing))

| Model   | MA| MP | GCN-A  
| ------ | ---- | ----- | ----- 
|uni-LA | 26.68 | **27.38** | 26.23 
|bi-LA | 28.66 | **28.94** | 28.67  

All of these model are avalable [here](https://drive.google.com/drive/folders/1uXPpS1MLo6nSfZ3bp37DtNjTzB1vqhg8?usp=sharing).
 
 - Config as below (GPU TESLA P100-PCIE-16GB):
 ```bash
    !CUDA_VISIBLE_DEVICES=0 python3 ./train.py "preprocessed" --arch lstm \
                                                              --with_amr True \
                                                              --aggr maxpooling \
                                                              --graph-encoder-embed-dim 128 \
                                                              --graph-hidden-dim 256 \
                                                              --graph-in-dropout 0.2 \
                                                              --concat-in-aggr "True" \
                                                              --n-highways 1 \
                                                              --n-graph-layers 2 \
                                                              --direction bi \
                                                              --lr 0.001 \
                                                              --clip-norm 0.1 \
                                                              --dropout 0.2 \
                                                              --max-tokens 3500 \
                                                              --criterion label_smoothed_cross_entropy \
                                                              --optimizer adam \
                                                              --label-smoothing 0.1 \
                                                              --lr-scheduler fixed \
                                                              --force-anneal 200 \
                                                              --save-dir "checkpointsgraph/lstm" \
                                                              --max-epoch 10 \
                                                              --log-format tqdm \
                                                              --log-interval 100 
```
**NB:** ``--graph-hidden-dim`` should be **always twice** as ``--graph-encoder-embed-dim``.\
The same for w/o AMR.
Find the usage [here](./codes/README.md).
### In progress:
- Cleaning AMR: \
Ex: "give-01" -> "give" (**DONE**)


## Usage
Similar with [fairseq](https://github.com/pytorch/fairseq).
Command-line arguments and passing are also the same but with some changes.
- To install this editable version:
```bash
    cd Thesis
    sudo pip3 install --editable fairseq_g
```
- To preprocess data:\
With AMR: Just pass the argument ```--amr``` like the following\
```./fairseq_cli/preprocess.py --source-lang en --amr amr --target-lang vi --trainpref "train_dir" --validpref "valid_dir" --testpref "test_dir" --destdir "preprocessed_dir"```\
Without AMR:\
```./fairseq_g/fairseq_g/fairseq_cli/preprocess.py --source-lang en --target-lang vi --trainpref "train_dir" --validpref "valid_dir" --testpref "test_dir" --destdir "preprocessed_dir"```

- To train model:\
With AMR: again, just pass ```--with_amr```:\
```./fairseq_g/fairseq_cli/train.py "preprocessed_dir" --arch lstm --with_amr "True" --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --lr-scheduler fixed --force-anneal 200 --save-dir "checkpoints/lstm" --max-epoch 50```\
Without AMR: pass ```--with_amr "False"``` or not:\
```./fairseq_g/fairseq_cli/train.py "preprocessed_dir" --arch lstm --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --lr-scheduler fixed --force-anneal 200 --save-dir "checkpoints/lstm" --max-epoch 50```\
```./fairseq_g/fairseq_cli/train.py "preprocessed_dir" --arch lstm --with_amr "False --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --lr-scheduler fixed --force-anneal 200 --save-dir "checkpoints/lstm" --max-epoch 50```

## NB:
Use ```CUDA_VISIBLE_DEVICES=0``` when training to train on a single GPU with an effective batch size that is equivalent to training on 8 GPUs:\
```CUDA_VISIBLE_DEVICES=0 train.py ...```

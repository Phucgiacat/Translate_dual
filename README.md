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

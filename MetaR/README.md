## Environment
* RTX A6000
* Memory 128G

#### Log and State

Folder ``./log`` and ``./state`` will be make after starting an expariment. The log and a whole state  will be saved at ```{--log_dir}/{--prefix}``` and ```{--state_dir}/{--prefix}``` each ``{--eval_epoch}`` and ``{--checkpoint_epoch}``.

```
.
|-- log
|   \-- prefix
|       |-- events.out.tfevents.{num}.{username}  # tensorboard log
|       \-- res.log  # evaluation log during training and test log from logging module 
\-- state
    \-- prefix
        |-- checkpoint  # saved state every checkpoint_epoch
        \-- state_dict  # final state
```

## Train
NELL
```bash
python main_ATypeRank_add_neg_multiple.py --dataset NELL-One --data_path ./NELL --few 5 --r_cos 0.97 --aug_hum 1 # aug_hum: 1 3 5 7
```


FB15K-237
```bash
python main_ATypeRank_add_neg_multiple.py --dataset FB15K-One --data_path ./FB15K --few 5 --r_cos 0.97 --aug_hum 1 # aug_hum: 1 3 5 7
```

Wiki
```bash
python main_ATypeRank_add_neg_multiple.py --dataset Wiki-One --data_path ./Wiki --few 5 --r_cos 0.8 --aug_hum 1 # aug_hum: 1 3 5 7
```

## Eval
Download the checkpoint and extract to the `state/` folder.

NELL
```bash
python main_ATypeRank_add_neg_multiple.py --dataset NELL-One --data_path ./NELL --few 5 --r_cos 0.97 --aug_hum 1 --step test # aug_hum: 1 3 5 7
```

FB15K-237
```bash
python main_ATypeRank_add_neg_multiple.py --dataset FB15K-One --data_path ./FB15K --few 5 --r_cos 0.97 --aug_hum 1 --step test # aug_hum: 1 3 5 7
```

Wiki
```bash
python main_ATypeRank_add_neg_multiple.py --dataset Wiki-One --data_path ./Wiki --few 5 --r_cos 0.8 --aug_hum 1 --step test # aug_hum: 1 3 5 7
```


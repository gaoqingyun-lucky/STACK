## Requirement
```
normflows==1.4
dgl==0.9.0
tensorboardx==2.5.1
```

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
python main_ATypeRank_add_neg.py --dataset NELL-One --data_path ./NELL --few 5
```

FB15K-237
```bash
python main_ATypeRank_add_neg.py --dataset FB15K-One --data_path ./FB15K --few 5
```

## Eval
NELL
```bash
python main_ATypeRank_add_neg.py --dataset NELL-One --data_path ./NELL --few 5 --step test
```

FB15K-237
```bash
python main_ATypeRank_add_neg.py --dataset FB15K-One --data_path ./FB15K --few 5 --step test
```


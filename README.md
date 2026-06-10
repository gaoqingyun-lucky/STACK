# STACK: Semi-Supervised Relation-Type-Aware Augmentation with Consistency Constraint for Few-Shot Knowledge Graph Completion

This repository contains the official implementation of **STACK**.

## Repository Structure

```
STACK/
├── MetaR/                          # Meta Relational Learning approach
│   ├── main_ATypeRank_add_neg_multiple.py    # Main training/evaluation script
│   ├── ATypeRank_models_add_neg_multiple.py  # Model definitions
│   ├── trainer_ATypeRank_add_neg_multiple.py # Training loop
│   ├── data_loader.py                        # Data loading utilities
│   ├── embedding_ATypeRank_fast1_add_neg_multiple.py  # Embedding module
│   ├── params.py                              # Hyperparameters
│   ├── prepare.py                             # Data preprocessing
│   ├── requirements.txt                       # Dependencies
│   └── README.md                              # MetaR-specific instructions
│
└── NPFKGC/                         # Normalizing Flow approach
    ├── main_ATypeRank_add_neg.py              # Main training/evaluation script
    ├── ATypeRank_add_neg_model.py             # Model definitions
    ├── ATypeRank_add_neg_relational_path_gnn.py  # Relational path GNN
    ├── trainer_ATypeRank_add_neg.py           # Training loop
    ├── data_loader.py                         # Data loading utilities
    ├── embedding.py                           # Embedding module
    ├── flow.py                                # Normalizing flow module
    ├── params.py                              # Hyperparameters
    └── README.md                              # NPFKGC-specific instructions
```

## Requirements

### MetaR
- Python 3.x
- PyTorch
- TensorBoardX
- See `MetaR/requirements.txt` for details

### NPFKGC
- Python 3.x
- PyTorch
- DGL (Deep Graph Library) >= 0.9.0
- normflows >= 1.4
- TensorBoardX >= 2.5.1

## Environment
- GPU: RTX A6000 (recommended)
- Memory: 128GB (recommended)

## Datasets
The code supports the following few-shot knowledge graph datasets:
- **NELL-One** (NELL)
- **FB15K-237** (FB15K-One)
- **Wiki-One** (Wiki)
### Original Dataset
* [NELL/Wiki](https://github.com/xwhan/One-shot-Relational-Learning)
* [FB15K-237](https://github.com/SongW-SW/REFORM)
### Processed Dataset
* [Dataset](https://drive.google.com/drive/u/0/folders/1vN1AMapGZaUnQ4c7gPiBmO_nB6vvhj1c)
* [Checkpoint](https://drive.google.com/drive/u/0/folders/1gpHkQDgr5KzAXptl_fa1pATvk__prYUc)

Download the datasets and extract to the project root folder.  

## Training

### MetaR
```bash
# NELL
python MetaR/main_ATypeRank_add_neg_multiple.py --dataset NELL-One --data_path ./NELL --few 5 --r_cos 0.97 --aug_hum 1

# FB15K-237
python MetaR/main_ATypeRank_add_neg_multiple.py --dataset FB15K-One --data_path ./FB15K --few 5 --r_cos 0.97 --aug_hum 1

# Wiki
python MetaR/main_ATypeRank_add_neg_multiple.py --dataset Wiki-One --data_path ./Wiki --few 5 --r_cos 0.8 --aug_hum 1
```

### NPFKGC
```bash
# NELL
python NPFKGC/main_ATypeRank_add_neg.py --dataset NELL-One --data_path ./NELL --few 5

# FB15K-237
python NPFKGC/main_ATypeRank_add_neg.py --dataset FB15K-One --data_path ./FB15K --few 5
```

## Evaluation

### MetaR
```bash
# NELL
python MetaR/main_ATypeRank_add_neg_multiple.py --dataset NELL-One --data_path ./NELL --few 5 --r_cos 0.97 --aug_hum 1 --step test

# FB15K-237
python MetaR/main_ATypeRank_add_neg_multiple.py --dataset FB15K-One --data_path ./FB15K --few 5 --r_cos 0.97 --aug_hum 1 --step test

# Wiki
python MetaR/main_ATypeRank_add_neg_multiple.py --dataset Wiki-One --data_path ./Wiki --few 5 --r_cos 0.8 --aug_hum 1 --step test
```

### NPFKGC
```bash
# NELL
python NPFKGC/main_ATypeRank_add_neg.py --dataset NELL-One --data_path ./NELL --few 5 --step test

# FB15K-237
python NPFKGC/main_ATypeRank_add_neg.py --dataset FB15K-One --data_path ./FB15K --few 5 --step test
```

## Logs and Checkpoints

Training logs and model checkpoints are saved automatically:
- `./log/{prefix}/` — TensorBoard logs and evaluation results
- `./state/{prefix}/` — Model checkpoints and final state dicts


## License

This project is licensed under the MIT License.

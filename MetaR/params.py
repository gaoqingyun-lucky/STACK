import torch
import argparse


def get_params():
    args = argparse.ArgumentParser()
    args.add_argument("-data", "--dataset", default="NELL-One", type=str)  # ["NELL-One", "Wiki-One"]
    args.add_argument("-path", "--data_path", default="./NELL", type=str)  # ["./NELL", "./Wiki"]
    args.add_argument("-form", "--data_form", default="Pre-Train", type=str)  # ["Pre-Train", "In-Train", "Discard"]
    args.add_argument("-seed", "--seed", default=None, type=int)
    args.add_argument("-few", "--few", default=1, type=int)
    args.add_argument("-nq", "--num_query", default=3, type=int)
    args.add_argument("-metric", "--metric", default="MRR", choices=["MRR", "Hits@10", "Hits@5", "Hits@1"])

    args.add_argument("-dim", "--embed_dim", default=100, type=int)
    args.add_argument("-bs", "--batch_size", default=1024, type=int)#1024 1024*4
    args.add_argument("-lr", "--learning_rate", default=0.001, type=float)
    args.add_argument("-es_p", "--early_stopping_patience", default=30, type=int)

    args.add_argument("-epo", "--epoch", default=10000, type=int)
    args.add_argument("-prt_epo", "--print_epoch", default=100, type=int)
    args.add_argument("-eval_epo", "--eval_epoch", default=1000, type=int)
    args.add_argument("-ckpt_epo", "--checkpoint_epoch", default=1000, type=int)
    args.add_argument("-training_load_epo", "--training_load_epoch", default=0, type=int)

    args.add_argument("-b", "--beta", default=5, type=float)
    args.add_argument("-m", "--margin", default=1, type=float)
    args.add_argument("-p", "--dropout_p", default=0.5, type=float)
    args.add_argument("-abla", "--ablation", default=False, type=bool)

    args.add_argument("-gpu", "--device", default=0, type=int)

    args.add_argument("-prefix", "--prefix", default="exp1", type=str)
    args.add_argument("-step", "--step", default="train", type=str, choices=['train', 'test', 'dev'])
    args.add_argument("-log_dir", "--log_dir", default="log", type=str)
    args.add_argument("-state_dir", "--state_dir", default="state", type=str)
    args.add_argument("-eval_ckpt", "--eval_ckpt", default=None, type=str)
    args.add_argument("-eval_by_rel", "--eval_by_rel", default=False, type=bool)
    args.add_argument("-rel_type", "--rel_type_name", default="test_tasks", type=str)  # ["1-n", "n-1", "n-n"]
    args.add_argument("-r_cos", "--r_cos", default=0.97, type=float)  # 0.97,0.95,0.92,0.9,0.8
    args.add_argument("-no_training_epoch", "--no_training_epoch", default=500, type=int)  # 0 100 500 1000
    args.add_argument("-aug_hum", "--aug_hum", default=1, type=int)  # 1 3 5 7
    args.add_argument("-r_aug_w", "--r_aug_w", default=0.1, type=float)  # 0.001	0.01	0.1	1	10
    args.add_argument("-beta_alpha", "--Balpha", default=0.1, type=float)  # 0.1	0.3	0.5	1	2


    args = args.parse_args()
    params = {}
    params['device'] = torch.device('cuda:'+str(args.device))
    
    if args.dataset == 'NELL-One':
        params['embed_dim'] = 100
    elif args.dataset == 'Wiki-One':
        params['embed_dim'] = 50
    for k, v in vars(args).items():
        params[k] = v



    

    return params


data_dir = {
    'train_tasks_in_train': '/train_tasks_in_train.json',
    'train_tasks': '/train_tasks.json',
    'test_tasks': '/test_tasks.json',
    'dev_tasks': '/dev_tasks.json',

    'rel2candidates_in_train': '/rel2candidates_in_train.json',
    'rel2candidates': '/rel2candidates.json',

    'e1rel_e2_in_train': '/e1rel_e2_in_train.json',
    'e1rel_e2': '/e1rel_e2.json',

    'ent2ids': '/ent2ids',
    'ent2vec': '/entity2vec.TransE',
     'rel2ids': '/rel2ids',
    #'rel2ids': '/relation2ids',
    # 'ent2vec': '/entity2vec.ComplEx',
}

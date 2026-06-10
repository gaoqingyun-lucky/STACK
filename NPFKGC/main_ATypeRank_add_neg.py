from data_loader import *
from params import *
from trainer_ATypeRank_add_neg import *
import json
import os


if __name__ == '__main__':

    params = get_params()
    # params['dataset'] = 'FB15K-One' # FB15K-One FB15K-One
    # params['data_path'] = './dataset/FB15K'# Nell-HiRe FB15K
    # params['few'] = 5
    params['aug_hum'] = 1
    params['r_cos'] = 0.97 #0.8 0.9 0.92 0.95 0.97
    params['r_aug_w'] = 0.10
    params['no_training_epoch'] = 500 #1000 500 100
    params['tau'] = 0.07
    params['finetune_t'] = True
    params['prefix'] = f"ATypeRank_addneg_{params['dataset']}_few{params['few']}_cos{int(params['r_cos']*100)}({params['no_training_epoch']})"
    # params['training_load_epoch'] = 3000
    # params['eval_ckpt'] = '3000'
    if params['dataset'] == 'NELL-One':
        params['embed_dim'] = 100
    elif params['dataset'] == 'FB15K':
        params['embed_dim'] = 100
    elif params['dataset'] == 'YAGO3-10':
        params['embed_dim'] = 100
    elif params['dataset'] == 'Wiki-One':
        params['embed_dim'] = 50
    

    print("---------Parameters---------")
    for k, v in params.items():
        print(k + ': ' + str(v))
    print("----------------------------")

    # control random seed
    if params['seed'] is not None:
        SEED = params['seed']
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        np.random.seed(SEED)
        random.seed(SEED)
    # select the dataset
    for k, v in data_dir.items():
        data_dir[k] = params['data_path'] + v


    tail = ''


    dataset = dict()
    print("loading train_tasks{} ... ...".format(tail))
    dataset['train_tasks'] = json.load(open(data_dir['train_tasks' + tail]))
    print(len(dataset['train_tasks']))
    print("loading test_tasks ... ...")
    dataset['test_tasks'] = json.load(open(data_dir['test_tasks']))
    print("loading dev_tasks ... ...")
    dataset['dev_tasks'] = json.load(open(data_dir['dev_tasks']))


    print("loading rel2candidates{} ... ...".format(tail))
    dataset['rel2candidates'] = json.load(open(data_dir['rel2candidates' + tail]))
    print("loading e1rel_e2{} ... ...".format(tail))
    dataset['e1rel_e2'] = json.load(open(data_dir['e1rel_e2' + tail]))

    path_graph_dict = {}
    with open(params['data_path'] + '/path_graph') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            subject, predicate, object_ = line.rstrip().split()
            if predicate not in path_graph_dict:
                path_graph_dict[predicate] = []
            path_graph_dict[predicate].append([subject, predicate, object_])


    print("loading ent2id ... ...")
    if 'YAGO3-10' in params['data_path']:
        data_dir['ent2ids'] = params['data_path'] + '/entities.dict'
        data_dir['rel2ids'] = params['data_path'] + '/relations.dict'
        with open(data_dir['ent2ids'], 'r', encoding='utf-8') as f:
            lines = f.readlines()
            ent_dict = {}
            for line in lines:
                id, name = line.strip().split()
                ent_dict[name] = int(id)
        dataset['ent2id'] = ent_dict
        with open(data_dir['rel2ids'], 'r', encoding='utf-8') as f:
            lines = f.readlines()
            relations_dict = {}
            for line in lines:
                id, name = line.strip().split()
                relations_dict[name] = int(id)
        dataset['rel2id'] = relations_dict
    else:
        dataset['ent2id'] = json.load(open(data_dir['ent2ids']))
        dataset['rel2id'] = json.load(open(data_dir['rel2ids']))
        dataset['id2ent'] = {v: k for k, v in dataset['ent2id'].items()}
        dataset['id2rel'] = {v: k for k, v in dataset['rel2id'].items()}


    if params['data_form'] == 'Pre-Train':
        print('loading embedding ... ...')
        if 'YAGO3-10' in params['data_path']:
            data_dir['ent2vec']  = params['data_path']+'/entity_embedding.npy'
            data_dir['rel2vec'] = params['data_path'] + '/relation_embedding.npy'
            dataset['ent2emb'] = np.load(data_dir['ent2vec'])
            dataset['rel2emb'] = np.load(data_dir['rel2vec'])

        else:
            dataset['ent2emb'] = np.loadtxt(params['data_path'] + '/entity2vec.{}'.format(params['embed_model']))
            dataset['rel2emb'] = np.loadtxt(params['data_path'] + '/relation2vec.{}'.format(params['embed_model']))
        if params['dataset'] != 'Wiki-One':
            torch_tensor1 = torch.from_numpy(dataset['ent2emb'])
            dists = torch.cdist(torch_tensor1, torch_tensor1)  
            dists.fill_diagonal_(float('inf'))
            #    smallest k distances => largest=False
            nearest_idx = torch.topk(dists, 4, largest=False).indices 
            nearest_list = nearest_idx.tolist()  
            dataset['nearest_dict'] = {
                idx: neighbors
                for idx, neighbors in enumerate(nearest_list)
            } 
        else:
            json_file_path = f'{params["data_path"]}/wiki_cluster_neighbors.json'
            with open(json_file_path, "r", encoding="utf-8") as f:
                wiki_nearest_dict = json.load(f)
            dataset['nearest_dict'] = {int(k): v for k, v in wiki_nearest_dict.items()}
    print("----------------------------")

    # data_loader
    train_data_loader = DataLoader(dataset, params, step='train')
    dev_data_loader = DataLoader(dataset, params, step='dev')
    test_data_loader = DataLoader(dataset, params, step='test')
    data_loaders = [train_data_loader, dev_data_loader, test_data_loader]

    # trainer
    trainer = Trainer(data_loaders, dataset, params)

    if params['step'] == 'train':
        trainer.train()
        print("test")
        print(params['prefix'])
        trainer.reload()
        trainer.eval(istest=True)
    elif params['step'] == 'test':
        print(params['prefix'])
        if params['eval_by_rel']:
            trainer.eval_by_relation(istest=True)
        else:
            trainer.eval(istest=True)
    elif params['step'] == 'dev':
        print(params['prefix'])
        if params['eval_by_rel']:
            trainer.eval_by_relation(istest=False)
        else:
            trainer.eval(istest=False)

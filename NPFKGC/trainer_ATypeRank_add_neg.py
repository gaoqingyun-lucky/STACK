import logging
import os
import shutil
import sys
from collections import defaultdict

import dgl
import torch
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from tqdm import tqdm
import json
import numpy as np

from ATypeRank_add_neg_model import *


class Trainer:
    def __init__(self, data_loaders, dataset, parameter):
        self.parameter = parameter
        self.dataset = dataset
        # data loader
        self.train_data_loader = data_loaders[0]
        self.dev_data_loader = data_loaders[1]
        self.test_data_loader = data_loaders[2]
        # parameters
        self.few = parameter['few']
        # add
        self.r_aug_w = parameter['r_aug_w']
        self.aug_num = parameter['aug_hum'] + 1
        self.num_query = parameter['num_query']
        self.batch_size = parameter['batch_size']
        self.eval_batch_size = parameter['eval_batch_size']
        self.learning_rate = parameter['learning_rate']
        self.early_stopping_patience = parameter['early_stopping_patience']
        # epoch
        self.epoch = parameter['epoch']
        self.print_epoch = parameter['print_epoch']
        self.eval_epoch = parameter['eval_epoch']
        self.checkpoint_epoch = parameter['checkpoint_epoch']
        # device
        self.device = parameter['device']

        self.training_load_epoch = parameter['training_load_epoch']

        self.data_path = parameter['data_path']
        self.embed_model = parameter['embed_model']
        self.max_neighbor = parameter['max_neighbor']

        self.load_embed()
        self.num_symbols = len(self.symbol2id.keys()) - 1  # one for 'PAD'
        self.pad_id = self.num_symbols
        self.ent2id = dataset['ent2id']
        self.rel2id = dataset['rel2id']
        self.num_ents = len(self.ent2id.keys())
        degrees = self.build_connection(max_=self.max_neighbor)
        kg = self.build_kg(dataset['ent2emb'], dataset['rel2emb'], max_=self.max_neighbor)

        self.metaR = NPFKGC(kg, dataset, parameter, self.num_symbols, embed=self.symbol2vec)
        self.metaR.to(self.device)
        # optimizer
        self.optimizer = torch.optim.Adam(self.metaR.parameters(), self.learning_rate)
        # tensorboard log writer
        if parameter['step'] == 'train':
            self.writer = SummaryWriter(os.path.join(parameter['log_dir'], parameter['prefix']))
        # dir
        self.state_dir = os.path.join(self.parameter['state_dir'], self.parameter['prefix'])
        if not os.path.isdir(self.state_dir):
            os.makedirs(self.state_dir)
        self.ckpt_dir = os.path.join(self.parameter['state_dir'], self.parameter['prefix'], 'checkpoint')
        if not os.path.isdir(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        self.state_dict_file = ''

        # logging
        logging_dir = os.path.join(
            self.parameter['log_dir'], self.parameter['prefix'])
        if not os.path.exists(logging_dir):
            os.makedirs(logging_dir)
        logging.basicConfig(filename=os.path.join(logging_dir, "res.log"),
                            level=logging.INFO, format="%(asctime)s - %(message)s", force=True)
        logging.info('*' * 100)
        logging.info('*** hyper-parameters ***')
        for k, v in parameter.items():
            logging.info(k + ': ' + str(v))
        logging.info('*' * 100)

        if parameter['step'] in ('test', 'dev') or self.training_load_epoch != 0:
            self.reload()

    def load_symbol2id(self):

        symbol_id = {}
        rel2id = json.load(open(self.data_path + '/relation2ids'))
        ent2id = json.load(open(self.data_path + '/ent2ids'))
        i = 0
        for key in rel2id.keys():
            if key not in ['', 'OOV']:
                symbol_id[key] = i
                i += 1

        for key in ent2id.keys():
            if key not in ['', 'OOV']:
                symbol_id[key] = i
                i += 1

        symbol_id['PAD'] = i
        self.symbol2id = symbol_id
        self.symbol2vec = None

    def load_embed(self):

        symbol_id = {}
        symbol_idinv = {}
        if 'YAGO3-10' in self.data_path:
            ent_dir = self.data_path + '/entities.dict'
            rel_dir = self.data_path + '/relations.dict'
            with open(ent_dir, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                ent_dict = {}
                for line in lines:
                    id, name = line.strip().split()
                    ent_dict[name] = int(id)
            ent2id = ent_dict
            with open(rel_dir, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                relations_dict = {}
                for line in lines:
                    id, name = line.strip().split()
                    relations_dict[name] = int(id)
            rel2id = relations_dict
        else:
            rel2id = json.load(open(self.data_path + '/relation2ids'))
            ent2id = json.load(open(self.data_path + '/ent2ids'))

        logging.info('LOADING PRE-TRAINED EMBEDDING')
        if self.embed_model in ['DistMult', 'TransE', 'ComplEx', 'RESCAL']:
            if 'YAGO3-10' in self.data_path:
                ent2vec_dir = self.data_path + '/entity_embedding.npy'
                rel2vec_dir = self.data_path + '/relation_embedding.npy'
                ent_embed = np.load(ent2vec_dir)
                rel_embed = np.load(rel2vec_dir)

            else:
                ent_embed = np.loadtxt(self.data_path + '/entity2vec.' + self.embed_model)
                rel_embed = np.loadtxt(self.data_path + '/relation2vec.' + self.embed_model)

            if self.embed_model == 'ComplEx':
                # normalize the complex embeddings
                ent_mean = np.mean(ent_embed, axis=1, keepdims=True)
                ent_std = np.std(ent_embed, axis=1, keepdims=True)
                rel_mean = np.mean(rel_embed, axis=1, keepdims=True)
                rel_std = np.std(rel_embed, axis=1, keepdims=True)
                eps = 1e-3
                ent_embed = (ent_embed - ent_mean) / (ent_std + eps)
                rel_embed = (rel_embed - rel_mean) / (rel_std + eps)

            assert ent_embed.shape[0] == len(ent2id.keys())
            assert rel_embed.shape[0] == len(rel2id.keys())

            i = 0
            embeddings = []
            for key in rel2id.keys():
                if key not in ['', 'OOV']:
                    symbol_id[key] = i
                    symbol_idinv[i] = key
                    i += 1
                    embeddings.append(list(rel_embed[rel2id[key], :]))

            for key in ent2id.keys():
                if key not in ['', 'OOV']:
                    symbol_id[key] = i
                    symbol_idinv[i] = key
                    i += 1
                    embeddings.append(list(ent_embed[ent2id[key], :]))

            symbol_id['PAD'] = i
            embeddings.append(list(np.zeros((rel_embed.shape[1],))))
            embeddings = np.array(embeddings)
            assert embeddings.shape[0] == len(symbol_id.keys())

            self.symbol2id = symbol_id
            self.symbol2vec = embeddings
     
    def build_kg(self, ent_emb, rel_emb, max_=100):
        print("Build KG...")
        src = []
        dst = []
        e_feat = []
        e_id = []
        if 'YAGO3-10' not in self.data_path:
            with open(self.data_path + '/path_graph') as f:
                lines = f.readlines()
                for line in tqdm(lines):
                    e1, rel, e2 = line.rstrip().split()
                    src.append(self.ent2id[e1])
                    dst.append(self.ent2id[e2])
                    e_feat.append(rel_emb[self.rel2id[rel]])
                    e_id.append(self.rel2id[rel])
                    # Reverse
                    src.append(self.ent2id[e2])
                    dst.append(self.ent2id[e1])
                    e_feat.append(rel_emb[self.rel2id[rel + '_inv']])
                    e_id.append(self.rel2id[rel + '_inv'])
        else:
            test_task = self.dataset['test_tasks']
            train_task = self.dataset['train_tasks']
            dev_task = self.dataset['dev_tasks']
            for _, ite in train_task.items():
                for e1, rel, e2 in ite:
                    src.append(self.ent2id[e1])
                    dst.append(self.ent2id[e2])
                    e_feat.append(rel_emb[self.rel2id[rel]])
                    e_id.append(self.rel2id[rel])
            for _, ite in dev_task.items():
                for e1, rel, e2 in ite:
                    src.append(self.ent2id[e1])
                    dst.append(self.ent2id[e2])
                    e_feat.append(rel_emb[self.rel2id[rel]])
                    e_id.append(self.rel2id[rel])
            for _, ite in test_task.items():
                for e1, rel, e2 in ite:
                    src.append(self.ent2id[e1])
                    dst.append(self.ent2id[e2])
                    e_feat.append(rel_emb[self.rel2id[rel]])
                    e_id.append(self.rel2id[rel])




        src = torch.LongTensor(src)
        dst = torch.LongTensor(src)
        kg = dgl.graph((src, dst))
        kg.ndata['feat'] = torch.FloatTensor(ent_emb)
        kg.edata['feat'] = torch.FloatTensor(np.array(e_feat))
        kg.edata['eid'] = torch.LongTensor(np.array(e_id))
        return kg

    def build_connection(self, max_=100):
        self.connections = (np.ones((self.num_ents, max_, 3)) * self.pad_id).astype(int)
        self.e1_rele2 = defaultdict(list)
        self.e1_degrees = defaultdict(int)
        if 'YAGO3-10' not in self.data_path:
            with open(self.data_path + '/path_graph') as f:
                lines = f.readlines()
                for line in tqdm(lines):
                    e1, rel, e2 = line.rstrip().split()
                    self.e1_rele2[e1].append((self.symbol2id[e1], self.symbol2id[rel], self.symbol2id[e2]))
                    self.e1_rele2[e2].append((self.symbol2id[e2], self.symbol2id[rel + '_inv'], self.symbol2id[e1]))
        else:
            test_task = self.dataset['test_tasks']
            train_task = self.dataset['train_tasks']
            dev_task = self.dataset['dev_tasks']
            for _,ite in train_task.items():
                for e1, rel, e2 in ite:
                    self.e1_rele2[e1].append((self.symbol2id[e1], self.symbol2id[rel], self.symbol2id[e2]))
            for _, ite in dev_task.items():
                for e1, rel, e2 in ite:
                    self.e1_rele2[e1].append((self.symbol2id[e1], self.symbol2id[rel], self.symbol2id[e2]))
            for  _, ite in test_task.items():
                for e1, rel, e2 in ite:
                    self.e1_rele2[e1].append((self.symbol2id[e1], self.symbol2id[rel], self.symbol2id[e2]))




        degrees = {}
        for ent, id_ in self.ent2id.items():
            neighbors = self.e1_rele2[ent]
            if len(neighbors) > max_:
                neighbors = neighbors[:max_]
            degrees[ent] = len(neighbors)
            self.e1_degrees[id_] = len(neighbors)  # add one for self conn
            for idx, _ in enumerate(neighbors):
                self.connections[id_, idx, 0] = _[0]
                self.connections[id_, idx, 1] = _[1]
                self.connections[id_, idx, 2] = _[2]

        return degrees

    def get_meta(self, left, right):
        left_connections = Variable(
            torch.LongTensor(np.stack([self.connections[_, :, :] for _ in left], axis=0))).cuda()
        left_degrees = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in left])).cuda()
        right_connections = Variable(
            torch.LongTensor(np.stack([self.connections[_, :, :] for _ in right], axis=0))).cuda()
        right_degrees = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in right])).cuda()
        return (left_connections, left_degrees, right_connections, right_degrees)

    def reload(self):
        if self.parameter['eval_ckpt'] is not None:
            state_dict_file = os.path.join(self.ckpt_dir, 'state_dict_' + self.parameter['eval_ckpt'] + '.ckpt')
        else:
            state_dict_file = os.path.join(self.state_dir, 'state_dict')
        self.state_dict_file = state_dict_file
        logging.info('Reload state_dict from {}'.format(state_dict_file))
        print('reload state_dict from {}'.format(state_dict_file))
        state = torch.load(state_dict_file, map_location=self.device)
        if os.path.isfile(state_dict_file):
            self.metaR.load_state_dict(state, strict=False)
        else:
            raise RuntimeError('No state dict in {}!'.format(state_dict_file))

    def save_checkpoint(self, epoch):
        torch.save(self.metaR.state_dict(), os.path.join(self.ckpt_dir, 'state_dict_' + str(epoch) + '.ckpt'))

    def del_checkpoint(self, epoch):
        path = os.path.join(self.ckpt_dir, 'state_dict_' + str(epoch) + '.ckpt')
        if os.path.exists(path):
            os.remove(path)
        else:
            raise RuntimeError('No such checkpoint to delete: {}'.format(path))

    def save_best_state_dict(self, best_epoch):
        shutil.copy(os.path.join(self.ckpt_dir, 'state_dict_' + str(best_epoch) + '.ckpt'),
                    os.path.join(self.state_dir, 'state_dict'))

    def write_training_log(self, data, epoch):
        self.writer.add_scalar('Training_Loss', data['Loss'], epoch)
        self.writer.add_scalar('KL_Loss', data['KL'], epoch)

    def write_validating_log(self, data, epoch):
        self.writer.add_scalar('Validating_MRR', data['MRR'], epoch)
        self.writer.add_scalar('Validating_Hits_10', data['Hits@10'], epoch)
        self.writer.add_scalar('Validating_Hits_5', data['Hits@5'], epoch)
        self.writer.add_scalar('Validating_Hits_1', data['Hits@1'], epoch)

    def logging_training_data(self, data, epoch):
        logging.info("Epoch: {}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
            epoch, data['MRR'], data['Hits@10'], data['Hits@5'], data['Hits@1']))

    def logging_eval_data(self, data, state_path, istest=False):
        setname = 'dev set'
        if istest:
            setname = 'test set'
        logging.info("Eval {} on {}".format(state_path, setname))
        logging.info("MRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
            data['MRR'], data['Hits@10'], data['Hits@5'], data['Hits@1']))

    def rank_predict(self, data, x, ranks):
        # query_idx is the idx of positive score
        query_idx = x.shape[0] - 1
        # sort all scores with descending, because more plausible triple has higher score
        _, idx = torch.sort(x, descending=True)
        rank = list(idx.cpu().numpy()).index(query_idx) + 1
        ranks.append(rank)
        # update data
        if rank <= 10:
            data['Hits@10'] += 1
        if rank <= 5:
            data['Hits@5'] += 1
        if rank == 1:
            data['Hits@1'] += 1
        data['MRR'] += 1.0 / rank

    def do_one_step(self, task, iseval=False, curr_rel='', istest=False, batch_eval=False, epoch_num = int):
        loss, p_score, n_score = 0, 0, 0
        support = task[0]
        support_left = [self.ent2id[few[0]] for batch in support for few in batch]
        support_right = [self.ent2id[few[2]] for batch in support for few in batch]
        if iseval == False:
            meta_left = [[0] * self.batch_size for i in range(self.few)]
            meta_right = [[0] * self.batch_size for i in range(self.few)]
        if iseval == True:
            meta_left = [[0] for i in range(self.few)]
            meta_right = [[0] for i in range(self.few)]

        for i in range(len(meta_left)):
            for j in range(len(meta_left[0])):
                meta_left[i][j] = support_left[j * self.few + i]
        for i in range(len(meta_right)):
            for j in range(len(meta_right[0])):
                meta_right[i][j] = support_right[j * self.few + i]

        support_meta = []
        for i in range(len(meta_left)):

            support_meta.append(self.get_meta(meta_left[i], meta_right[i]))
        if not iseval:
            self.optimizer.zero_grad()
            p_score, n_score, kld, loss_r_aug, cl_loss = self.metaR(task, iseval, curr_rel, support_meta, istest, epoch_num)
            y = torch.ones_like(p_score).to(self.device)
            kl_loss = kld.mean(0)
            loss = self.metaR.loss_func(p_score, n_score, y) + kl_loss + self.r_aug_w * (loss_r_aug + cl_loss)
            loss.backward()
            self.optimizer.step()
            return loss, kl_loss, loss_r_aug, cl_loss, p_score, n_score
        elif curr_rel != '':
            with torch.no_grad():
                if batch_eval:
                    p_score, n_score = self.metaR.eval_forward(task, iseval, curr_rel, support_meta, istest)
                else:
                    p_score_raw, n_score_raw = self.metaR(task, iseval, curr_rel, support_meta, istest, epoch_num)
                    total = n_score_raw.numel()  
                    valid_total = (total // self.aug_num) * self.aug_num 
                    if total != valid_total:
                        n_score_raw = n_score_raw.flatten()[:valid_total].view(
                            n_score_raw.shape[:-1] + (valid_total // n_score_raw.shape[0],))
                    n_score_grouped = n_score_raw.view(-1, self.aug_num)  
                    n_score = n_score_grouped.mean(dim=1).view(1, -1)  
                    p_score = p_score_raw.mean().view(1, -1)
                y = torch.ones_like(p_score).to(self.device)
                loss = self.metaR.loss_func(p_score, n_score, y)
            return loss, p_score, n_score

    def train(self):
        # initialization
        best_epoch = 0
        best_value = 0
        bad_counts = 0

        # training by epoch
        for e in range(self.training_load_epoch, self.epoch):
            # sample one batch from data_loader
            self.metaR.train()
            train_task, curr_rel = self.train_data_loader.next_batch()
            loss, kl_loss, loss_r_aug, cl_loss, _, _ = self.do_one_step(train_task, iseval=False, curr_rel=curr_rel, istest=False, epoch_num = e)
            # print the loss on specific epoch
            if e % self.print_epoch == 0:
                loss_num = loss.item()
                self.write_training_log({'Loss': loss_num, 'KL': kl_loss.item(), 'R_Aug': loss_r_aug.item(), 'CL': cl_loss.item()}, e)
                log_message = "Epoch: {}\tLoss: {:.4f}\tkl_loss: {:.4f}\tloss_r_aug: {:.4f}\tcl_loss: {:.4f}".format(e, loss_num,kl_loss.item(),loss_r_aug.item(),cl_loss.item())
                logging.info(log_message)  # 将信息写入日志文件
                print("Epoch: {}\tLoss: {:.4f}\tkl_loss: {:.4f}\tloss_r_aug: {:.4f}\tcl_loss: {:.4f}".format(e, loss_num,kl_loss.item(),loss_r_aug.item(),cl_loss.item()))
            # save checkpoint on specific epoch
            if e % self.checkpoint_epoch == 0 and e != 0:
                print('Epoch  {} has finished, saving...'.format(e))
                self.save_checkpoint(e)
            # do evaluation on specific epoch
            if e % self.eval_epoch == 0 and e != 0:
                print('Epoch  {} has finished, validating...'.format(e))

                valid_data = self.eval(istest=False, epoch=e)
                self.write_validating_log(valid_data, e)

                metric = self.parameter['metric']
                # early stopping checking
                if valid_data[metric] > best_value:
                    best_value = valid_data[metric]
                    best_epoch = e
                    print('\tBest model | {0} of valid set is {1:.3f}'.format(metric, best_value))
                    bad_counts = 0
                    # save current best
                    self.save_checkpoint(best_epoch)
                    test_data = self.eval(istest=True)
                else:
                    print('\tBest {0} of valid set is {1:.3f} at {2} | bad count is {3}'.format(
                        metric, best_value, best_epoch, bad_counts))
                    bad_counts += 1

                if bad_counts >= self.early_stopping_patience:
                    print('\tEarly stopping at epoch %d' % e)
                    break

        print('Training has finished')
        print('\tBest epoch is {0} | {1} of valid set is {2:.3f}'.format(best_epoch, metric, best_value))
        self.save_best_state_dict(best_epoch)
        print('Finish')

    def eval(self, istest=False, epoch=None):
        self.metaR.eval()
        # clear sharing rel_q
        self.metaR.rel_q_sharing = dict()

        if istest:
            data_loader = self.test_data_loader
        else:
            data_loader = self.dev_data_loader
        data_loader.curr_tri_idx = 0

        # initial return data of validation
        data = {'MRR': 0, 'Hits@1': 0, 'Hits@5': 0, 'Hits@10': 0}
        ranks = []

        t = 0
        temp = dict()
        while True:
            # sample all the eval tasks
            eval_task, curr_rel = data_loader.next_one_on_eval()
            # at the end of sample tasks, a symbol 'EOT' will return
            if eval_task == 'EOT' or curr_rel == 'EOT':
                break
            t += 1

            if self.eval_batch_size > 0:
                def batch(iterable, n=1):
                    l = len(iterable)
                    for ndx in range(0, l, n):
                        yield [iterable[ndx:min(ndx + n, l)]]

                score_list = []
                support_triples, support_negative_triples, query_triple, negative_triples = eval_task
                self.metaR.eval_reset()
                for neg_batch in batch(negative_triples[0], self.eval_batch_size):
                    eval_task_batch = [support_triples, support_negative_triples, query_triple, neg_batch]
                    _, p_score, n_score = self.do_one_step(eval_task_batch, iseval=True, curr_rel=curr_rel,
                                                           istest=istest, batch_eval=True, epoch_num = epoch)
                    score_list.append(n_score.detach().cpu())
                score_list.append(p_score.detach().cpu())
                x = torch.cat(score_list, 1).squeeze()
            else:
                _, p_score, n_score = self.do_one_step(eval_task, iseval=True, curr_rel=curr_rel, istest=istest, epoch_num = epoch)

                x = torch.cat([n_score, p_score], 1).squeeze()

            self.rank_predict(data, x, ranks)

            # print current temp data dynamically
            for k in data.keys():
                temp[k] = data[k] / t
            sys.stdout.write("{}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
                t, temp['MRR'], temp['Hits@10'], temp['Hits@5'], temp['Hits@1']))
            sys.stdout.flush()
            # if t>50:
            #    break

        # print overall evaluation result and return it
        for k in data.keys():
            data[k] = round(data[k] / t, 3)

        if self.parameter['step'] == 'train':
            self.logging_training_data(data, epoch)
        else:
            self.logging_eval_data(data, self.state_dict_file, istest)

        print("{}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
            t, data['MRR'], data['Hits@10'], data['Hits@5'], data['Hits@1']))

        return data

    def eval_by_relation(self, istest=False, epoch=None):
        self.metaR.eval()
        self.metaR.rel_q_sharing = dict()

        if istest:
            data_loader = self.test_data_loader
        else:
            data_loader = self.dev_data_loader
        data_loader.curr_tri_idx = 0

        all_data = {'MRR': 0, 'Hits@1': 0, 'Hits@5': 0, 'Hits@10': 0}
        all_t = 0
        all_ranks = []

        for rel in data_loader.all_rels:
            print("rel: {}, num_cands: {}, num_tasks:{}".format(
                rel, len(data_loader.rel2candidates[rel]), len(data_loader.tasks[rel][self.few:])))
            data = {'MRR': 0, 'Hits@1': 0, 'Hits@5': 0, 'Hits@10': 0}
            temp = dict()
            t = 0
            ranks = []
            while True:
                eval_task, curr_rel = data_loader.next_one_on_eval_by_relation(rel)
                if eval_task == 'EOT' or curr_rel == 'EOT':
                    break
                t += 1

                _, p_score, n_score = self.do_one_step(eval_task, iseval=True, curr_rel=rel, istest=istest, epoch_num = epoch)
                x = torch.cat([n_score, p_score], 1).squeeze()

                self.rank_predict(data, x, ranks)

                for k in data.keys():
                    temp[k] = data[k] / t
                sys.stdout.write("{}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
                    t, temp['MRR'], temp['Hits@10'], temp['Hits@5'], temp['Hits@1']))
                sys.stdout.flush()

            print("{}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
                t, temp['MRR'], temp['Hits@10'], temp['Hits@5'], temp['Hits@1']))

            for k in data.keys():
                all_data[k] += data[k]
            all_t += t
            all_ranks.extend(ranks)

        print('Overall')
        for k in all_data.keys():
            all_data[k] = round(all_data[k] / all_t, 3)
        print("{}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
            all_t, all_data['MRR'], all_data['Hits@10'], all_data['Hits@5'], all_data['Hits@1']))

        return all_data

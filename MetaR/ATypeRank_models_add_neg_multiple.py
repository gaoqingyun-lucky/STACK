from embedding_ATypeRank_fast1_add_neg_multiple import *
from collections import OrderedDict
import torch


class RelationMetaLearner(nn.Module):
    def __init__(self, few, embed_size=100, num_hidden1=500, num_hidden2=200, out_size=100, dropout_p=0.5):
        super(RelationMetaLearner, self).__init__()
        self.embed_size = embed_size
        self.few = few
        self.out_size = out_size
        self.rel_fc1 = nn.Sequential(OrderedDict([
            ('fc',   nn.Linear(2*embed_size, num_hidden1)),
            ('bn',   nn.BatchNorm1d(few)),
            ('relu', nn.LeakyReLU()),
            ('drop', nn.Dropout(p=dropout_p)),
        ]))
        self.rel_fc2 = nn.Sequential(OrderedDict([
            ('fc',   nn.Linear(num_hidden1, num_hidden2)),
            ('bn',   nn.BatchNorm1d(few)),
            ('relu', nn.LeakyReLU()),
            ('drop', nn.Dropout(p=dropout_p)),
        ]))
        self.rel_fc3 = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(num_hidden2, out_size)),
            ('bn', nn.BatchNorm1d(few)),
        ]))
        nn.init.xavier_normal_(self.rel_fc1.fc.weight)
        nn.init.xavier_normal_(self.rel_fc2.fc.weight)
        nn.init.xavier_normal_(self.rel_fc3.fc.weight)

    def forward(self, inputs):
        size = inputs.shape
        x = inputs.contiguous().view(size[0], size[1], -1)
        x = self.rel_fc1(x)
        x = self.rel_fc2(x)
        x = self.rel_fc3(x)
        x = torch.mean(x, 1)

        return x.view(size[0], 1, 1, self.out_size)


class EmbeddingLearner(nn.Module):
    def __init__(self):
        super(EmbeddingLearner, self).__init__()

    def forward(self, h, t, r, pos_num):
        score = -torch.norm(h + r - t, 2, -1).squeeze(2)
        p_score = score[:, :pos_num]
        n_score = score[:, pos_num:]
        return p_score, n_score


class MetaR(nn.Module):
    def __init__(self, dataset, parameter):
        super(MetaR, self).__init__()
        self.device = parameter['device']
        self.beta = parameter['beta']
        self.dropout_p = parameter['dropout_p']
        self.embed_dim = parameter['embed_dim']
        self.margin = parameter['margin']
        self.abla = parameter['ablation']
        self.embedding = Embedding(dataset, parameter)
        self.aug_num = parameter['aug_hum']
        self.few = parameter['few'] * (self.aug_num + 1)
        self.r_cos = parameter['r_cos']
        self.no_training_epoch = parameter['no_training_epoch']
        self.tau = nn.Parameter(torch.tensor(parameter['tau'])) if parameter['finetune_t'] else torch.tensor(parameter['tau'])
        self.r_criterion_loss = KLWithCosineLoss(threshold=self.r_cos, init_beta=0.5)


        if parameter['dataset'] == 'Wiki-One':
            self.relation_learner = RelationMetaLearner(self.few, embed_size=50, num_hidden1=250,
                                                        num_hidden2=100, out_size=50, dropout_p=self.dropout_p)
        elif parameter['dataset'] == 'NELL-One':
            self.relation_learner = RelationMetaLearner(self.few, embed_size=100, num_hidden1=500,
                                                        num_hidden2=200, out_size=100, dropout_p=self.dropout_p)
        elif parameter['dataset'] == 'FB15K':
            self.relation_learner = RelationMetaLearner(self.few, embed_size=100, num_hidden1=500,
                                                        num_hidden2=200, out_size=100, dropout_p=self.dropout_p)

        elif parameter['dataset'] == 'YAGO3-10':
            self.relation_learner = RelationMetaLearner(self.few, embed_size=100, num_hidden1=500,
                                                        num_hidden2=200, out_size=100, dropout_p=self.dropout_p)
        self.embedding_learner = EmbeddingLearner()

        self.loss_func = nn.MarginRankingLoss(self.margin)
        self.rel_q_sharing = dict()

    def split_concat(self, positive, negative):
        pos_neg_e1 = torch.cat([positive[:, :, 0, :],
                                negative[:, :, 0, :]], 1).unsqueeze(2)
        pos_neg_e2 = torch.cat([positive[:, :, 1, :],
                                negative[:, :, 1, :]], 1).unsqueeze(2)
        return pos_neg_e1, pos_neg_e2

    def info_nce_loss(self, positive_scores: torch.Tensor,
                      negative_scores: torch.Tensor) -> torch.Tensor:
        batch_size, num_positives = positive_scores.shape
        num_negatives = negative_scores.shape[1]

        # 扩展正样本维度：(batch_size, few) -> (batch_size, few, 1)
        positive_scores = positive_scores.unsqueeze(2)

        # 计算正样本和负样本的phi值（使用可训练的tau）
        # tau需要进行维度调整以匹配广播机制
        tau_expanded = self.tau.view(1, 1, 1)  # 调整为(1,1,1)以适配广播
        positive_phi = positive_scores / tau_expanded
        negative_scores_exp = negative_scores.unsqueeze(1)
        negative_scores_exp = negative_scores_exp.expand(batch_size, num_positives, num_negatives)
        negative_phi = negative_scores_exp / tau_expanded  # (batch_size, 1, few) -> 广播为(batch_size, few, few)

        # 拼接正样本和负样本得分：(batch_size, few, 1 + few)
        logits = torch.cat([positive_phi, negative_phi], dim=2)

        # 计算softmax并取正样本位置（索引0）的对数概率
        log_probs = F.log_softmax(logits, dim=2)
        loss = -log_probs[:, :, 0].mean()  # 对所有正样本的损失取平均

        return loss

    def forward(self, task, iseval=False, curr_rel='', current_epoch=int):
        few = self.few
        # aug_num = self.aug_num

        # --- Step 1: 获取 support/query/negative embeddings ---
        if isinstance(current_epoch, int):
            if current_epoch < self.no_training_epoch:#100 500
                support, support_negative, query, negative = [self.embedding(t) for t in task]
                support = support.repeat_interleave(self.aug_num + 1, dim=1)
                support_negative = support_negative.repeat_interleave(self.aug_num + 1, dim=1)
                query = query.repeat_interleave(self.aug_num + 1, dim=1)
                negative = negative.repeat_interleave(self.aug_num + 1, dim=1)
                rel = self.relation_learner(support)
                loss_r_aug = torch.tensor(0., device=self.device)
            else:
                support_raw = self.embedding(task[0])
                support_raw = support_raw.repeat_interleave(self.aug_num + 1, dim=1)
                rel_raw = self.relation_learner(support_raw)
                support, r_type, support_neg_aug, neg_aug_types = self.embedding.support_Aug(task[0], rel_raw)
                rel = self.relation_learner(support)
                query, neg_aug = self.embedding.query_Aug(task[2], r_type, neg_aug_types)
                support_negative = self.embedding(task[1])
                support_negative = torch.cat([support_negative, support_neg_aug], dim=1)
                negative = self.embedding(task[-1])
                negative = torch.cat([negative, neg_aug], dim=1)

                # Augmentation loss
                logvar1 = torch.randn_like(rel_raw)
                logvar2 = torch.randn_like(rel)
                loss_r_aug = self.r_criterion_loss(rel_raw, logvar1, rel, logvar2)
        else:
            support_raw = self.embedding(task[0])
            support_raw = support_raw.repeat_interleave(self.aug_num + 1, dim=1)
            rel_raw = self.relation_learner(support_raw)
            support, r_type, support_neg_aug, neg_aug_types = self.embedding.support_Aug(task[0], rel_raw)
            rel = self.relation_learner(support)
            query, neg_aug = self.embedding.query_Aug(task[2], r_type, neg_aug_types)
            support_negative = self.embedding(task[1])
            support_negative = torch.cat([support_negative, support_neg_aug], dim=1)
            if task[-1]!= [[]]:
                negative = self.embedding(task[-1])
                negative = torch.cat([negative, neg_aug], dim=1)
            else:
                negative = neg_aug.repeat_interleave(self.aug_num + 1, dim=1)

            # Augmentation loss
            logvar1 = torch.randn_like(rel_raw)
            logvar2 = torch.randn_like(rel)
            loss_r_aug = self.r_criterion_loss(rel_raw, logvar1, rel, logvar2)

        # --- Step 2: Meta-learning relation update ---
        rel_s = rel.expand(-1, few + support_negative.shape[1], -1, -1)

        if iseval and curr_rel != '' and curr_rel in self.rel_q_sharing.keys():
            rel_q = self.rel_q_sharing[curr_rel]
        else:
            if isinstance(current_epoch, int):
                if not self.abla and current_epoch > -1:#100
                    # split e1/e2 and concat pos/neg
                    sup_neg_e1, sup_neg_e2 = self.split_concat(support, support_negative)
                    p_score, n_score = self.embedding_learner(sup_neg_e1, sup_neg_e2, rel_s, few)
                    y = torch.ones_like(p_score, device=self.device)
                    sup_cl_loss  = self.info_nce_loss(p_score, n_score)
                    # 计算 meta 梯度
                    grad_meta = torch.autograd.grad(self.loss_func(p_score, n_score, y) + sup_cl_loss,
                                                    rel, retain_graph=True, create_graph=False)[0]
                    rel_q = (rel - self.beta * grad_meta).detach()
                else:
                    rel_q = rel
            else:
                sup_neg_e1, sup_neg_e2 = self.split_concat(support, support_negative)
                p_score, n_score = self.embedding_learner(sup_neg_e1, sup_neg_e2, rel_s, few)
                y = torch.ones_like(p_score, device=self.device)
                sup_cl_loss = self.info_nce_loss(p_score, n_score)

                # 计算 meta 梯度
                grad_meta = torch.autograd.grad(self.loss_func(p_score, n_score, y) + sup_cl_loss,
                                                rel, retain_graph=True, create_graph=False)[0]
                rel_q = (rel - self.beta * grad_meta).detach()
            self.rel_q_sharing[curr_rel] = rel_q



        # --- Step 3: 扩展 rel_q 对 query ---
        rel_q = rel_q.expand(-1, query.shape[1] + negative.shape[1], -1, -1)

        # --- Step 4: 计算 query/negative 分数 ---
        que_neg_e1, que_neg_e2 = self.split_concat(query, negative)
        p_score, n_score = self.embedding_learner(que_neg_e1, que_neg_e2, rel_q, query.shape[1])
        cl_loss = self.info_nce_loss(p_score, n_score)

        return p_score, n_score, loss_r_aug, cl_loss


class KLWithCosineLoss(nn.Module):
    def __init__(self, threshold=0.9, init_beta=0.5):
        super().__init__()
        self.log_beta = nn.Parameter(torch.log(torch.tensor(init_beta)))  # 自适应权重
        self.threshold = threshold

    def kl_diag_gaussians(self, mu1, logvar1, mu2, logvar2, eps=1e-8):
        # var
        var1 = torch.exp(logvar1) + eps
        var2 = torch.exp(logvar2) + eps
        # KL
        term1 = logvar2 - logvar1
        term2 = (var1 + (mu1 - mu2)**2) / var2
        kl_per_dim = 0.5 * (term1 + term2 - 1.0)
        kl = torch.sum(kl_per_dim, dim=1)
        return kl

    def cosine_similarity_loss(self, mu1, mu2):
        cos_sim = F.cosine_similarity(mu1, mu2, dim=1)
        sim_loss = F.relu(self.threshold - cos_sim)
        return sim_loss

    def forward(self, mu1, logvar1, mu2, logvar2):
        # 确保 logvar 和 mu 形状及 device 一致
        logvar1 = logvar1.to(mu1.device).reshape_as(mu1)
        logvar2 = logvar2.to(mu1.device).reshape_as(mu1)

        kl_loss = self.kl_diag_gaussians(mu1, logvar1, mu2, logvar2)
        cos_loss = self.cosine_similarity_loss(mu1, mu2)

        beta = torch.exp(self.log_beta)
        total_loss = kl_loss + beta * cos_loss
        return total_loss.mean()



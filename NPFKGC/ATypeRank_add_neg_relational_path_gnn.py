import dgl
import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.functional import edge_softmax
from torch.distributions import Normal, Beta

class MLP(nn.Module):
    def __init__(self, embed_num):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_num, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, embed_num)
        )

    def forward(self, x):
        return self.net(x)


class RelationalPathGNN(nn.Module):
    def __init__(self, g, dataset, num_rel, parameter):
        super(RelationalPathGNN, self).__init__()
        self.ent2id_dict = dataset['ent2id']
        self.num_ent = len(self.ent2id_dict)
        self.device = parameter['device']
        self.hop = parameter['hop']
        self.es = parameter['embed_dim']
        self.g_batch = parameter['g_batch']
        self.g = g
        self.sampler = dgl.dataloading.MultiLayerFullNeighborSampler(parameter['hop'], prefetch_node_feats=['feat'],
                                                                     prefetch_edge_feats=['feat', 'eid'])
        self.gcn = RPGNN(self.es, self.es * 2, self.es, self.hop,
                         num_rel)
        self.num_rel = num_rel
        self.few = parameter['few'] * (parameter['aug_hum'] + 1)
        self.nearest_dict = dataset['nearest_dict']
        self.proj_rel_embed = MLP(self.es)
        # Beta 参数
        self.beta_weight = nn.Parameter(torch.tensor(0.1))
        self.dist = Normal(0, 1)

        # 缓存
        self.build_cache(num_neighbors=5)


    def build_cache(self, num_neighbors=1):
        """缓存实体 embedding 和 top-z 最近邻到 GPU"""
        device = torch.device(self.device)

        # nearest neighbors 缓存 - 向量化实现
        num_ent = self.num_ent
        nearest_tensor = torch.zeros(num_ent, num_neighbors, dtype=torch.long, device=device)

        # 转换nearest_dict为张量索引
        keys = torch.tensor(list(self.nearest_dict.keys()), dtype=torch.long, device=device)
        values = [torch.tensor(neighbors, dtype=torch.long, device=device)
                  for neighbors in self.nearest_dict.values()]

        # 填充最近邻张量
        for i, key in enumerate(keys):
            neighbors = values[i]
            n = len(neighbors)
            if n >= num_neighbors:
                nearest_tensor[key] = neighbors[:num_neighbors]
            else:
                # 填充不足部分
                pad_length = num_neighbors - n
                pad = neighbors[-1].repeat(pad_length)
                nearest_tensor[key] = torch.cat([neighbors, pad])

        self.nearest_cache = nearest_tensor  # (num_ent, num_neighbors)

        # proj_rel_embed 放到 device
        self.proj_rel_embed.to(device)
        self.to(device)

        print(f"nearest_cache: {self.nearest_cache.shape}")

    # 三元组评分
    def triple_score(self, head_tail_embed, r_embed):
        """计算三元组得分 L1(t_proj - h_proj - r)"""
        if head_tail_embed.dim() == 2:
            head_tail_embed = head_tail_embed.unsqueeze(0)

        B, two, d = head_tail_embed.shape
        head = head_tail_embed[:, 0, :]
        tail = head_tail_embed[:, 1, :]

        # 处理关系嵌入的向量化扩展
        r_embed = r_embed.squeeze()
        if r_embed.dim() == 1:
            r_embed = r_embed.unsqueeze(0).expand(B, d)
        elif r_embed.size(0) == 1:
            r_embed = r_embed.expand(B, d)

        w = self.proj_rel_embed(r_embed)
        w = F.normalize(w, p=2, dim=-1)

        # 向量化投影计算
        head_proj = head - (head * w).sum(dim=-1, keepdim=True) * w
        tail_proj = tail - (tail * w).sum(dim=-1, keepdim=True) * w

        score = torch.norm(tail_proj - head_proj - r_embed, p=1, dim=-1)
        return score


    def ent2id(self, triples):
        idx = [[[self.ent2id_dict[t[0]], self.ent2id_dict[t[2]]] for t in batch] for batch in triples]
        idx = torch.LongTensor(idx).to(self.device)
        return idx  # B * few * 2

    def support_Aug(self, triples, rel_emd):
        # 生成alpha_s（向量化）
        alpha_s = 0.5 + self.beta_weight * self.dist.sample((len(triples),)).to(self.device)
        alpha_s = alpha_s.unsqueeze(1).unsqueeze(2).unsqueeze(3)  # 适配广播维度

        # 提取所有head和tail ID（向量化）
        all_head_ids = torch.tensor([[self.ent2id_dict[x[0]] for x in batch] for batch in triples],
                                    device=self.device)  # (triples_len, few)
        all_tail_ids = torch.tensor([[self.ent2id_dict[x[2]] for x in batch] for batch in triples],
                                    device=self.device)  # (triples_len, few)
                # 构建pair ID和嵌入
        batch_size, few_size = all_head_ids.shape
        pair_ids = torch.stack([all_head_ids, all_tail_ids], dim=2)  # (batch, few, 2)
        # pair_emd = self.ent_cache[pair_ids]  # (batch, few, 2, es)
        pair_ids = pair_ids.view(-1)
        dataloader = dgl.dataloading.DataLoader(
            self.g, pair_ids, self.sampler,
            batch_size=self.g_batch,
            shuffle=False,
            drop_last=False,
            device=self.device,
            use_uva=True)
        pair_out_list = []
        for _, _, blocks in dataloader:
            input_features = blocks[0].srcdata['feat']
            out_features = self.gcn(blocks, input_features)
            pair_out_list.append(out_features)
        pair_emb = torch.cat(pair_out_list, dim=0)
        pair_emd = pair_emb.view(batch_size, few_size, 2, -1)

        # 获取最近邻（向量化）
        nearest_h = self.nearest_cache[all_head_ids, 0]  # (batch, few)
        nearest_t = self.nearest_cache[all_tail_ids, 0]  # (batch, few)

        # 生成所有增强类型的嵌入（一次性计算所有可能的增强）
        head_aug_ids = torch.stack([nearest_h, all_tail_ids], dim=2)  # (batch, few, 2)
        tail_aug_ids = torch.stack([all_head_ids, nearest_t], dim=2)  # (batch, few, 2)
        head_tail_aug_ids = torch.stack([nearest_h, nearest_t], dim=2)  # (batch, few, 2)
        # 获取增强嵌入
       
        head_aug_ids = head_aug_ids.view(-1)
        dataloader = dgl.dataloading.DataLoader(
            self.g, head_aug_ids, self.sampler,
            batch_size=self.g_batch,
            shuffle=False,
            drop_last=False,
            device=self.device,
            use_uva=True)
        head_aug_out_list = []
        for _, _, blocks in dataloader:
            input_features = blocks[0].srcdata['feat']
            out_features = self.gcn(blocks, input_features)
            head_aug_out_list.append(out_features)
        head_aug = torch.cat(head_aug_out_list, dim=0)
        head_aug = head_aug.view(batch_size, few_size, 2, -1)

        tail_aug_ids = tail_aug_ids.view(-1)
        dataloader = dgl.dataloading.DataLoader(
            self.g, tail_aug_ids, self.sampler,
            batch_size=self.g_batch,
            shuffle=False,
            drop_last=False,
            device=self.device,
            use_uva=True)
        tail_aug_out_list = []
        for _, _, blocks in dataloader:
            input_features = blocks[0].srcdata['feat']
            out_features = self.gcn(blocks, input_features)
            tail_aug_out_list.append(out_features)
        tail_aug = torch.cat(tail_aug_out_list, dim=0)
        tail_aug = tail_aug.view(batch_size, few_size, 2, -1)

        head_tail_aug_ids = head_tail_aug_ids.view(-1)
        dataloader = dgl.dataloading.DataLoader(
            self.g, head_tail_aug_ids, self.sampler,
            batch_size=self.g_batch,
            shuffle=False,
            drop_last=False,
            device=self.device,
            use_uva=True)
        head_tail_aug_out_list = []
        for _, _, blocks in dataloader:
            input_features = blocks[0].srcdata['feat']
            out_features = self.gcn(blocks, input_features)
            head_tail_aug_out_list.append(out_features)
        head_tail_aug = torch.cat(head_tail_aug_out_list, dim=0)
        head_tail_aug = head_tail_aug.view(batch_size, few_size, 2, -1)

        # 混合增强（向量化）
        head_aug = alpha_s * head_aug + (1 - alpha_s) * pair_emd
        tail_aug = alpha_s * tail_aug + (1 - alpha_s) * pair_emd
        head_tail_aug = alpha_s * head_tail_aug + (1 - alpha_s) * pair_emd

        # 计算所有增强类型的分数（向量化）
        rel_emd_expanded = rel_emd.expand(-1, few_size, -1, -1)  # 适配batch和few维度
        score_h = self.triple_score(head_aug.reshape(-1, 2, self.es),
                                    rel_emd_expanded.reshape(-1, self.es)).view(batch_size, few_size).mean(dim=1)
        score_t = self.triple_score(tail_aug.reshape(-1, 2, self.es),
                                    rel_emd_expanded.reshape(-1, self.es)).view(batch_size, few_size).mean(dim=1)
        score_ht = self.triple_score(head_tail_aug.reshape(-1, 2, self.es),
                                     rel_emd_expanded.reshape(-1, self.es)).view(batch_size, few_size).mean(dim=1)

        # 选择最佳增强类型
        scores = torch.stack([score_h, score_t, score_ht], dim=1)  # (batch, 3)
        best_indices = torch.argmax(scores, dim=1)  # (batch,)
        neg_aug_indices = torch.argmin(scores, dim=1)  # (batch,)
        r_type_list = [0, 1, 2]  # 映射到增强类型
        aug_types = [r_type_list[i] for i in best_indices.cpu().numpy()]
        neg_aug_types = [r_type_list[i] for i in neg_aug_indices.cpu().numpy()]

        # 选择最佳增强嵌入（使用高级索引）
        batch_indices = torch.arange(batch_size, device=self.device)
        all_aug = torch.stack([head_aug, tail_aug, head_tail_aug], dim=1)  # (batch, 3, few, 2, es)
        best_aug = all_aug[batch_indices, best_indices]  # (batch, few, 2, es)
        neg_aug = all_aug[batch_indices, neg_aug_indices]  # (batch, few, 2, es)

        # 合并原始和增强嵌入
        merged = torch.cat([pair_emd, best_aug], dim=1)  # (batch, 2*few, 2, es)

        return merged, aug_types, neg_aug, neg_aug_types
    
    def query_Aug(self, triples, rel_type, neg_aug_types):
        # 生成alpha_s（向量化）
        alpha_s = 0.5 + self.beta_weight * self.dist.sample((len(triples),)).to(self.device)
        alpha_s = alpha_s.unsqueeze(1).unsqueeze(2).unsqueeze(3)  # 适配广播维度

        # 提取所有head和tail ID（向量化）
        all_head_ids = torch.tensor([[self.ent2id_dict[x[0]] for x in batch] for batch in triples],
                                    device=self.device)  # (triples_len, few)
        all_tail_ids = torch.tensor([[self.ent2id_dict[x[2]] for x in batch] for batch in triples],
                                    device=self.device)  # (triples_len, few)

        # 构建pair ID和嵌入
        batch_size, few_size = all_head_ids.shape
        pair_ids = torch.stack([all_head_ids, all_tail_ids], dim=2)  # (batch, few, 2)
        
        pair_ids_flat = pair_ids.view(-1)
        dataloader = dgl.dataloading.DataLoader(
            self.g, pair_ids_flat, self.sampler,
            batch_size=self.g_batch,
            shuffle=False,
            drop_last=False,
            device=self.device,
            use_uva=True)
        pair_out_list = []
        for _, _, blocks in dataloader:
            input_features = blocks[0].srcdata['feat']
            out_features = self.gcn(blocks, input_features)
            pair_out_list.append(out_features)
        pair_emb = torch.cat(pair_out_list, dim=0)
        pair_emd = pair_emb.view(batch_size, few_size, 2, -1)

        # 获取最近邻（向量化）
        nearest_h = self.nearest_cache[all_head_ids, 0]  # (batch, few)
        nearest_t = self.nearest_cache[all_tail_ids, 0]  # (batch, few)

        # 创建增强ID的基础张量，使用正确的形状 (batch, few, 2)
        aug_ids = torch.zeros_like(pair_ids)  # (batch, few, 2)
        neg_aug_ids = torch.zeros_like(pair_ids)  # (batch, few, 2)

        # 向量化处理不同增强类型
        rel_type_tensor = torch.tensor(rel_type, device=self.device)  # (batch,)
        neg_aug_types_tensor = torch.tensor(neg_aug_types, device=self.device)  # (batch,)

        # 使用掩码选择不同增强类型的ID组合，形状为 (batch, few, 1) 以匹配 (batch, few, 2)
        mask0 = (rel_type_tensor == 0).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1) -> 广播到 (batch, few, 2)
        mask1 = (rel_type_tensor == 1).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1)
        mask2 = (rel_type_tensor == 2).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1)
        
        neg_mask0 = (neg_aug_types_tensor == 0).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1)
        neg_mask1 = (neg_aug_types_tensor == 1).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1)
        neg_mask2 = (neg_aug_types_tensor == 2).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1)

        # 填充不同增强类型的ID
        aug_ids = torch.where(mask0,
                              torch.stack([nearest_h, all_tail_ids], dim=2),
                              aug_ids)
        aug_ids = torch.where(mask1,
                              torch.stack([all_head_ids, nearest_t], dim=2),
                              aug_ids)
        aug_ids = torch.where(mask2,
                              torch.stack([nearest_h, nearest_t], dim=2),
                              aug_ids)
        
        neg_aug_ids = torch.where(neg_mask0,
                              torch.stack([nearest_h, all_tail_ids], dim=2),
                              neg_aug_ids)
        neg_aug_ids = torch.where(neg_mask1,
                              torch.stack([all_head_ids, nearest_t], dim=2),
                              neg_aug_ids)
        neg_aug_ids = torch.where(neg_mask2,
                              torch.stack([nearest_h, nearest_t], dim=2),
                              neg_aug_ids)

        # 获取增强嵌入并混合
        
        aug_ids = aug_ids.view(-1)
        dataloader = dgl.dataloading.DataLoader(
            self.g, aug_ids, self.sampler,
            batch_size=self.g_batch,
            shuffle=False,
            drop_last=False,
            device=self.device,
            use_uva=True)
        aug_out_list = []
        for _, _, blocks in dataloader:
            input_features = blocks[0].srcdata['feat']
            out_features = self.gcn(blocks, input_features)
            aug_out_list.append(out_features)
        aug_emd = torch.cat(aug_out_list, dim=0)
        aug_emd = aug_emd.view(batch_size, few_size, 2, -1)
        aug_emd = alpha_s * aug_emd + (1 - alpha_s) * pair_emd

        neg_aug_ids = neg_aug_ids.view(-1)
        dataloader = dgl.dataloading.DataLoader(
            self.g, neg_aug_ids, self.sampler,
            batch_size=self.g_batch,
            shuffle=False,
            drop_last=False,
            device=self.device,
            use_uva=True)
        neg_aug_out_list = []
        for _, _, blocks in dataloader:
            input_features = blocks[0].srcdata['feat']
            out_features = self.gcn(blocks, input_features)
            neg_aug_out_list.append(out_features)
        neg_aug_emd = torch.cat(neg_aug_out_list, dim=0)
        neg_aug_emd = neg_aug_emd.view(batch_size, few_size, 2, -1)
        neg_aug_emd = alpha_s * neg_aug_emd + (1 - alpha_s) * pair_emd

        # 合并嵌入
        merged = torch.cat([pair_emd, aug_emd], dim=1)  # (batch, 2*few, 2, es)

        return merged, neg_aug_emd



    def forward(self, triples):
        '''
        inputs:
            task: Batch triplets, B * few
        outputs:
            emb: B * few * es
        '''

        idx = self.ent2id(triples)
        batch_size, few_shot = idx.shape[0], idx.shape[1]
        idx = idx.view(-1)
        dataloader = dgl.dataloading.DataLoader(
            self.g, idx, self.sampler,
            batch_size=self.g_batch,
            shuffle=False,
            drop_last=False,
            device=self.device,
            use_uva=True)
        out_emb = []
        for input_nodes, output_nodes, blocks in dataloader:
            input_features = blocks[0].srcdata['feat']
            out_features = self.gcn(blocks, input_features)
            out_emb.append(out_features)
        out_emb = torch.cat(out_emb, dim=0)
        out_emb = out_emb.view(batch_size, few_shot, 2, -1)
        return out_emb
    def forward_embed(self, triples_id):
        '''
        inputs:
            task: Batch triplets, B * few
        outputs:
            emb: B * few * es
        '''

        idx = triples_id
        batch_size, few_shot = idx.shape[0], idx.shape[1]
        idx = idx.view(-1)
        dataloader = dgl.dataloading.DataLoader(
            self.g, idx, self.sampler,
            batch_size=self.g_batch,
            shuffle=False,
            drop_last=False,
            device=self.device,
            use_uva=True)
        out_emb = []
        for input_nodes, output_nodes, blocks in dataloader:
            input_features = blocks[0].srcdata['feat']
            out_features = self.gcn(blocks, input_features)
            out_emb.append(out_features)
        out_emb = torch.cat(out_emb, dim=0)
        out_emb = out_emb.view(batch_size, few_shot, 2, -1)
        return out_emb

    


class StochasticTwoLayerGCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.conv1 = dglnn.GraphConv(in_features, hidden_features, allow_zero_in_degree=True)
        self.conv2 = dglnn.GraphConv(hidden_features, out_features, allow_zero_in_degree=True)

    def forward(self, blocks, x):
        x = F.relu(self.conv1(blocks[0], x))
        x = F.relu(self.conv2(blocks[1], x))
        return x


class RPGNN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, hop, n_rel):
        super().__init__()
        emb_dim = in_features
        self.conv_in = RPLayer(emb_dim, in_features, hidden_features, n_rel)
        self.conv_out = RPLayer(emb_dim, hidden_features, out_features, n_rel)
        self.hop = hop
        if hop > 2:
            self.conv_hidden = nn.ModuleList(
                [RPLayer(emb_dim, hidden_features, hidden_features, n_rel) for _ in range(hop - 2)])

    def forward(self, blocks, x):
        x = F.relu(self.conv_in(blocks[0], x))
        if self.hop > 2:
            for i, conv in enumerate(self.conv_hidden):
                x = F.relu(conv(blocks[i + 1], x))
        x = F.relu(self.conv_out(blocks[-1], x))
        return x


class RPLayer(nn.Module):
    def __init__(self, emb_dim, in_feat, out_feat, num_rels):
        super().__init__()
        self.num_rels = num_rels
        self.linear_r = dgl.nn.pytorch.TypedLinear(in_feat + emb_dim * 2, out_feat, num_rels)
        self.attn_fc = nn.Linear(emb_dim + out_feat, 1, bias=False)
        self.h_bias = nn.Parameter(torch.Tensor(out_feat))
        self.loop_weight = nn.Parameter(torch.Tensor(emb_dim, out_feat))
        nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))

    def edge_agg(self, edges):
        """Relation Message Passing"""
        x = torch.cat([edges.src['h'], edges.data['feat'], edges.dst['feat']], dim=1)
        m = self.linear_r(x, edges.data['eid'])
        attn = F.leaky_relu(self.attn_fc(torch.cat([edges.dst['feat'], m], dim=1)))
        return {'h': m, 'z': attn}

    def forward(self, g, feat):
        with g.local_scope():
            # Norm
            degs = g.out_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = torch.reshape(norm, shp)
            feat = feat * norm
            g.srcdata['h'] = feat
            g.apply_edges(self.edge_agg)
            e = g.edata.pop('z')
            a = edge_softmax(g, e)
            g.edata['h'] = a * g.edata['h']
            g.update_all(dgl.function.copy_e('h', 'm'), dgl.function.sum('m', 'h'))
            h = g.dstdata['h']
            h = h + g.dstdata['feat'] @ self.loop_weight
            # Norm 
            degs = g.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            shp = norm.shape + (1,) * (h.dim() - 1)
            norm = torch.reshape(norm, shp)
            rst = h * norm
            h = rst + self.h_bias

            return h

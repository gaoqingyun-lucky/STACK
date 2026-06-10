import torch
import torch.nn as nn
import torch.nn.functional as F
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


class Embedding(nn.Module):
    def __init__(self, dataset, parameter):
        super().__init__()

        self.device = parameter['device']
        self.ent2id = dataset['ent2id']
        self.nearest_dict = dataset['nearest_dict']
        self.rel2id = dataset['rel2id']
        self.num_ent = len(self.ent2id)
        self.es = parameter['embed_dim']
        self.aug_hum = parameter['aug_hum']
        self.few = parameter['few'] * (self.aug_hum + 1)

        # 初始化 embedding
        self.embedding = nn.Embedding(self.num_ent, self.es).to(self.device)
        self.proj_rel_embed = MLP(self.es)

        if parameter['data_form'] == 'Pre-Train':
            ent2emb = dataset['ent2emb']
            self.embedding.weight.data.copy_(torch.from_numpy(ent2emb))
        else:
            nn.init.xavier_uniform_(self.embedding.weight)

        # Beta 参数
        self.beta_weight = nn.Parameter(torch.tensor(0.1))
        self.dist = Normal(0, 1)

        # 缓存
        self.build_cache(num_neighbors=self.aug_hum)

    # 构建缓存
    def build_cache(self, num_neighbors=1):
        """缓存实体 embedding 和 top-z 最近邻到 GPU"""
        device = torch.device(self.device)

        # 确保 embedding 在目标 device
        self.embedding.to(device)

        # 所有实体 id
        all_ids = torch.arange(self.num_ent, dtype=torch.long, device=device)

        # 缓存 embedding
        self.ent_cache = self.embedding(all_ids)  # (num_ent, es)

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

        print(f"[Cache built] ent_cache: {self.ent_cache.shape}, nearest_cache: {self.nearest_cache.shape}")

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

    # 支持集增强 - 优化版
    def support_Aug(self, triples, rel_emd):
        aug_hum = self.aug_hum  # 获取近邻数量
        # 生成alpha_s（向量化）
        alpha_s = 0.5 + self.beta_weight * self.dist.sample((len(triples),)).to(self.device)
        alpha_s = alpha_s.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)  # 适配广播维度

        # 提取所有head和tail ID（向量化）
        all_head_ids = torch.tensor([[self.ent2id[x[0]] for x in batch] for batch in triples],
                                    device=self.device)  # (triples_len, few)
        all_tail_ids = torch.tensor([[self.ent2id[x[2]] for x in batch] for batch in triples],
                                    device=self.device)  # (triples_len, few)

        # 构建pair ID和嵌入
        batch_size, few_size = all_head_ids.shape
        pair_ids = torch.stack([all_head_ids, all_tail_ids], dim=2)  # (batch, few, 2)
        pair_emd = self.ent_cache[pair_ids]  # (batch, few, 2, es)
        pair_emd_expanded = pair_emd.unsqueeze(2).expand(-1, -1, aug_hum, -1, -1)# (batch, few, aug_hum, 2, es)

        # 获取最近邻（向量化）
        nearest_h = self.nearest_cache[all_head_ids]  # (batch, few, aug_hum)
        nearest_t = self.nearest_cache[all_tail_ids]  # (batch, few, aug_hum)

        # 生成所有增强类型的嵌入（一次性计算所有可能的增强）
        # head_aug：替换head为其aug_hum个近邻，tail保持不变
        tail_expanded = all_tail_ids.unsqueeze(-1).expand(-1, -1, aug_hum)  # (batch, few, aug_hum)
        head_aug_ids = torch.stack([nearest_h, tail_expanded], dim=-1)  # (batch, few, aug_hum, 2)
        # tail_aug：head不变，tail替换为其aug_hum个近邻
        head_expanded = all_head_ids.unsqueeze(-1).expand(-1, -1, aug_hum)  # (batch, few, aug_hum)
        tail_aug_ids = torch.stack([head_expanded, nearest_t], dim=-1)  # (batch, few, aug_hum, 2)
        # head_tail_aug：head和tail都替换为各自的aug_hum个近邻
        head_tail_aug_ids = torch.stack([nearest_h, nearest_t], dim=-1)  # (batch, few, aug_hum, 2)

        # 获取增强嵌入
        head_aug = self.ent_cache[head_aug_ids] # (batch, few, aug_hum, 2, es)
        tail_aug = self.ent_cache[tail_aug_ids] # (batch, few, aug_hum, 2, es)
        head_tail_aug = self.ent_cache[head_tail_aug_ids] # (batch, few, aug_hum, 2, es)

        # 混合增强（向量化）
        head_aug = alpha_s * head_aug + (1 - alpha_s) * pair_emd_expanded
        tail_aug = alpha_s * tail_aug + (1 - alpha_s) * pair_emd_expanded
        head_tail_aug = alpha_s * head_tail_aug + (1 - alpha_s) * pair_emd_expanded

        # 计算所有增强类型的分数（适配aug_hum维度）
        # 调整rel_emd维度：适配batch、few、aug_hum
        rel_emd_expanded = rel_emd.expand(batch_size, few_size, aug_hum, -1)  # (batch, few, aug_hum, es)
        rel_emd_reshaped = rel_emd_expanded.reshape(-1, self.es)  # (batch*few*aug_hum, es)

        # 计算head_aug分数：先reshape→计算→还原维度→平均aug_hum→平均few
        head_aug_reshaped = head_aug.reshape(-1, 2, self.es)  # (batch*few*aug_hum, 2, es)
        score_h = self.triple_score(head_aug_reshaped, rel_emd_reshaped)
        score_h = score_h.view(batch_size, few_size, aug_hum).mean(dim=-1).mean(dim=1)  # (batch,)

        # 计算tail_aug分数
        tail_aug_reshaped = tail_aug.reshape(-1, 2, self.es)
        score_t = self.triple_score(tail_aug_reshaped, rel_emd_reshaped)
        score_t = score_t.view(batch_size, few_size, aug_hum).mean(dim=-1).mean(dim=1)  # (batch,)

        # 计算head_tail_aug分数
        head_tail_aug_reshaped = head_tail_aug.reshape(-1, 2, self.es)
        score_ht = self.triple_score(head_tail_aug_reshaped, rel_emd_reshaped)
        score_ht = score_ht.view(batch_size, few_size, aug_hum).mean(dim=-1).mean(dim=1)  # (batch,)


        # 选择最佳增强类型
        scores = torch.stack([score_h, score_t, score_ht], dim=1)  # (batch, 3)
        best_indices = torch.argmax(scores, dim=1)  # (batch,)
        neg_aug_indices = torch.argmin(scores, dim=1)  # (batch,)
        r_type_list = [0, 1, 2]  # 映射到增强类型 0:head_aug, 1:tail_aug, 2:head_tail_aug
        aug_types = [r_type_list[i] for i in best_indices.cpu().numpy()]
        neg_aug_types = [r_type_list[i] for i in neg_aug_indices.cpu().numpy()]

        # -------------------------- 核心修改部分 --------------------------
        # 维度重塑：(batch, few, aug_hum, 2, es) → (batch, few*aug_hum, 2, es)
        # 替代原本的mean(dim=2)平均操作
        head_aug_reshaped = head_aug.reshape(batch_size, few_size * aug_hum, 2, self.es)
        tail_aug_reshaped = tail_aug.reshape(batch_size, few_size * aug_hum, 2, self.es)
        head_tail_aug_reshaped = head_tail_aug.reshape(batch_size, few_size * aug_hum, 2, self.es)
        # -----------------------------------------------------------------



        # 选择最佳增强嵌入（使用高级索引）
        all_aug = torch.stack([head_aug_reshaped, tail_aug_reshaped, head_tail_aug_reshaped], dim=1)
        batch_indices = torch.arange(batch_size, device=self.device)
        best_aug = all_aug[batch_indices, best_indices]  # (batch, few*aug_hum, 2, es)
        neg_aug = all_aug[batch_indices, neg_aug_indices]  # (batch, few*aug_hum, 2, es)

        # 合并原始和增强嵌入
        merged = torch.cat([pair_emd, best_aug], dim=1)  # (batch, few + few*aug_hum, 2, es)

        return merged, aug_types, neg_aug, neg_aug_types

    # 查询集增强 - 优化版
    def query_Aug(self, triples, rel_type, neg_aug_types):
        aug_hum = self.aug_hum
        # 生成alpha_s（向量化）
        alpha_s = 0.5 + self.beta_weight * self.dist.sample((len(triples),)).to(self.device)
        alpha_s = alpha_s.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)  # 适配广播维度

        # 提取所有head和tail ID（向量化）
        all_head_ids = torch.tensor([[self.ent2id[x[0]] for x in batch] for batch in triples],
                                    device=self.device)  # (triples_len, few)
        all_tail_ids = torch.tensor([[self.ent2id[x[2]] for x in batch] for batch in triples],
                                    device=self.device)  # (triples_len, few)

        # 构建pair ID和嵌入
        batch_size, few_size = all_head_ids.shape
        pair_ids = torch.stack([all_head_ids, all_tail_ids], dim=2)  # (batch, few, 2)
        pair_emd = self.ent_cache[pair_ids]  # (batch, few, 2, es)
        pair_emd_expanded = pair_emd.unsqueeze(2).expand(-1, -1, aug_hum, -1, -1)  # (batch, few, aug_hum, 2, es)

        # 获取最近邻（向量化）
        nearest_h = self.nearest_cache[all_head_ids]  #  (batch, few, aug_hum)
        nearest_t = self.nearest_cache[all_tail_ids]  #  (batch, few, aug_hum)

        # -------------------------- 生成增强ID（适配aug_hum维度） --------------------------
        # 生成三种增强类型的ID（和support_Aug逻辑对齐）
        # 0: head_aug → head替换为近邻，tail不变
        tail_expanded = all_tail_ids.unsqueeze(-1).expand(-1, -1, aug_hum)  # (batch, few, aug_hum)
        head_aug_ids = torch.stack([nearest_h, tail_expanded], dim=-1)  # (batch, few, aug_hum, 2)
        # 1: tail_aug → head不变，tail替换为近邻
        head_expanded = all_head_ids.unsqueeze(-1).expand(-1, -1, aug_hum)  # (batch, few, aug_hum)
        tail_aug_ids = torch.stack([head_expanded, nearest_t], dim=-1)  # (batch, few, aug_hum, 2)
        # 2: head_tail_aug → head和tail都替换为近邻
        head_tail_aug_ids = torch.stack([nearest_h, nearest_t], dim=-1)  # (batch, few, aug_hum, 2)

        # 整合所有增强类型的ID
        all_aug_ids = torch.stack([head_aug_ids, tail_aug_ids, head_tail_aug_ids], dim=1)  # (batch, 3, few, aug_hum, 2)
        all_neg_aug_ids = torch.stack([head_aug_ids, tail_aug_ids, head_tail_aug_ids], dim=1)  # (batch, 3, few, aug_hum, 2)


        # 根据rel_type选择对应增强类型的ID（正向增强）
        rel_type_tensor = torch.tensor(rel_type, device=self.device)  # (batch,)
        batch_indices = torch.arange(batch_size, device=self.device)
        aug_ids = all_aug_ids[batch_indices, rel_type_tensor]  # (batch, few, aug_hum, 2)

        # 根据neg_aug_types选择对应增强类型的ID（负向增强）
        neg_aug_types_tensor = torch.tensor(neg_aug_types, device=self.device)  # (batch,)
        neg_aug_ids = all_neg_aug_ids[batch_indices, neg_aug_types_tensor]  # (batch, few, aug_hum, 2)
        
         # -------------------------- 生成增强嵌入并混合 --------------------------
        # 获取增强嵌入
        aug_emd = self.ent_cache[aug_ids]  # (batch, few, aug_hum, 2, es)
        neg_aug_emd = self.ent_cache[neg_aug_ids]  # (batch, few, aug_hum, 2, es)
        aug_emd = alpha_s * aug_emd + (1 - alpha_s) * pair_emd_expanded  # (batch, few, aug_hum, 2, es)
        neg_aug_emd = alpha_s * neg_aug_emd + (1 - alpha_s) * pair_emd_expanded  # (batch, few, aug_hum, 2, es)
        aug_emd_reshaped = aug_emd.reshape(batch_size, few_size * aug_hum, 2, self.es)
        neg_aug_emd = neg_aug_emd.reshape(batch_size, few_size * aug_hum, 2, self.es)  
        merged = torch.cat([pair_emd, aug_emd_reshaped], dim=1)  # (batch, few + few*aug_hum, 2, es)
        
        return merged, neg_aug_emd

    def forward(self, triples):
        idx = [[[self.ent2id[t[0]], self.ent2id[t[2]]] for t in batch] for batch in triples]
        idx = torch.LongTensor(idx).to(self.device)
        return self.embedding(idx)
import numpy as np
import torch
import torch.nn as nn
import dgl
from dgl.nn.pytorch import PNAConv
from torch.distributions.normal import Normal
from Encoder.GAT import EfficientGAT
"""
针对不同强化学习算法的Actor与Critic，使用了Encoder、Decoder中的深度学习算法实现、也包括了一些基本的MLP等网络
"""

""" Actor部分: 主要在于通过state选择action"""


class ActorPPO(nn.Module):
    """Discrete version"""

    def __init__(self, mid_dim, num_layers, state_dim, action_dim, gat_args=None):
        super(ActorPPO, self).__init__()
        self.net = build_mlp(mid_dim, num_layers, state_dim, action_dim)  # 后面可以被其他encoder替代
        # self.net = EfficientGAT(num_of_layers=2, num_head_per_layers=[8, 1], num_features_per_layers=[2, 64, 64],
        #                         add_skip_connection=True, bias=True, dropout=0.6)
        self.action_std_log = nn.Parameter(torch.zeros((1, action_dim)) - 0.5, requires_grad=True)
        self.soft_max = nn.Softmax(dim=-1)
        self.log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))
        self.Categorical = torch.distributions.Categorical

    def forward(self, state):
        return self.net(state)  # action prob without softmax

    def get_action(self, state):
        #
        # modified, discrete version, TODO
        prob = self.net(state).squeeze(-1)
        actions_prob = self.soft_max(prob)
        action = torch.multinomial(actions_prob, num_samples=1, replacement=True)
        # action_std = self.action_std_log.exp()
        # dist = Normal(action_avg, action_std)
        # action = dist.sample()
        # log_prob = dist.log_prob(action).sum(1)
        return action, actions_prob.gather(dim=-1, index=action)

    def get_logprob(self, state, action):
        action_avg = self.net(state)
        action_std = self.action_std_log.exp()

        delta = ((action_avg - action) / action_std).pow(2).__mul__(0.5)
        log_prob = -(self.action_std_log + self.log_sqrt_2pi + delta)
        return log_prob

    def get_logprob_entropy(self, state, action):
        # modified
        # action_avg = self.net(state)
        # action_std = self.action_std_log.exp()
        #
        # delta = ((action_avg - action) / action_std).pow(2) * 0.5
        # log_prob = -(self.action_std_log + self.log_sqrt_2pi + delta).sum(1)

        a_prob = self.soft_max(self.net(state).squeeze(-1))
        dist = self.Categorical(a_prob)
        dist_entropy = dist.entropy().mean()
        log_prob = dist.log_prob(action.squeeze(1))
        return log_prob, dist_entropy

    def get_old_logprob(self, action, a_prob):
        dist = self.Categorical(action)
        old_log_prob = dist.log_prob(action.squeeze(1))
        return old_log_prob
        # delta = noise.pow(2).__mul__(0.5)
        # return -(self.action_std_log + self.log_sqrt_2pi + delta).sum(1)

    @staticmethod
    def convert_action_for_env(action):
        return action.int()


class ActorGATBatch(nn.Module):
    def __init__(self, device, gat_args=None):
        super().__init__()
        self.device = device

        if gat_args is None:
            n_layers = 3
            n_head_per_layers = [8, 8, 1]
            in_fea_dim = 3
            out_fea_dim = 64
            n_features_per_layers = [in_fea_dim, 64, 128, out_fea_dim]
            n_jobs = 6
            self.n_tasks = 60
        else:
            n_layers = gat_args["n_layers"]
            n_head_per_layers = gat_args["n_head_per_layers"]
            in_fea_dim = gat_args["in_fea_dim"]
            out_fea_dim = gat_args["out_fea_dim"]
            n_features_per_layers = gat_args["n_features_per_layers"]
            n_jobs = gat_args["jobs_num"]
            self.n_tasks = gat_args["tasks_num"]

        # gat to get the embedding of all nodes (nxn, nxFIN) -> nxFOUT,  nxn->adj, nxFIN->node features, FIN = state_dim, FOUT = embedding_dim = 64
        self.gat = EfficientGAT(num_of_layers=n_layers, num_head_per_layers=n_head_per_layers,
                                num_features_per_layers=n_features_per_layers, add_skip_connection=True, bias=True,
                                dropout=0.6)

        # proj mat (n_jobs, n_tasks) all node_fea (batch_size, n_tasks, FOUT) -> pooling feat (batch_size, n_jobs, FOUT)
        # self.pooling_proj = nn.Parameter(torch.ones((1, n_tasks), requires_grad=True))     # TODO 这样就不能泛化到不同size的实例了，需要修改
        # mlp to get the prob of candidate prob
        self.get_action_prob = build_mlp(256, 3, out_fea_dim, 1)
        self.soft_max = nn.Softmax(dim=-1)
        self.categorical = torch.distributions.Categorical

    def forward(self, states):
        """
        从states中得到每个candidates的概率
        :param states:
        :return:
        """
        # state (batch_size, n, n+FIN)
        # 将adj与fea放在同一个tensor中传入 前n列是adj，后FIN列是node_fea
        # 首先拆分adj与fea
        # batch_size = states.shape[0]
        n_nodes = states.shape[0]
        # adj fea 可以将一个batch的图 作为一个大图输入，因为单个图之间不连通，因此不会有影响，adj不能简单的reshape，(B, N, N) -> (BN, BN)
        adj = states[:, :n_nodes]  # adj (B, N, N) -> (BN, BN) block_diag
        # adj = torch.block_diag(*adj).to_sparse_coo().indices()  # 每单个图不连通，大图就是一个分块对角矩阵      TODO 显存瓶颈
        # temp_adj = adj.to_sparse_coo().indices()
        # temp_adj[1] += temp_adj[0] * n_nodes
        # temp_adj[2] += temp_adj[0] * n_nodes
        adj = adj.to_sparse_coo().indices()

        fea = states[:, n_nodes:]  # reshape fea (B, N, FIN) -> (BN, FIN)

        # batch_id = torch.where(fea[:, :, -1] == 1.0)[0]
        # fea = fea.reshape(-1, fea.shape[-1])
        candidates_id = torch.where(fea[:, -1] == 1.0)[0]  # 选出candidate node idx (C, )    一般不为BJ，而是 C <= BJ

        # candidates = (candidates_id % n_nodes).reshape(batch_size, -1)      # (B, J)    TODO 事实上每个batch中job的执行情况不同，候选动作的数量其实也不一样

        # adj 图的邻接矩阵边的向量， fea 顶点的feature
        nodes_embedding, edge_index = self.gat((fea, adj))  # (BN, FOUT)
        candidates_feature = torch.index_select(nodes_embedding, dim=0, index=candidates_id)  # (C, FOUT)      一般不为BJ
        # nodes_embedding = nodes_embedding.reshape(batch_size, -1, nodes_embedding.shape[-1])   # (B, N, FOUT)
        # candidates_feature = candidates_feature.reshape(batch_size, -1, candidates_feature.shape[-1])  # (B, J, FOUT)

        # feature_pooling = torch.matmul(self.pooling_proj, nodes_embedding)        # (B, J, FOUT)
        # feature_pooling = feature_pooling.expand_as(candidates_feature)
        # candidate_graph_fea = torch.cat((candidates_feature, feature_pooling), dim=-1)      # (B, J, 2*FOUT)
        candidates_prob = self.get_action_prob(candidates_feature).squeeze(-1)  # (C, 1)

        # candidates_prob, candidates = self.split_batch(candidates_prob, candidates_id, batch_size, batch_id)

        return candidates_prob, candidates_id

    def get_action(self, states):
        batch_size = states.shape[0]
        actions = torch.zeros(batch_size, dtype=torch.int32).to(self.device)
        actions_prob = torch.zeros(batch_size, dtype=torch.float32).to(self.device)
        actions_id = torch.zeros(batch_size, dtype=torch.float32).to(self.device)
        for i, state in enumerate(states):
            candidate_prob, candidate = self.forward(state)
            prob = self.soft_max(candidate_prob)
            action_id = torch.multinomial(prob, 1, replacement=False)

            actions[i] = candidate[action_id]
            actions_prob[i] = prob[action_id]
            actions_id[i] = action_id

        # candidates_prob = self.soft_max(candidates_prob)
        # actions_id = torch.multinomial(candidates_prob, 1, replacement=True)
        # actions = candidates.gather(-1, index=actions_id)
        # actions_prob = candidates_prob.gather(dim=-1, index=actions_id)
        return actions, actions_prob, actions_id

    def greedy_get_action(self, state):
        """单个state, 测试时选择贪心的选择概率最大的action"""
        candidates_prob, candidates = self.forward(state)
        prob = self.soft_max(candidates_prob[0])
        action_prob, action_id = prob.max(0)
        return candidates[0][action_id]

    def get_logprob_entropy(self, states, actions_id):
        batch_size = states.shape[0]

        # candidates_prob = torch.zeros(batch_size, dtype=torch.float32).to(self.device)

        log_probs = torch.zeros(batch_size, dtype=torch.float32).to(self.device)
        entropy = torch.zeros(batch_size, dtype=torch.float32).to(self.device)

        # candidates_prob, _ = self.forward(states)

        for i in range(batch_size):
            prob, _ = self.forward(states[i])
            dist = self.categorical(self.soft_max(prob))
            log_probs[i] = dist.log_prob(actions_id[i].squeeze(-1))
            entropy[i] = dist.entropy().mean()

        return log_probs, entropy

    def get_logprob(self, states, actions):
        candidates_prob, _ = self.forward(states)
        candidates_prob = self.soft_max(candidates_prob)
        dist = self.categorical(candidates_prob)

        log_prob = dist.log_prob(actions.squeeze(-1))
        return log_prob

    def get_old_logprob(self, actions, actions_prob):
        dist = self.categorical(actions_prob)
        old_log_prob = dist.log_prob(actions.squeeze(-1))
        return old_log_prob

    @staticmethod
    def convert_action_for_env(action):
        return action.int()


class ActorGAT(nn.Module):
    def __init__(self, device, gat_args=None):
        super().__init__()
        self.device = device

        # n_layers = getattr(gat_args, "n_layers", 3)
        # n_head_per_layers = getattr(gat_args, "n_head_per_layers", [8, 8, 1])
        # in_fea_dim = getattr(gat_args, "in_fea_dim", 3)
        # out_fea_dim = getattr(gat_args, "out_fea_dim", 64)
        # n_features_per_layers = getattr(gat_args, "n_features_per_layers", [in_fea_dim, 64, 128, out_fea_dim])
        # n_jobs = getattr(gat_args, "n_jobs", 20)
        # self.n_tasks = getattr(gat_args, "n_tasks", 300)

        if gat_args is None:
            n_layers = 3
            n_head_per_layers = [8, 8, 1]
            in_fea_dim = 3
            out_fea_dim = 64
            n_features_per_layers = [in_fea_dim, 64, 128, out_fea_dim]
            n_jobs = 6
            self.n_tasks = 60
        else:
            n_layers = gat_args["n_layers"]
            n_head_per_layers = gat_args["n_head_per_layers"]
            in_fea_dim = gat_args["in_fea_dim"]
            out_fea_dim = gat_args["out_fea_dim"]
            n_features_per_layers = gat_args["n_features_per_layers"]
            n_jobs = gat_args["jobs_num"]
            self.n_tasks = gat_args["tasks_num"]

        # gat to get the embedding of all nodes (nxn, nxFIN) -> nxFOUT,  nxn->adj, nxFIN->node features, FIN = state_dim, FOUT = embedding_dim = 64
        self.gat = EfficientGAT(num_of_layers=n_layers, num_head_per_layers=n_head_per_layers,
                                num_features_per_layers=n_features_per_layers, add_skip_connection=True, bias=True,
                                dropout=0.6)

        # proj mat (n_jobs, n_tasks) all node_fea (batch_size, n_tasks, FOUT) -> pooling feat (batch_size, n_jobs, FOUT)
        # self.pooling_proj = nn.Parameter(torch.ones((1, n_tasks), requires_grad=True))     # TODO 这样就不能泛化到不同size的实例了，需要修改
        # mlp to get the prob of candidate prob
        self.get_action_prob = build_mlp(256, 3, out_fea_dim, 1)
        self.soft_max = nn.Softmax(dim=-1)
        self.categorical = torch.distributions.Categorical

    def forward(self, states):
        """
        从states中得到每个candidates的概率
        :param states:
        :return:
        """
        # state (batch_size, n, n+FIN)
        # 将adj与fea放在同一个tensor中传入 前n列是adj，后FIN列是node_fea
        # 首先拆分adj与fea
        batch_size = states.shape[0]
        n_nodes = states.shape[1]
        # adj fea 可以将一个batch的图 作为一个大图输入，因为单个图之间不连通，因此不会有影响，adj不能简单的reshape，(B, N, N) -> (BN, BN)
        adj = states[:, :, :n_nodes]  # adj (B, N, N) -> (BN, BN) block_diag
        # adj = torch.block_diag(*adj).to_sparse_coo().indices()  # 每单个图不连通，大图就是一个分块对角矩阵      TODO 显存瓶颈
        temp_adj = adj.to_sparse().indices()
        temp_adj[1] += temp_adj[0] * n_nodes
        temp_adj[2] += temp_adj[0] * n_nodes
        adj = temp_adj[1:]

        fea = states[:, :, n_nodes:]  # reshape fea (B, N, FIN) -> (BN, FIN)

        batch_id = torch.where(fea[:, :, -1] == 1.0)[0]

        fea = fea.reshape(-1, fea.shape[-1])

        candidates_id = torch.where(fea[:, -1] == 1.0)[0]  # 选出candidate node idx (C, )    一般不为BJ，而是 C <= BJ

        # candidates = (candidates_id % n_nodes).reshape(batch_size, -1)      # (B, J)    TODO 事实上每个batch中job的执行情况不同，候选动作的数量其实也不一样

        # adj 图的邻接矩阵边的向量， fea 顶点的feature
        nodes_embedding, edge_index = self.gat((fea, adj))  # (BN, FOUT)
        candidates_feature = torch.index_select(nodes_embedding, dim=0, index=candidates_id)  # (C, FOUT)      一般不为BJ
        # nodes_embedding = nodes_embedding.reshape(batch_size, -1, nodes_embedding.shape[-1])   # (B, N, FOUT)
        # candidates_feature = candidates_feature.reshape(batch_size, -1, candidates_feature.shape[-1])  # (B, J, FOUT)

        # feature_pooling = torch.matmul(self.pooling_proj, nodes_embedding)        # (B, J, FOUT)
        # feature_pooling = feature_pooling.expand_as(candidates_feature)
        # candidate_graph_fea = torch.cat((candidates_feature, feature_pooling), dim=-1)      # (B, J, 2*FOUT)
        candidates_prob = self.get_action_prob(candidates_feature).squeeze(-1)  # (C, 1)

        candidates_prob, candidates = self.split_batch(candidates_prob, candidates_id, batch_size, batch_id, n_nodes)

        return candidates_prob, candidates

    def get_action(self, states):
        batch_size = states.shape[0]
        candidates_prob, candidates = self.forward(states)

        actions = torch.zeros(batch_size, dtype=torch.int32).to(self.device)
        actions_prob = torch.zeros(batch_size, dtype=torch.float32).to(self.device)
        actions_id = torch.zeros(batch_size, dtype=torch.float32).to(self.device)
        for i in range(batch_size):
            cand_prob = self.soft_max(candidates_prob[i])
            dist = self.categorical(cand_prob)
            action_id = dist.sample()       # 训练时会根据概率采样action，有可能不是概率最大的action
            actions[i] = candidates[i][action_id]
            actions_prob[i] = cand_prob[action_id]
            actions_id[i] = action_id
        # candidates_prob = self.soft_max(candidates_prob)
        # actions_id = torch.multinomial(candidates_prob, 1, replacement=True)
        # actions = candidates.gather(-1, index=actions_id)
        # actions_prob = candidates_prob.gather(dim=-1, index=actions_id)
        return actions, actions_prob, actions_id

    def greedy_get_action(self, state):
        """单个state, 测试时选择贪心的选择概率最大的action"""
        candidates_prob, candidates = self.forward(state)
        prob = self.soft_max(candidates_prob[0])
        action_prob, action_id = prob.max(0)
        return candidates[0][action_id]


    def get_logprob_entropy(self, states, actions_id):
        batch_size = states.shape[0]
        log_probs = torch.zeros(batch_size, dtype=torch.float32).to(self.device)
        entropy = torch.zeros(batch_size, dtype=torch.float32).to(self.device)

        candidates_prob, _ = self.forward(states)

        for i in range(batch_size):
            dist = self.categorical(self.soft_max(candidates_prob[i]))
            log_probs[i] = dist.log_prob(actions_id[i].squeeze(-1))
            entropy[i] = dist.entropy().mean()

        return log_probs, entropy

    def get_logprob(self, states, actions):
        candidates_prob, _ = self.forward(states)
        candidates_prob = self.soft_max(candidates_prob)
        dist = self.categorical(candidates_prob)

        log_prob = dist.log_prob(actions.squeeze(-1))
        return log_prob

    def get_old_logprob(self, actions, actions_prob):
        dist = self.categorical(actions_prob)
        old_log_prob = dist.log_prob(actions.squeeze(-1))
        return old_log_prob

    def split_batch(self, prob, candidates_id, batch_size, batch_id, n_tasks):
        # 获得每个batch中candidates的位置
        candidates_prob_each_batch = []
        candidates_each_batch = []
        for i in range(batch_size):
            idx = torch.where(batch_id == i)[0]
            candidates_prob_each_batch.append(prob[idx])
            candidates_each_batch.append(candidates_id[idx] % n_tasks)

        return candidates_prob_each_batch, candidates_each_batch

    @staticmethod
    def convert_action_for_env(action):
        return action.int()


class ActorHGAT(nn.Module):
    """
    柔性作业车间调度问题中提取异构图的信息
    """
    def __init__(self, device, gat_args=None):
        self.device = device

        n_layers = 3
        n_head_per_layers = [8, 8, 1]
        task_in_fea_dim = 4
        mach_in_fea_dim = 100
        out_fea_dim = 256
        n_features_per_layers = [out_fea_dim, 64, 128, out_fea_dim]
        n_jobs = 6
        self.n_tasks = 60

        self.task_node_proj = nn.Linear(task_in_fea_dim, out_fea_dim)
        self.mach_node_proj = nn.Linear(mach_in_fea_dim, out_fea_dim)

        self.pna = PNAConv(out_fea_dim, out_fea_dim, ['mean', 'max', 'sum', 'std'],
                           ['identity', 'amplification', 'attenuation'], 2.5, 0.4)
        self.gat = EfficientGAT(num_of_layers=n_layers, num_head_per_layers=n_head_per_layers,
                                num_features_per_layers=n_features_per_layers, add_skip_connection=True, bias=True,
                                dropout=0.6)

    def forward(self, state):
        n_nodes = state['nodes_num']
        adj = state['graph']

        temp_adj = adj.to_sparse().indices()
        temp_adj[1] += temp_adj[0] * n_nodes
        temp_adj[2] += temp_adj[0] * n_nodes
        adj = temp_adj[1:]
        graph = dgl.graph(adj)
        task_fea = self.task_node_proj(state['task_fea'])
        mach_fea = self.mach_node_proj(state['mach_fea'])
        edge_fea = state['edge_fea']

        node_fea = torch.cat((task_fea, mach_fea))
        aggr_node_fea = self.pna(graph, node_fea, edge_fea)

        node_embedding = self.gat((aggr_node_fea, adj))
        # 分离机器与任务节点，建立边的embedding



""" Critic部分: 主要在于评估给定state下的action的Q值"""


class CriticPPO(nn.Module):
    """
    Simple MLP Critic, PPO的Critic评估状态的价值，
    """

    def __init__(self, mid_dim, num_layer, state_dim, _action_dim):
        super().__init__()
        self.net = build_mlp(mid_dim, num_layer, state_dim, 1)

    def forward(self, state):
        return self.net(state)


class CriticGAT(ActorGAT):
    """
    Critic utilize simpler GAT model
    """

    def __init__(self, gat_args=None):
        super().__init__(gat_args)
        in_fea_dim = getattr(gat_args, "in_fea_dim", 3)
        # out_fea_dim = getattr(gat_args, "out_fea_dim", 64)
        # n_jobs = getattr(gat_args, "jobs_num", 6)
        # n_tasks = getattr(gat_args, "tasks_num", 60)
        # gat to get the embedding of all nodes (nxn, nxFIN) -> nxFOUT,  nxn->adj, nxFIN->node features, FIN = state_dim, FOUT = embedding_dim = 64
        self.gat = EfficientGAT(num_of_layers=2, num_head_per_layers=[4, 1],
                                num_features_per_layers=[in_fea_dim, 32, 32], add_skip_connection=True, bias=True,
                                dropout=0.6)
        # mlp to get the prob of candidate prob
        self.get_values = build_mlp(256, 5, 32, 1)

    def forward(self, states):
        batch_size = states.shape[0]
        n_nodes = states.shape[1]
        # adj fea 可以将一个batch的图 作为一个大图输入，因为单个图之间不连通，因此不会有影响，adj不能简单的reshape，(B, N, N) -> (BN, BN)
        adj = states[:, :, :n_nodes]  # adj (B, N, N) -> (BN, BN) block_diag
        # adj = torch.block_diag(*adj).to_sparse_coo().indices()  # 每单个图不连通，大图就是一个分块对角矩阵，     TODO 显存瓶颈
        temp_adj = adj.to_sparse().indices()
        temp_adj[1] += temp_adj[0] * n_nodes
        temp_adj[2] += temp_adj[0] * n_nodes
        adj = temp_adj[1:]
        fea = states[:, :, n_nodes:]  # reshape fea (B, N, FIN) -> (BN, FIN)
        fea = fea.reshape(-1, fea.shape[-1])

        nodes_embedding, edge_index = self.gat((fea, adj))
        nodes_embedding = nodes_embedding.reshape(batch_size, -1, nodes_embedding.shape[-1])
        values = self.get_values(nodes_embedding)
        return values


def build_mlp(mid_dim, num_layer, input_dim, output_dim):
    assert num_layer >= 1
    net_list = []
    if num_layer == 1:
        net_list.extend([nn.Linear(input_dim, output_dim), ])
    else:
        net_list.extend([nn.Linear(input_dim, mid_dim), nn.Tanh()])
        for _ in range(num_layer - 2):
            net_list.extend([nn.Linear(mid_dim, mid_dim), nn.Tanh()])
        net_list.extend([nn.Linear(mid_dim, output_dim), ])
    return nn.Sequential(*net_list)


if __name__ == "__main__":
    # pass
    # actor = ActorPPO(mid_dim=64, num_layers=5, state_dim=62, action_dim=1)
    # input_x = torch.randint(low=1, high=99, size=(8, 60, 62), dtype=torch.float32)
    # action, log_prob = actor.get_action(input_x)
    device = torch.device("cuda:0")
    actor_gat = ActorGATBatch(device)
    critic_gat = CriticGAT(3)
    env_args = {
        'env_num': 1,
        'env_name': 'SJssPEnv',
        'max_step': 10000,
        'fea_dim': 3,
        'action_dim': 1,
        'if_discrete': True,
        'target_return': 0
    }
    from Env.SJSSPEnv import SJsspEnv
    env = SJsspEnv(job_num=6, machine_num=10, args=env_args)
    states = [env.reset() for _ in range(16)]
    states = torch.as_tensor(np.array(states), dtype=torch.float32)
    action = actor_gat.get_action(states)

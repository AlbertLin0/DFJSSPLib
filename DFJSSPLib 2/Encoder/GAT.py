import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyRelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])

        e = Wh1 + Wh2.T
        return self.leakyRelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + '->' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_attr = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, graph_pool,
                padded_nei, adj):
        adj = adj.to_dense()
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_attr(x, adj))

        h_nodes = x.clone()
        pooling_h = torch.sparse.mm(graph_pool, x)
        return pooling_h, h_nodes


class SpecialSpmmFunction(torch.autograd.Function):
    """Function for sparse region backpropagation layer"""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]

        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)

        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT Layer
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, x, adj):
        dv = 'cuda' if x.is_cuda else 'cpu'

        N = x.size()[0]
        edge = adj.nonzero().t()

        h = torch.mm(x, self.W)

        assert not torch.isnan(h).any()

        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv))
        edge_e = self.dropout(edge_e)

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()

        h_prime = h_prime.div(e_rowsum)
        assert not torch.isnan(h_prime).any()

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + '->' + str(self.out_features) + ')'


class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True)
                           for _ in range(nheads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_attr = SpGraphAttentionLayer(nhid*nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, graph_pool, padded_nei, adj):
        adj = adj.to_dense()
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([attr(x, adj) for attr in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_attr(x, adj))

        h_nodes = x.clone()
        pooling_h = torch.sparse.mm(graph_pool, x)
        return pooling_h, h_nodes


class GATLayer(nn.Module):

    src_nodes_dim = 0
    trg_nodes_dim = 1

    nodes_dim = 0
    head_dim = 1

    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(), dropout=0.6,
                 add_skip_connection=True, bias=True, log_attention_weights=False, alpha=0.2):
        super(GATLayer, self).__init__()

        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features
        self.concat = concat
        self.add_skip_connection = add_skip_connection

        #
        # Trainable Parameter
        # linear projection matrix W
        self.W = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)

        # attention a
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        # End Trainable
        self.leakyRelu = nn.LeakyReLU(alpha)
        self.softmax = nn.Softmax(dim=-1)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.log_attention_weights = log_attention_weights
        self.attention_weights = None

        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, data):

        in_features, edge_index = data
        #
        # Linear project + regularization
        num_of_nodes = in_features.shape[self.nodes_dim]
        assert edge_index.shape[0] == 2, f'Edge Index should with shape=(2,E)'

        # in features shape = (N, FIN) N - number of nodes, FIN - input features per node
        in_features = self.dropout(in_features)

        # (N, FIN) * (FIN, NH * FOUT) -> (N, NH, FOUT)
        nodes_features_proj = self.W(in_features).view(-1, self.num_of_heads, self.num_out_features)
        nodes_features_proj = self.dropout(nodes_features_proj)

        #
        # edge attention
        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)

        scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = self.lift(scores_source, scores_target, nodes_features_proj, edge_index)
        scores_per_edge = self.leakyRelu(scores_source_lifted + scores_target_lifted)

        attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, edge_index[self.trg_nodes_dim], num_of_nodes)
        attentions_per_edge = self.dropout(attentions_per_edge)

        #
        # neighborhood aggregation
        #
        nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attentions_per_edge

        out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted_weighted, edge_index, in_features, num_of_nodes)

        out_nodes_features = self.skip_concat_bias(attentions_per_edge, in_features, out_nodes_features)
        return out_nodes_features, edge_index

    def skip_concat_bias(self, attention_coefficients, in_node_features, out_node_features):
        if self.log_attention_weights:
            self.attention_weights = attention_coefficients

        if not out_node_features.is_contiguous():
            out_node_features = out_node_features.contiguous()

        if self.add_skip_connection:
            if out_node_features.shape[-1] == in_node_features.shape[-1]:  # if FIN == FOUT
                # unsqueeze: (N, FIM) -> (N, 1, FIN),   out_features: (N, NH, FOUT), unsqueeze in_feature and add
                out_node_features += in_node_features.unsqueeze(1)
            else:  # FIN != FOUT,  need to project input features
                out_node_features += self.skip_proj(in_node_features).view(-1, self.num_of_heads, self.num_out_features)

        if self.concat:
            # not the last layer (N, NH, FOUT) -> (N, NH*FOUT)
            out_node_features = out_node_features.view(-1, self.num_of_heads * self.num_out_features)
        else:
            # last layer (N, NH, FOUT) -> (N, FOUT)
            out_node_features = out_node_features.mean(dim=self.head_dim)

        if self.bias is not None:
            out_node_features += self.bias

        return out_node_features if self.activation is None else self.activation(out_node_features)

    # Auxiliary Functions
    def neighborhood_aware_softmax(self, score_per_edge, trg_index, num_of_nodes):
        score_per_edge = score_per_edge - score_per_edge.max()  # improve numerical stability
        exp_scores_per_edge = score_per_edge.exp()

        neighborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index, num_of_nodes)
        attention_per_edge = exp_scores_per_edge / (neighborhood_aware_denominator + 1e-6)

        return attention_per_edge.unsqueeze(-1)

    def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, num_of_nodes):
        trg_index_broadcasted = self.explicit_broadcast(trg_index, exp_scores_per_edge)
        size = list(exp_scores_per_edge.shape)
        size[self.nodes_dim] = num_of_nodes
        neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)
        neighborhood_sums.scatter_add_(self.nodes_dim, trg_index_broadcasted, exp_scores_per_edge)
        return neighborhood_sums.index_select(self.nodes_dim, trg_index)

    def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes):
        size = list(nodes_features_proj_lifted_weighted.shape)
        size[self.nodes_dim] = num_of_nodes
        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)

        trg_index_broadcasted = self.explicit_broadcast(edge_index[self.trg_nodes_dim], nodes_features_proj_lifted_weighted)

        out_nodes_features.scatter_add_(self.nodes_dim, trg_index_broadcasted, nodes_features_proj_lifted_weighted)
        return out_nodes_features

    def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]

        scores_source = scores_source.index_select(self.nodes_dim, src_nodes_index)
        scores_target = scores_target.index_select(self.nodes_dim, trg_nodes_index)
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(self.nodes_dim, src_nodes_index)

        return scores_source, scores_target, nodes_features_matrix_proj_lifted

    def explicit_broadcast(self, this, other):
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)

        return this.expand_as(other)


class EfficientGAT(nn.Module):
    def __init__(self, num_of_layers, num_head_per_layers, num_features_per_layers, add_skip_connection=True,
                 bias=True, dropout=0.6, log_attention_weights=False):
        """

        :param num_of_layers: Attention Layer的层数
        :param num_head_per_layers: List 每层Attention Layer的head
        :param num_features_per_layers: List 每层Attention Layer的中间特征维度
        :param add_skip_connection:
        :param bias:
        :param dropout:
        :param log_attention_weights:
        """
        super(EfficientGAT, self).__init__()

        assert num_of_layers == len(num_head_per_layers) == len(num_features_per_layers) - 1

        num_head_per_layers = [1] + num_head_per_layers

        attentions = []
        for i in range(num_of_layers):
            layer = GATLayer(
                num_in_features=num_features_per_layers[i] * num_head_per_layers[i],
                num_out_features=num_features_per_layers[i + 1],
                num_of_heads=num_head_per_layers[i + 1],
                concat=True if i < num_of_layers - 1 else False,
                activation=nn.ELU() if i < num_of_layers - 1 else None,
                dropout=dropout,
                add_skip_connection=add_skip_connection,
                bias=bias,
                log_attention_weights=log_attention_weights
            )
            attentions.append(layer)

        self.attentions = nn.Sequential(*attentions)

    # def forward(self, x, graph_pool, padded_nei, adj):
    def forward(self, data):
        # x shape [N, FIN], N - 节点数量，也即task数量 n_j * n_m
        # adj shape [N, N], adj._indices() 稀疏形式 [2, E], E图中边的数量
        # data = (x, adj._indices())

        out_features, edge_index = self.attentions(data)    # [N, FIN] + [2, E] -> [N, FOUT]
        # h_nodes = out_features.clone()
        # pooled_h = torch.sparse.mm(graph_pool, out_features)
        return out_features, edge_index


if __name__ == "__main__":
    encoder = EfficientGAT(num_of_layers=2, num_head_per_layers=[8, 1], num_features_per_layers=[2, 64, 4],
                           add_skip_connection=True, bias=True, dropout=0.6)

    x = torch.rand([36, 2])     # [N, FIN]
    adj = torch.rand([36, 36]).to_sparse().indices()  # [N, N]
    data = (x, adj)
    out, edge_index = encoder(data)
    print(out.shape)


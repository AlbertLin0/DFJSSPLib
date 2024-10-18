import os.path

import numpy as np
import numpy.random as rd
import torch


class ReplayBuffer:  # for off policy, 非ppo、a2c算法
    def __init__(self, max_capacity, state_dim, action_dim, gpu_id, if_use_per=False):
        self.prev_p = 0     # pointer to prev traj
        self.next_p = 0     # pointer to next traj
        self.if_full = False
        self.cur_capacity = 0
        self.max_capacity = max_capacity
        self.add_capacity = 0   # update_buffer时增加的量

        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available()) and (gpu_id >= 0) else "cpu")

        self.buf_action = torch.empty((max_capacity, action_dim), dtype=torch.float32, device=self.device)
        self.buf_reward = torch.empty((max_capacity, 1), dtype=torch.float32, device=self.device)
        self.buf_mask = torch.empty((max_capacity, 1), dtype=torch.float32, device=self.device)

        buf_state_size = (max_capacity, state_dim) if isinstance(state_dim, int) else (max_capacity, *state_dim)  # in jssp state dim is (N, N + FIN), N 节点数， FIN 每个节点的特征向量， [N, N]是图的邻接矩阵
        self.buf_state = torch.empty(buf_state_size, dtype=torch.float32, device=self.device)

        self.if_use_per = if_use_per
        if if_use_per:
            self.per_tree = BinarySearchTree(max_capacity)
            self.sample_batch = self.sample_batch_per

    def update_buffer(self, traj_list):
        """

        :param traj_list: 探索env产生的路径列表，[(states, ...), (rewards, ...), (masks, ...), (actions, ...)]
        :return:
        """
        traj_items = list(map(list, zip(*traj_list)))   # might be bugs, 也许是多余的操作，explore_env中返回了convert_trajectory，将探索路径中的项拆分出来了

        states, rewards, masks, actions = [torch.cat(item, dim=0) for item in traj_items]
        self.add_capacity = rewards.shape[0]
        p = self.next_p + self.add_capacity

        if self.if_use_per:
            self.per_tree.update_ids(data_ids=np.arange(self.next_p, p) % self.max_capacity)

        if p > self.max_capacity:
            self.buf_state[self.next_p:self.max_capacity] = states[:self.max_capacity - self.next_p]
            self.buf_reward[self.next_p:self.max_capacity] = rewards[:self.max_capacity - self.next_p]
            self.buf_mask[self.next_p:self.max_capacity] = masks[:self.max_capacity - self.next_p]
            self.buf_action[self.next_p:self.max_capacity] = actions[:self.max_capacity - self.next_p]
            self.if_full = True

            p = p - self.max_capacity
            self.buf_state[0:p] = states[-p:]
            self.buf_reward[0:p] = rewards[-p:]
            self.buf_mask[0:p] = masks[-p:]
            self.buf_action[0:p] = actions[-p:]
        else:
            self.buf_state[self.next_p:p] = states
            self.buf_reward[self.next_p:p] = rewards
            self.buf_mask[self.next_p:p] = masks
            self.buf_action[self.next_p:p] = actions

        self.next_p = p
        self.cur_capacity = self.max_capacity if self.if_full else self.next_p

        steps = rewards.shapes[0]
        r_exp = rewards.mean().item()
        return steps, r_exp

    def sample_batch(self, batch_size):
        indices = torch.randint(self.cur_capacity - 1, size=(batch_size, ), device=self.device)

        i1 = self.next_p
        i0 = self.next_p - self.add_capacity
        num_new_indices = 1
        new_indices = torch.randint(i0, i1, size=(num_new_indices, )) % (self.max_capacity - 1)

        return (
            self.buf_reward[indices],
            self.buf_mask[indices],
            self.buf_action[indices],
            self.buf_state[indices],
            self.buf_state[indices + 1]     # next state
        )

    def sample_batch_per(self, batch_size):
        beg = -self.max_capacity
        end = (self.cur_capacity - self.max_capacity) if (self.cur_capacity < self.max_capacity) else None

        indices, is_weights = self.per_tree.get_indices_is_weights(batch_size, beg, end)

        return (
            self.buf_reward[indices],
            self.buf_mask[indices],
            self.buf_action[indices],
            self.buf_state[indices],
            self.buf_state[indices + 1],     # next state
            torch.as_tensor(is_weights, dtype=torch.float32, device=self.device)
        )

    def td_error_update(self, td_error):
        self.per_tree.td_error_update(td_error)

    def save_or_load_history(self, cwd, if_save):
        obj_names = (
            (self.buf_reward, "reward"),
            (self.buf_mask, "mask"),
            (self.buf_action, "action"),
            (self.buf_state, "state"),
        )

        if if_save:
            print(f"| {self.__class__.__name__} : Saving in {cwd}")
            for obj, name in obj_names:
                if self.cur_capacity == self.next_p:
                    buf_tensor = obj[:self.cur_capacity]
                else:
                    buf_tensor = torch.vstack((obj[self.next_p:self.cur_capacity], obj[0:self.next_p]))

                torch.save(buf_tensor, f"{cwd}/replay_buffer_{name}.pt")

        elif os.path.isfile(f"{cwd}/replay_buffer_state.pt"):
            print(f"| {self.__class__.__name__} : Loading in {cwd}")
            for obj, name in obj_names:
                buf_tensor = torch.load(f"{cwd}/replay_buffer_{name}.pt")
                buf_capacity = buf_tensor.shape[0]

                obj[:buf_capacity] = buf_tensor
            self.cur_capacity = buf_capacity
            print(f"| {self.__class__.__name__} : Loaded in {cwd}")

    def get_state_norm(self, cwd, state_avg=0.0, state_std=1.0):
        try:
            torch.save(state_avg, f"{cwd}/env_state_avg.pt")
            torch.save(state_std, f"{cwd}/env_state_std.pt")
        except Exception as error:
            print(error)

        state_avg, state_std = get_state_avg_std(buf_state=self.buf_state, batch_size=2**10, state_avg=state_avg, state_std=state_std)

        torch.save(state_avg, f"{cwd}/env_state_avg.pt")
        print(f"| {self.__class__.__name__}: state_avg = {state_avg}")
        torch.save(state_std, f"{cwd}/env_state_std.pt")
        print(f"| {self.__class__.__name__}: state_std = {state_std}")

    def concatenate_state(self):
        pass

    def concatenate_buffer(self):
        pass


class ReplayBufferList(list):
    def __init__(self):
        list.__init__(self)

    def update_buffer(self, traj_list):
        cur_items = list(map(list, zip(*traj_list)))    # might be bugs, traj_list传入的时候就已经是items的形式了
        cur_items = traj_list
        self[:] = [torch.cat(item, dim=0) for item in cur_items]

        steps = self[1].shape[0]
        r_exp = self[1].mean().items()

        return steps, r_exp

    def get_state_norm(self, cwd='.'):
        batch_size = 2**10
        buf_state = self[0]

        state_len = buf_state.shape[0]
        state_avg = torch.zeros_like(buf_state[0])
        state_std = torch.zeros_like(buf_state[0])

        i = 0
        for i in range(0, state_len, batch_size):
            state_avg += buf_state[i:i+batch_size].mean(axis=0)
            state_std += buf_state[i:i+batch_size].std(axis=0)

        i += 1

        state_avg /= i
        torch.save(state_avg, f"{cwd}/state_norm_avg.pt")
        print(f"| {self.__class__.__name__}: state_avg {state_avg}")

        state_std /= i
        torch.save(state_std, f"{cwd}/state_norm_std.pt")
        print(f"| {self.__class__.__name__}: state_std {state_std}")


def get_state_avg_std(buf_state, batch_size, state_avg, state_std):
    """
    使用缓冲池中的状态，更新状态的平均值和标准差
    :param buf_state:
    :param batch_size:
    :param state_avg:
    :param state_std:
    :return:
    """
    state_len = buf_state.shape[0]
    state_avg_temp = torch.zeros_like(buf_state.shape[0])
    state_std_temp = torch.zeros_like(buf_state.shape[0])

    from tqdm import trange
    for i in trange(0, state_len, batch_size):
        state_part = buf_state[i:i + batch_size]
        state_avg_temp += state_part.mean(axis=0)
        state_std_temp += state_part.std(axis=0)

    num = max(1, state_len / batch_size)
    state_std_temp /= num
    state_avg_temp /= num

    new_state_avg = state_avg_temp.data.cpu() * state_std + state_avg
    new_state_std = state_std_temp.data.cpu() * state_std
    return new_state_avg, new_state_std


class BinarySearchTree:
    """二叉树通过数组存储"""

    def __init__(self, memo_len):
        self.memo_len = memo_len
        self.prob_ary = np.zeros((memo_len - 1) + memo_len)      # 父节点数量 +  叶节点数量
        self.max_capacity = len(self.prob_ary)
        self.cur_capacity = self.memo_len - 1
        self.indices = None
        self.depth = int(np.log2(self.max_capacity))

        # PER
        # alpha, beta = 0.7, 0.5 for rank-based variant
        # alpha, beta = 0.6, 0.4 for proportional variant
        self.per_alpha = 0.6
        self.per_beta = 0.4

    def update_id(self, data_id, prob=10):      # 10 is max prob
        tree_id = data_id + self.memo_len - 1
        if self.cur_capacity == tree_id:
            self.cur_capacity += 1

        delta = prob - self.prob_ary[tree_id]
        self.prob_ary[tree_id] = prob

        while tree_id != 0:     # 沿着路径反向传播更改
            tree_id = (tree_id - 1) // 2
            self.prob_ary[tree_id] += delta

    def update_ids(self, data_ids, prob=10):
        ids = data_ids + self.memo_len - 1
        self.cur_capacity += (ids >= self.cur_capacity).sum()

        upper_step = self.depth - 1
        self.prob_ary[ids] = prob       # ids 给定子节点的下标
        p_ids = (ids - 1) // 2

        while upper_step:
            ids = p_ids * 2 + 1         # 左子节点的下标
            self.prob_ary[p_ids] = self.prob_ary[ids] + self.prob_ary[ids + 1]
            p_ids = (p_ids - 1) // 2
            upper_step -= 1

        self.prob_ary[0] = self.prob_ary[1] + self.prob_ary[2]      # 根节点单独更新，因为深度减一了，while中没有更新到根节点

    def get_lead_id(self, v):
        parent_idx = 0
        while True:
            l_idx = 2 * parent_idx + 1      # 左子节点
            r_idx = l_idx + 1               # 右子节点

            if r_idx >= (len(self.prob_ary)):   # 到达叶节点层
                leaf_idx = parent_idx
                break
            else:
                if v <= self.prob_ary[l_idx]:
                    parent_idx = l_idx
                else:
                    v -= self.prob_ary[l_idx]
                    parent_idx = r_idx

        return min(leaf_idx, self.cur_capacity - 2)

    def get_indices_is_weights(self, batch_size, start, end):
        self.per_beta = min(1., self.per_beta + 0.001)

        values = (rd.rand(batch_size) + np.arange(batch_size)) * (self.prob_ary[0] / batch_size)

        leaf_ids = np.array([self.get_lead_id(v) for v in values])
        self.indices = leaf_ids - (self.memo_len - 1)

        prob_ary = self.prob_ary[leaf_ids] / self.prob_ary[start:end].min()
        is_weights = np.power(prob_ary, -self.per_beta)     # importance sampling weights
        return self.indices, is_weights

    def td_error_update(self, td_error):
        prob = td_error.squeeze().clamp(1e-6, 10).pow(self.per_alpha)
        prob = prob.cpu().numpy()
        self.update_ids(self.indices, prob)




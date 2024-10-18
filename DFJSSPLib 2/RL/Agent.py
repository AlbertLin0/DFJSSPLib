import os.path
import types

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR

from Train.config import Arguments
from Train.replay_buffer import ReplayBuffer


class Agent:
    """
    具体智能体会依据不同的强化学习算法进行重写
    Base agent for all DRL agent
    implement based env exploration, optimize actor and critic model
    """

    def __init__(self, net_dim, state_dim, action_dim, gpu_id=0, args=None):
        """
        需要被不同的DRL算法重写
        :param net_dim: 网络的维度
        :param state_dim: 状态的特征维度，比如说JSSP中，状态被表示为析取图的状态，[图节点的加工时间、节点是否完成、边的连接情况]，fea+adj共同表示状态
        :param action_dim: 动作的维度，JSSP中为下一步选择的节点，即为1
        :param gpu_id: 使用的显卡
        :param args: 训练参数，包括actor、critic使用的模型，以及一系列超参数
        """
        self.gamma = getattr(args, 'gamma', 0.99)
        self.env_num = getattr(args, 'env_num', 1)
        self.num_layer = getattr(args, 'num_layer', 3)
        self.batch_size = getattr(args, 'batch_size', 128)
        self.action_dim = getattr(args, 'action_dim', 3)
        self.repeat_times = getattr(args, 'repeat_times', 1.)
        self.reward_scale = getattr(args, 'reward_scale', 1.)
        self.lambda_critic = getattr(args, 'lambda_critic', 1.)
        self.learning_rate = getattr(args, 'learning_rate', 2 ** -15)
        self.clip_grad_norm = getattr(args, 'clip_grad_norm', 3.0)
        self.soft_update_tau = getattr(args, 'soft_update_tau', 2 ** -8)
        self.actor_args = getattr(args, 'gat_args', None)
        # self.fea_dim = getattr(args, 'fea_dim', 2)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.if_use_per = getattr(args, 'if_use_per', None)
        self.if_off_policy = getattr(args, 'if_off_policy', None)
        self.if_use_old_traj = getattr(args, 'if_use_old_traj', False)

        self.lr_update_step = getattr(args, 'lr_update_step', 20000)

        self.states = None  # self.states.shap == (env_num, state.dim)
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
        self.traj_list = [[torch.tensor((), dtype=torch.float32, device=self.device)
                           for _ in range(4 if self.if_off_policy else 5)]
                          for _ in range(self.env_num)]

        """模型网络设置"""
        # Actor 和 Critic模型
        act_class = getattr(self, 'act_class', None)  # actor类
        cri_class = getattr(self, 'cri_class', None)  # critic类
        # modified
        self.actor = act_class(self.device, self.actor_args).to(self.device)

        # # test
        # load_torch_file(self.actor, "./TrainingLog/SJsspEnv_PPO_0_J20_M15_2023_02_14/actor.pth")

        self.critic = cri_class(self.actor_args).to(self.device) \
            if cri_class else self.actor

        # 优化器设置，增加scheduler
        self.act_optimizer = torch.optim.Adam(self.actor.parameters(), self.learning_rate, weight_decay=0.005)
        self.act_optim_scheduler = StepLR(self.act_optimizer, step_size=self.lr_update_step, gamma=0.5)
        self.cri_optimizer = torch.optim.Adam(self.critic.parameters(), self.learning_rate, weight_decay=0.005) \
            if cri_class else self.act_optimizer
        self.cri_optim_scheduler = StepLR(self.cri_optimizer, step_size=self.lr_update_step, gamma=0.5)
        self.act_optimizer.parameters = types.MethodType(get_optim_param, self.act_optimizer)
        self.cri_optimizer.parameters = types.MethodType(get_optim_param, self.cri_optimizer)

        """目标网络"""
        from copy import deepcopy
        self.if_act_target = args.if_act_target if hasattr(args, 'if_act_target') else getattr(self, 'if_act_target', None)
        self.if_cri_target = args.if_cri_target if hasattr(args, 'if_cri_target') else getattr(self, 'if_cri_target', None)

        self.act_target = deepcopy(self.actor) if self.if_act_target else self.actor
        self.cri_target = deepcopy(self.critic) if self.if_cri_target else self.critic

        if self.env_num == 1:
            self.explore_env = self.explore_one_env
        else:
            self.explore_env = self.explore_vec_env

        if self.if_use_per:
            self.criterion = torch.nn.MSELoss(reduction="mean")
            self.get_obj_critic = self.get_obj_critic_per
        else:
            self.criterion = torch.nn.MSELoss(reduction="mean")
            self.get_obj_critic = self.get_obj_critic_raw

        """H-Term"""
        self.h_term_gamma = getattr(args, 'h_term_gamma', 0.8)
        self.h_term_k_step = getattr(args, 'h_term_k_step', 4)
        self.h_term_lambda = getattr(args, 'h_term_lambda', 2 ** -3)
        self.h_term_update_gap = getattr(args, 'h_term_update_gap', 1)
        self.h_term_drop_rate = getattr(args, 'h_term_drop_rate', 2 ** -3)
        self.h_term_sample_rate = getattr(args, 'h_term_sample_rate', 2 ** -4)
        self.h_term_buffer = []
        self.ten_state = None
        self.ten_action = None
        self.ten_r_norm = None
        self.ten_reward = None
        self.ten_mask = None
        self.ten_v_sum = None

    def explore_one_env(self, env, target_step):
        """
        探索一个环境x步，返回探索路径，具体的实现仍然是以向量环境为基础实现的
        :param env: 模拟强化学习环境，类型是向量化环境，向量化环境能够串行或者并行一组环境
        :param target_step: 探索的步数
        :return: 每一步的探索路径包括 [(state, reward, mask, action, ), ...]
        """
        traj_list = []
        last_dones = [0, ]  # 记录环境交互完成时的最后步数

        i = 0
        done = False
        while i < target_step:
            state = env.reset()
            while not done:
                tensor_state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
                tensor_action = self.actor.get_action(tensor_state.to(self.device)).detach().cpu()  # ? 应该与actor以及环境的实现有关
                next_state, reward, done, _ = env.step(tensor_action[0].numpy())

                traj_list.append((tensor_state, reward, done, tensor_action))

                i += 1
                state = next_state

        self.states[0] = state
        last_dones[0] = i

        return self.convert_trajectory(traj_list, last_dones)

    def explore_vec_env(self, env, target_step):
        """
        探索一组环境x步，返回探索路径
        :param env: 模拟强化学习环境，向量化环境
        :param target_step: 探索的步数
        :return: 每一步的探索路径 [(state, reward, mask, action, ), ...]
        """
        traj_list = []
        last_dones = torch.zeros(self.env_num, dtype=torch.int, device=self.device)     # 记录每个环境交互完成时的步数
        states = self.states if self.if_use_old_traj else env.reset()

        i = 0
        dones = torch.zeros(self.env_num, dtype=torch.int, device=self.device)
        while i < target_step or not any(dones):    # 探索指定步数 或者 至少一个环境交互结束，如果全部环境都未结束，到达了指定步数后仍会继续探索至一个环境结束
            actions = self.actor.get_action(states).detach()
            next_states, rewards, dones, _ = env.step(actions)

            traj_list.append((states.clones(), rewards.clone(), dones.clone(), actions))

            i += 1
            last_dones[torch.where(dones)[0]] = i   # 记录已结束环境的最新步数
            states = next_states

        self.states = states
        return self.convert_trajectory(traj_list, last_dones)

    def convert_trajectory(self, traj_list, last_dones):
        """

        :param traj_list: 路径列表，包含了每一步的(state, reward, mask, action)，列表长度小于等于target_step，如果有环境结束了就小于
        :param last_dones: 记录结束环境的步数
        :return:
        """
        traj_list1 = list(map(list, zip(*traj_list)))  # 取出每个路径中的[[states], [rewards], [dones], [actions]]

        del traj_list

        traj_states = torch.stack(traj_list1[0])
        traj_actions = torch.stack(traj_list1[3])

        if len(traj_actions.shape) == 2:
            traj_actions = traj_actions.unsqueeze(2)

        if self.env_num > 1:
            traj_rewards = (torch.stack(traj_list1[1]) * self.reward_scale).unsqueeze(2)
            traj_masks = ((1 - torch.stack(traj_list1[2])) * self.gamma).unsqueeze(2)
        else:
            traj_rewards = (torch.tensor(traj_list1[1], dtype=torch.float32) * self.reward_scale).reshape(-1, 1, 1)
            traj_masks = ((1 - torch.tensor(traj_list1[2], dtype=torch.float32)) * self.gamma).reshape(-1, 1, 1)

        if len(traj_list1) <= 4:
            # 没有noise
            traj_list2 = [traj_states, traj_rewards, traj_masks, traj_actions]
        else:
            traj_noise = torch.stack(traj_list1[4])
            traj_list2 = [traj_states, traj_rewards, traj_masks, traj_actions, traj_noise]

        del traj_list1

        traj_list3 = []
        for j in range(len(traj_list2)):
            # 遍历状态、奖励、mask、动作、噪声
            cur_item = []  # 取每个环境下结束步之前的值，如果使用旧路径，则将前一步的路径插入到当前item中
            buf_item = traj_list2[j]

            for env_i in range(self.env_num):
                # 遍历每个环境
                last_step = last_dones[env_i]

                pre_item = self.traj_list[env_i][j]
                if len(pre_item):
                    cur_item.append(pre_item)

                cur_item.append(buf_item[:last_step, env_i])

                if self.if_use_old_traj:
                    self.traj_list[env_i][j] = buf_item[last_step:, env_i]

            traj_list3.append(torch.vstack(cur_item))
        return traj_list3

    def update_net(self, buffer):
        """
        从replay_buffer中采样更新网络模型参数，具体采样以及更新方式由具体模型实现
        :param buffer:
        :return:
        """
        pass

    @staticmethod
    def soft_update(target_net, current_net, tau):
        """
        软更新策略，使用当前网络参数更新目标网络参数， target = tau * current  + (1-tau)* target，每次更新只取当前网络很小的权重
        :param target_net: 目标网络，目标网络参数更新使用当前网络的参数
        :param current_net: 当前网络，当前网络参数更新通过优化器
        :param tau: 软更新策略系数，超参数，目标网络更新时，当前网络参数的占比重，一般很小
        :return:
        """
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1 - tau))

    def optimizer_update(self, optimizer, scheduler, objective):
        """
        更新网络参数，最小化目标函数
        :param optimizer: 优化器
        :param scheduler:
        :param objective: 优化目标函数，比如loss函数
        :return:
        """
        optimizer.zero_grad()
        objective.backward()
        clip_grad_norm_(parameters=optimizer.param_groups[0]["params"], max_norm=self.clip_grad_norm)
        optimizer.step()
        scheduler.step()

    def optimizer_update_amp(self, optimizer, scheduler, objective):
        """
        自动混合精度更新网络参数
        :param optimizer:
        :param scheduler:
        :param objective:
        :return:
        """
        amp_scale = torch.cuda.amp.GradScaler()
        optimizer.zero_grad()
        amp_scale.scale(objective).backward()
        amp_scale.unscale_(optimizer)
        clip_grad_norm_(parameters=optimizer.param_groups[0]["params"], max_norm=self.clip_grad_norm)
        amp_scale.step(optimizer)
        amp_scale.update()
        scheduler.step()

    def get_obj_critic_raw(self, buffer, batch_size):
        """
        从replay buffer中采样（均匀采样）实例，计算模型网络的loss
        1. 从经验池中采样(奖励、mask、动作、状态、下一个状态)
        2. 使用actor_target计算下一个动作，critic_target评价actor_target算出的下一个动作，从critic_target的评价中计算出q值标签
        3. 使用critic网络预测原先的state、action下的q值，作为预测值，critic_target的标签q值和critic网络的预测q值计算loss
        :param buffer: replay buffer，其中包含了探索路径即 (状态、奖励、动作、...)
        :param batch_size: 采样的数据量，使用batch_size个探索路径进行梯度下降
        :return: 网络loss以及状态
        """
        with torch.no_grad():
            reward, mask, action, state, next_state = buffer.sample_batch(batch_size)
            next_action = self.act_target(next_state)
            critic_targets = self.cri_target(next_state, next_action)
            (next_q, min_indices) = torch.min(critic_targets, dim=1, keepdim=True)
            q_label = reward + mask * next_q

        q = self.critic(state, action)
        obj_critic = self.criterion(q, q_label)
        return obj_critic, state

    def get_obj_critic_per(self, buffer, batch_size):
        """
        从replay_buffer中使用PER采样实例，计算模型网络的loss
        :param buffer: 经验池
        :param batch_size:
        :return: 网络loss以及状态
        """
        with torch.no_grad():
            reward, mask, action, state, next_state, is_weights = buffer.sample_batch(batch_size)
            next_action = self.act_target(next_state)
            critic_target = self.cri_target(next_state, next_action)
            (next_q, min_indices) = torch.min(critic_target, dim=1, keepdim=True)
            q_label = reward + mask * next_q

        q = self.critic(state, action)
        td_error = self.criterion(q, q_label)
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, state

    def get_obj_critic_her(self, buffer, batch_size):
        """
        TODO 尝试使用HER采样实例
        :param buffer:
        :param batch_size:
        :return:
        """
        pass

    def get_buf_h_term_k(self):
        # TODO 稳定训练 使用H-term
        pass

    def get_obj_h_term_k(self):
        pass

    def save_or_load_agent(self, cwd, if_save):
        """
        保存或者加载agent模型
        :param cwd: current working directory，当前工作目录，训练文件保存目录，包括了日志、模型
        :param if_save: True: save,  False: load
        :return:
        """

        name_obj_list = [
            ("actor", self.actor),
            ("act_target", self.act_target),
            ("act_optimizer", self.act_optimizer),
            ("critic", self.critic),
            ("cri_target", self.cri_target),
            ("cri_optimizer", self.cri_optimizer),
        ]
        name_obj_list = [(name, obj) for name, obj in name_obj_list if obj is not None]

        if if_save:
            for name, obj in name_obj_list:
                save_path = f"{cwd}/{name}.pth"
                torch.save(obj.state_dict(), save_path)
        else:
            for name, obj in name_obj_list:
                save_path = f"{cwd}/{name}.pth"
                load_torch_file(obj, save_path) if os.path.isfile(save_path) else None


def load_torch_file(model, path):
    state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)

def get_optim_param(optimizer):
    params_list = []
    for params_dict in optimizer.state_dict["state"].values:
        params_list.extend([t for t in params_dict.values if isinstance(t, torch.Tensor)])
    return params_list




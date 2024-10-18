import os
import numpy as np
import torch
from copy import deepcopy


class Arguments:
    """
    All Parameters for RL agents, 有些是超参数，有些是由强化学习环境指定（与环境保持一致）
    """
    def __init__(self, agent_class=None, env=None, env_func=None, env_args=None):
        self.env = env
        self.env_func = env_func    # 创建一个环境，env=env_func(env_args)
        self.env_args = env_args

        self.env_num = self.update_attr('env_num')          # 环境实例的个数，比如一个taillard实例就是一个环境
        self.max_step = self.update_attr('max_step')        # 一个回合最大的探索步数
        self.env_name = self.update_attr('env_name')
        self.state_dim = self.update_attr('state_dim')      # 状态向量的维度，JSSP中状态包括了当前图结构、task节点的运行情况、每个task节点的运行时间 TODO 状态的设计能否优化
        self.action_dim = self.update_attr('action_dim')    # 动作向量的维度，JSSP中动作即是所选择的task节点 TODO 动作的设计能否优化
        self.if_discrete = self.update_attr('if_discrete')  # 动作空间是否离散
        self.target_return = self.update_attr('target_return')  # 平均每回合期望的返回值，满足期望则学习成功
        self.gat_args = self.update_attr('gat_args')        # actor gat模型的基本参数
        self.job_num = self.update_attr('num_of_jobs')
        self.machine_num = self.update_attr('num_of_machines')
        self.high_bound = self.update_attr("high_bound")

        '''模型超参数'''
        self.agent_class = agent_class      # 指定强化学习算法
        self.net_dim = 2 ** 4               # 网络宽度，也许用不上
        self.num_layer = 3                  # MLP 中全连接层数
        self.explore_times = 6
        self.horizon_len = self.explore_times * env.num_of_tasks               # 每回合探索的步数 target_step，探索环境4遍
        if self.if_off_policy():
            self.max_memo = 2 ** 21         # 经验池的容量大小
            self.batch_size = self.net_dim  # 从经验回放缓冲中采样交互路径的数量，相当于一次训练使用的样本数量
            self.repeat_times = 2 ** 0      # epoch nums, 1个epoch指经验池采样结束一次
            self.if_use_per = False         # 稀疏奖励时可以尝试使用
            self.num_seed_steps = 2         # for warmup 初始化经验池，warm up阶段探索到的样本量为 num_seed_steps * env_nums * num_step_per_episode
            self.num_step_per_episode = 128
            self.n_step = 1
        else:  # on policy模型，PPO / A2C
            self.max_memo = 2 ** 16
            self.target_step = self.max_memo
            self.batch_size = 256           # tried 512, 64
            self.repeat_times = 3           # batch size 设置偏大时，repeat times的最好也适当调大
            self.if_use_gae = False          # 稀疏奖励中，使用PER ： Generalized Advantage Estimation
            # self.if_use_per = True

        '''训练超参数'''
        self.gamma = 0.99                   # 奖励的折扣因子
        self.reward_scale = 2 ** 0          # 裁剪reward
        self.lambda_critic = 2 ** 0         # critic目标函数的系数
        self.learning_rate = 5e-5       # 学习率   default 2**-10
        self.soft_update_tau = 2 ** -8
        self.clip_grad_norm = 3.0           # 梯度裁剪，0.1 ~ 4.0
        self.if_off_policy = self.if_off_policy()
        self.if_use_old_traj = False

        self.lr_update_step = 2000 * int(self.horizon_len * self.repeat_times / self.batch_size)
        # h-term 中才用得上
        self.if_act_target = True
        self.if_cri_target = True

        self.worker_num = 2
        self.thread_num = 8
        self.random_seed = 0
        self.learner_gpus = 0

        '''验证参数'''
        self.cwd = None                 # 当前工作文件夹，保存模型、训练测试结果
        self.if_remove = True           # 是否删除cwd文件夹
        self.break_epoch = 10000
        self.break_step = self.break_epoch * 4 * self.env.num_of_tasks      # total_step > break_step时结束训练
        self.if_over_write = False      # 覆盖最佳策略模型，否-保存全部策略模型，并单独记录最佳策略
        self.if_allow_break = True       # 是否允许提前终止，如果达到了预期则提前终止

        self.save_gap = 2               # 保存策略网络（actor）的间隔 step?
        self.eval_gap = 2 ** 4          # 评估agent的间隔 seconds?
        self.eval_times = 2 ** 3
        self.eval_env_func = None
        self.eval_env_args = None

    def init_before_train(self):
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.set_num_threads(self.thread_num)
        torch.set_default_dtype(torch.float32)
        import time
        day = time.strftime("%Y_%m_%d", time.localtime())

        if self.cwd is None:
            self.cwd = f"./TrainingLog/{self.env_name}_{self.agent_class.__name__[5:]}_{self.learner_gpus}_" \
                       f"J{self.job_num}_M{self.machine_num}_H{self.high_bound}_{day}"

        if self.if_remove is None:
            self.if_remove = bool(input(f"| Arguments PRESS 'y' to REMOVE: {self.cwd}? ") == 'y')
        elif self.if_remove:
            import shutil
            shutil.rmtree(self.cwd, ignore_errors=True)
            print(f"| Arguments Remove cwd: {self.cwd}")
        else:
            print(f"| Arguments Keep cwd: {self.cwd}")
        os.makedirs(self.cwd, exist_ok=True)

    def update_attr(self, attr):
        """
        从环境中读取参数
        :param attr:
        :return:
        """
        try:
            attr_val = getattr(self.env, attr) if self.env_args is None else self.env_args[attr]
        except Exception as error:
            print(f"| Argument.update_attr({attr}) Error: {error}")
            attr_val = None

        return attr_val

    def if_off_policy(self):
        name = self.agent_class.__name__
        if_off_policy = all((name.find('PPO') == -1, name.find('A2C') == -1))    # 如果是基于PPO或者A2C的RL模型，则是on-policy；不基于PPO和A2C的是off-policy
        return if_off_policy


def build_env(env=None, env_func=None, env_args=None):
    if env is not None:
        env = deepcopy(env)
    elif env_func.__module__ == 'gym.envs.registration':
        import gym
        gym.logger.set_level(40)
        env = env_func(id=env_args['env_name'])
    else:
        env = env_func(**kwargs_filter(env_func.__init__, env_args.copy()))

    for attr_str in ('state_dim', 'action_dim', 'max_step', 'if_discrete', 'target_return'):
        if (not hasattr(env, attr_str)) and (attr_str in env_args):
            setattr(env, attr_str, env_args[attr_str])

    return env


def kwargs_filter(func, kwargs):
    import inspect

    sign = inspect.signature(func).parameters.values()
    sign = set([val.name for val in sign])

    common_args = sign.intersection(kwargs.keys())
    filtered_kwargs = {key : kwargs[key] for key in common_args}
    return filtered_kwargs

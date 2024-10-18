import os
import time

import numpy as np
import torch

from Train.config import Arguments, build_env
from Train.evaluate import Evaluator
from Train.replay_buffer import ReplayBuffer, ReplayBufferList

from util.uniform_instance_gen import uni_instance_gen


def init_agent(args, gpu_id, env):
    """
    初始化智能体
    :param args:    Argument，智能体参数以及训练运行参数
    :param gpu_id:  int，GPU编号
    :param env:     强化学习环境
    :return:
    """
    agent = args.agent_class(args.net_dim, args.state_dim, args.action_dim, gpu_id=gpu_id, args=args)
    agent.save_or_load_agent(args.cwd, if_save=False)

    if env is not None:
        '''prepare init state for exploration'''
        if args.env_num == 1:
            state = env.reset()
            assert isinstance(state, np.ndarray) or isinstance(state, torch.Tensor)
            assert state.shape == args.state_dim
            states = [state, ]
        else:
            states = env.reset()
            assert isinstance(states, torch.Tensor)
            assert states.shape == (args.env_num, args.state_dim)
        agent.states = states

    return agent


def init_buffer(args, gpu_id):
    if args.if_off_policy:
        buffer = ReplayBuffer(gpu_id=gpu_id, max_capacity=args.max_memo, state_dim=args.state_dim, action_dim=1 if args.if_discrete else args.action_dim)
        buffer.save_or_load_history(args.cwd, if_save=False)
    else:
        buffer = ReplayBufferList()

    return buffer


def init_evaluator(args, gpu_id):
    eval_func = args.eval_env_func if getattr(args, "eval_env_func") else args.env_func
    eval_args = args.eval_env_args if getattr(args, "eval_env_args") else args.env_args
    eval_env = build_env(args.env, eval_func, eval_args)
    evaluator = Evaluator(cwd=args.cwd, agent_id=gpu_id, eval_env=eval_env, args=args)
    return evaluator


def train_and_evaluate(args):
    """
    学习训练和验证入口
    :param args:
    :return:
    """
    torch.set_grad_enabled(False)
    args.init_before_train()
    gpu_id = args.learner_gpus

    '''初始化，包括agent、环境、经验池、验证器'''
    env = args.env
    steps = 0

    agent = init_agent(args, gpu_id, env)
    buffer = init_buffer(args, gpu_id)
    evaluator = init_evaluator(args, gpu_id)

    agent.state = env.reset()

    if args.if_off_policy:
        trajectory = agent.explore_env(env, args.num_seed_steps * args.num_steps_per_episode)
        buffer.update_buffer(trajectory)

    '''agent训练开始'''
    cwd = args.cwd
    break_step = args.break_step
    break_epoch = args.break_epoch
    target_step = args.horizon_len      # TODO 探索步数应该最少是tasks数
    explore_num = args.explore_times
    if_allow_break = args.if_allow_break
    if_offer_policy = args.if_off_policy

    steps = 0
    if_train = True
    while if_train:
        # epoch_start = time.time()
        trajectory = agent.explore_env(env, target_step)
        # explore_end = time.time()
        steps = target_step

        if if_offer_policy:
            _, r_exp = buffer.update_buffer(trajectory)
            explore_reward_avg = np.mean(trajectory[-1])
            torch.set_grad_enabled(True)
            logging_tuple = agent.update_net(buffer)
            torch.set_grad_enabled(False)
        else:
            r_exp = trajectory[3].mean().item()
            explore_reward_avg = trajectory[3].sum().item() / explore_num
            torch.set_grad_enabled(True)
            logging_tuple = agent.update_net(trajectory)
            torch.set_grad_enabled(False)

        # update_end = time.time()
        (if_reach_goal, if_save) = evaluator.evaluate_save_and_plot(agent.actor, steps, r_exp, explore_reward_avg, logging_tuple)
        dont_break = not if_allow_break
        not_reached_goal = not if_reach_goal
        stop_dir_absent = not os.path.exists(f"{cwd}/stop")
        if_train = (
            (dont_break or not_reached_goal) and evaluator.epoch <= break_epoch and evaluator.total_step <= break_step and stop_dir_absent
        )
        # epoch_end = time.time()
        # print("Total Time: " + str(epoch_end-epoch_start) + "| Explore Time: " + str(explore_end-epoch_start) + "| Update Time: " + str(update_end-explore_end))

    print(f'| UsedTime: {time.time() - evaluator.start_time: .0f} | SavaDir: {cwd}')

    agent.save_or_load_agent(cwd, if_save=True)


'''TODO 多进程训练'''


def train_and_evaluate_mp(args: Arguments):
    pass


class PipeLearner:
    def __init__(self):
        pass

    @staticmethod
    def run():
        pass


class PipeEvaluator:
    def __init__(self):
        pass

    def evaluate_and_save_mp(self):
        pass

    def run(self):
        pass


def safely_terminate_process(processes):
    for p in processes:
        try:
            p.kill()
        except OSError as e:
            print(e)


import sys, os
sys.path.append(os.path.dirname(sys.path[0]))

from Train.config import Arguments
from Train.train import init_agent
from Env.SJSSPEnv import SJsspEnv
from Env.DJSSPEnv import DJsspEnv
from RL.AgentPPO import AgentPPO
import time

import numpy as np
import torch

"""求解instances"""


def solve(env_args, instances, pretrained_actor_path):
    """
    加载预训练模型，求解样例
    :param pretrained_actor_path:
    :param env_args:  环境模型参数，用于加载训练好的模型
    :param instances:   待求解的样例
    :return:
    """

    # 1.加载模型 Agent.actor
    env = SJsspEnv(job_num=env_args['n_jobs'], machine_num=env_args['n_machines'], args=env_args)
    agent = AgentPPO
    args = Arguments(agent, env)

    agent = init_agent(args, args.learner_gpus, env)
    # actor_path = "../TrainingLog/SJsspEnv_PPO_0_J20_M15_2022_12_19/actor_000003538800_-1041.500.pth"
    agent.actor.load_state_dict(torch.load(pretrained_actor_path))
    #
    # instances_path = "../Instances/Benchmark/ta/J20_M15.npy"
    # instances = np.load(instances_path)
    # 2.创建实例环境，逐步求解
    all_solutions = []
    all_times = []
    for instance in instances:
        sols = []
        run_time = []
        for i in range(1):
            state = env.reset(instance)

            start = time.time()
            while not env.done():
                state = torch.as_tensor(state, dtype=torch.float32, device=agent.device)
                action = agent.actor.get_action(state.unsqueeze(0))[0]
                action = agent.actor.convert_action_for_env(action).detach().cpu().numpy().item()
                state, reward, _, _ = env.step(action)
            end = time.time()

            sols.append(env.max_end_time)
            run_time.append(end-start)
            # if env.max_end_time < sol:
            #     sol = int(env.max_end_time)
            # print(env.max_end_time)

        # print(sols)
        all_solutions.append(sols)
        all_times.append(np.mean(run_time))
        # print(np.mean(run_time))

    # print(min_solution)
    print(np.mean(all_times))
    return all_solutions, all_times


    # 3.打印保存解


def solve_dynamic(env, instances, bias, pretrained_actor_path, repeat=200):
    agent = AgentPPO
    args = Arguments(agent, env)

    agent = init_agent(args, args.learner_gpus, env)
    agent.actor.load_state_dict(torch.load(pretrained_actor_path))

    all_solutions = []
    all_times = []

    num = instances.shape[0]
    for i in range(num):
        instance = instances[i]
        b = bias[i]
        sols = []
        run_time = []

        for _ in range(repeat):
            state = env.reset(instance, b)

            start = time.time()
            while not env.done():
                state = torch.as_tensor(state, dtype=torch.float32, device=agent.device)
                action = agent.actor.get_action(state.unsqueeze(0))[0]
                action = agent.actor.convert_action_for_env(action).detach().cpu().numpy().item()
                state, reward, _, _ = env.step(action)
            end = time.time()

            sols.append(env.max_end_time)
            run_time.append(end - start)

        all_solutions.append(sols)
        all_times.append(np.mean(run_time))

    return all_solutions, all_times


if __name__ == "__main__":
    n_jobs = 20
    n_machines = 15

    gat_args = {
        "n_layers": 3,
        "n_head_per_layers": [8, 8, 1],
        "in_fea_dim": 3,
        "out_fea_dim": 64,
        "n_features_per_layers": [3, 64, 128, 64],
        "jobs_num": n_jobs,
        "tasks_num": n_jobs * n_machines
    }
    env_args = {
        'n_jobs': 20,
        'n_machines': 15,
        'env_num': 1,
        'env_name': 'SJssPEnv',
        'max_step': 10000,
        'fea_dim': 3,
        'action_dim': 1,
        'if_discrete': True,
        'target_return': 0,
        'gat_args': gat_args
    }

    sizes = {'ta': [(15, 15), (20, 15), (20, 20), (30, 15), (30, 20), (50, 15), (50, 20), (100, 20)],
             'dmu': [(20, 15), (20, 20), (30, 15), (30, 20), (40, 15), (40, 20), (50, 15), (50, 20)],
             'gen': [(40, 30, 100), (50, 30, 100), (100, 30, 100), (20, 15, 50), (20, 15, 150), (20, 15, 300)]}
    pre_trained_actor_path = "../TrainingLog/SJssPEnv_PPO_0_J30_M20_H199_2023_03_22/actor_000000216000_-2806.625.pth"
    instances_path = "../Instances/Benchmark/ta/J20_M15.npy"
    instances = np.load(instances_path)
    for set_name in ['gen']:
        for size in sizes[set_name]:
            # name = f'gen_size20_15_range{high}'
            job_num = size[0]
            machine_num = size[1]
            high_bound = size[2]
            instances_path = f"../Instances/DataGen/gen_J{job_num}_M{machine_num}_H{high_bound}.npy"
            instances = np.load(instances_path)[0:20]
            env_args["n_jobs"] = size[0]
            env_args["n_machines"] = size[1]

            print(f"{set_name} J{job_num} M{machine_num} H{high_bound}")
            solutions, times = solve(env_args, instances, pre_trained_actor_path)

            # np.save(f"../TestLog/{name}_solu", solutions)
            np.save(f"../TestLog/gen/gat_J{job_num}_M{machine_num}_H{high_bound}_time.npy", times)

    # sizes = {'ta': [(15, 15), (20, 15), (20, 20), (30, 15), (30, 20)],
    #          'dmu': [(20, 15), (20, 20), (30, 15), (30, 20), (40, 15)]}
    # # ta
    # # set_name = 'ta'
    # for set_name in ['ta', 'dmu']:
    #     for size in sizes[set_name]:
    #         for sigma in [10, 20]:
    #             J = size[0]
    #             M = size[1]
    #             name = f"J{J}_M{M}"
    #             instance_path = f"../Instances/Benchmark/{set_name}/{name }.npy"
    #             bias_path = f"../Instances/Benchmark/{set_name}/bias/{sigma}/{name}.npy"
    #             instances = np.load(instance_path)
    #             bias = np.load(bias_path)
    #
    #             lamb = 1.0 - sigma/100
    #             theta = lamb/2
    #
    #             env = DJsspEnv(J, M, sigma=sigma/100, lamb=lamb, theta=theta, args=env_args)
    #
    #             solutions, times = solve_dynamic(env, instances, bias, pre_trained_actor_path)
    #             np.save(f"../TestLog/{set_name}/dynamic/{sigma}/{name}_solu.npy", solutions)
    #             np.save(f"../TestLog/{set_name}/dynamic/{sigma}/{name}_time.npy", times)
    #             print(f"{set_name}_{name }_Dynamic={sigma/100} Done.")
    # #
    # # state = env.reset(instance=instances[0], bias=bias[0])
    # # # solutions, times = solve(env_args, instances, pre_trained_actor_path)
    # #
    # # agent = AgentPPO
    # # args = Arguments(agent, env)
    # #
    # # agent = init_agent(args, args.learner_gpus, env)
    # # agent.actor.load_state_dict(torch.load(pre_trained_actor_path))
    # #
    # # while not env.done():
    # #     state = torch.as_tensor(state, dtype=torch.float32, device=agent.device)
    # #     action = agent.actor.get_action(state.unsqueeze(0))[0]
    # #     action = agent.actor.convert_action_for_env(action).detach().cpu().numpy().item()
    # #     state, reward, _, _ = env.step(action)
    # #
    # # print(env.max_end_time)
# dynamic_10_ta_15_15_solu = np.load("TestLog/ta/dynamic/10/J15_M15_solu.npy")
# dynamic_10_ta_15_15_solu.sort(axis=1)
# dynamic_10_ta_20_15_solu = np.load("TestLog/ta/dynamic/10/J20_M15_solu.npy")
# dynamic_10_ta_20_15_solu.sort(axis=1)
# dynamic_10_ta_20_20_solu = np.load("TestLog/ta/dynamic/10/J20_M20_solu.npy")
# dynamic_10_ta_20_20_solu.sort(axis=1)
# dynamic_10_ta_30_15_solu = np.load("TestLog/ta/dynamic/10/J30_M15_solu.npy")
# dynamic_10_ta_30_15_solu.sort(axis=1)
# dynamic_10_ta_30_20_solu = np.load("TestLog/ta/dynamic/10/J30_M20_solu.npy")
# dynamic_10_ta_30_20_solu.sort(axis=1)
#
# dynamic_10_dmu_20_15_solu = np.load("TestLog/dmu/dynamic/10/J20_M15_solu.npy")
# dynamic_10_dmu_20_15_solu.sort(axis=1)
# dynamic_10_dmu_20_20_solu = np.load("TestLog/dmu/dynamic/10/J20_M20_solu.npy")
# dynamic_10_dmu_20_20_solu.sort(axis=1)
# dynamic_10_dmu_30_15_solu = np.load("TestLog/dmu/dynamic/10/J30_M15_solu.npy")
# dynamic_10_dmu_30_15_solu.sort(axis=1)
# dynamic_10_dmu_30_20_solu = np.load("TestLog/dmu/dynamic/10/J30_M20_solu.npy")
# dynamic_10_dmu_30_20_solu.sort(axis=1)
# dynamic_10_dmu_40_15_solu = np.load("TestLog/dmu/dynamic/10/J40_M15_solu.npy")
# dynamic_10_dmu_40_15_solu.sort(axis=1)
#
# dynamic_20_dmu_20_15_solu = np.load("TestLog/dmu/dynamic/20/J20_M15_solu.npy")
# dynamic_20_dmu_20_15_solu.sort(axis=1)
# dynamic_20_dmu_20_20_solu = np.load("TestLog/dmu/dynamic/20/J20_M20_solu.npy")
# dynamic_20_dmu_20_20_solu.sort(axis=1)
# dynamic_20_dmu_30_15_solu = np.load("TestLog/dmu/dynamic/20/J30_M15_solu.npy")
# dynamic_20_dmu_30_15_solu.sort(axis=1)
# dynamic_20_dmu_30_20_solu = np.load("TestLog/dmu/dynamic/20/J30_M20_solu.npy")
# dynamic_20_dmu_30_20_solu.sort(axis=1)
# dynamic_20_dmu_40_15_solu = np.load("TestLog/dmu/dynamic/20/J40_M15_solu.npy")
# dynamic_20_dmu_40_15_solu.sort(axis=1)
#
# dynamic_20_ta_15_15_solu = np.load("TestLog/ta/dynamic/20/J15_M15_solu.npy")
# dynamic_20_ta_15_15_solu.sort(axis=1)
# dynamic_20_ta_20_15_solu = np.load("TestLog/ta/dynamic/20/J20_M15_solu.npy")
# dynamic_20_ta_20_15_solu.sort(axis=1)
# dynamic_20_ta_20_20_solu = np.load("TestLog/ta/dynamic/20/J20_M20_solu.npy")
# dynamic_20_ta_20_20_solu.sort(axis=1)
# dynamic_20_ta_30_15_solu = np.load("TestLog/ta/dynamic/20/J30_M15_solu.npy")
# dynamic_20_ta_30_15_solu.sort(axis=1)
# dynamic_20_ta_30_20_solu = np.load("TestLog/ta/dynamic/20/J30_M20_solu.npy")
# dynamic_20_ta_30_20_solu.sort(axis=1)

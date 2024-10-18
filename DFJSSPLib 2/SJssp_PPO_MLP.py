from Train.config import Arguments
from RL.AgentPPO import AgentPPO
from Train.train import train_and_evaluate
from Env.SJSSPEnv import SJsspEnv
from Env.DJSSPEnv import DJsspEnv

"""
配置环境、模型，训练入口
"""


class Dict:
    # __setattr__ = dict.__setitem__
    # __getattr__ = dict.__getitem__

    def __init__(self, ele):
        self.dictionary = ele

    def __getattr__(self, key):
        if key in self.dictionary:
            return self.dictionary[key]
        else:
            raise AttributeError(f"No Attribute '{key}'")


def dict_to_obj(dictObj):
    if not isinstance(dictObj, dict):
        return dictObj
    d = Dict()
    for k, v in dictObj.items():
        d[k] = dict_to_obj(v)
    return d


def run():
    n_jobs = 30
    n_machines = 20

    gat_args = {
        "n_layers": 3,
        "n_head_per_layers": [8, 8, 1],
        "in_fea_dim": 3,
        "out_fea_dim": 64,
        "n_features_per_layers": [3, 64, 128, 64],
        "jobs_num": n_jobs,
        "tasks_num": n_jobs * n_machines
    }
    # gat_args = Dict(gat_args)
    env_args = {
        'env_num': 1,
        'env_name': 'SJssPEnv',
        'high_bound': 199,
        'max_step': 10000,
        'fea_dim': 3,
        'action_dim': 1,
        'if_discrete': True,
        'target_return': 0,
        'gat_args': gat_args
    }
    env_args = Dict(env_args)
    env = SJsspEnv(job_num=n_jobs, machine_num=n_machines, args=env_args)
    agent = AgentPPO
    args = Arguments(agent, env)

    train_and_evaluate(args)


if __name__ == "__main__":
    run()

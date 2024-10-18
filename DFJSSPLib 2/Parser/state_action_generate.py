import math
import numpy as np
from Env.SJSSPEnv import SJsspEnv


# 以什么样的文件结构存放state-action
# 和action一起放在instance文件夹下，然后按照size存一个batch
npy_path = '../Instances/OrlibNpy'

with open('../Instances/dataset_info') as f:
    columns = f.readline()
    instances = f.readlines()

for i in instances:
    i = i.split()
    instance_name = i[0]
    job_num = int(i[1])
    machine_num = int(i[2])

    instance_dir = npy_path + '/' + instance_name
    instance_path = instance_dir + '/' + instance_name + '.npy'
    instance = np.load(instance_path)

    actions_seq = np.load(instance_dir + '/actions.npy')

    env = SJsspEnv(job_num, machine_num)

    all_solutions = []
    for actions in actions_seq:

        state = env.reset(instance)

        pairs = []

        for a in actions:
            pairs.append(state)
            state = env.step(a)[0]

        all_solutions.append(pairs)

    np.save(instance_dir+'/data.npy', all_solutions)
    # np.load(instance_dir+'/data.npy', allow_pickle=True)
    print(instance_name + " Done.")

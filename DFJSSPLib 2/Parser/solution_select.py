import os
import math
import random

import numpy as np
import pandas as pd


def select_sol(sols, nums=5):
    # 依据余弦相似度筛选解
    # 最远点采样

    # 随机采样
    idx = [random.randint(0, len(sols)) for _ in range(nums)]
    sample_result = random.sample(sols, nums)

    return sample_result
    pass


def operation2task(operation_ordering, job_num, machine_num):
    # operation order to task idx
    # 操作的编号是矩阵中从上往下编号，属于列优先，任务id是行优先，从左往右编号
    task_ordering = []
    for i in operation_ordering:
        j = i % job_num
        if j == 0:
            j = job_num

        k = math.ceil(i / job_num)

        t = machine_num * (j-1) + (k-1)

        task_ordering.append(t)

    return task_ordering


optimals_path = '../Instances/Optimals'
npy_path = '../Instances/OrlibNpy'

solutions = os.listdir(optimals_path)

# 读取optimal solution
#
num = 0
dataset_info = []
for s in solutions:

    instance_name = s.split('_')[0]

    s_path = optimals_path + '/' + s

    with open(s_path) as f:
        lines = f.readlines()

    sols = [list(map(int, i.split())) for i in lines]

    if len(lines) > 10:
        # 如果最优解数量过多筛选差距最大的10个
        sols = select_sol(sols, nums=10)
        pass

    # operation ordering to task id
    job_num, machine_num = np.load(npy_path+'/'+instance_name+'/size.npy')

    info = [instance_name, job_num, machine_num, len(sols)]
    dataset_info.append(info)
    num += len(sols) * job_num * machine_num
    action_seqs = [operation2task(sol, job_num, machine_num) for sol in sols]
    np.save(npy_path+'/'+instance_name+'/actions.npy', action_seqs)
    print(instance_name+" " + str(len(sols)) + " Done.")

t = pd.DataFrame(dataset_info, columns=['instance', 'job_num', 'machine_num', 'solution_num'])
t.to_csv('../Instances/dataset_info', sep='\t', index=False)
# np.savetxt('../Instances/dataset_info', dataset_info, delimiter='\t', header="instance,job_num,machine_num,solution_num")
print(num)


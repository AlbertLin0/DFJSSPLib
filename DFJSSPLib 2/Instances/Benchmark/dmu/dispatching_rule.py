import os
import time

# import matplotlib.cm
import numpy as np
import torch

"""
优先调度规则：
    SPT：加工时间短的优先
    MWR：最长剩余加工时间的优先
    MOR：最多剩余任务的优先
"""


def spt(instance, job_num, machine_num, bias=None):
    """
    for non-flexible job shop

    :param instance: [time, order]
    :param job_num:
    :param machine_num:
    :param bias: dynamic use
    :return:
    """
    time = instance[0]
    order = instance[1]
    start_time = np.zeros((job_num, machine_num), dtype=np.int32)
    end_time = np.zeros((job_num, machine_num), dtype=np.int32)

    machine_process = [[] for _ in range(machine_num)]  # 记录每个机器上任务的执行情况 (job id, task id in job)
    candidates = []  # 记录可被调度的任务信息 (job_id, task_id_in_job, machine id, processing time)

    for i in range(job_num):
        candidates.append((i, 0, order[i, 0] - 1, time[i, 0]))

    # 初始化
    # for i in range(job_num):
    #     candidates.append((i, 0, order[i, 0]-1, time[i, 0]))
    task_seq = []
    # for k in range(machine_num):
    #     candidates = []
    #     for i in range(job_num):
    #         candidates.append((i, k, order[i, k] - 1, time[i, k]))
    #
    #     candidates_sorted = sorted(candidates, key=lambda x: x[-1])
    while True:
        candidates_s = sorted(candidates, key=lambda x: x[-1])
        p = candidates_s[0]
        # for p in candidates_sorted:
        if p[-1] == np.inf:
            break

        job_id = p[0]
        task_id_in_job = p[1]
        machine_id = p[2]
        processing_time = p[3]

        action = job_id*machine_num + task_id_in_job

        # [[action], [machine_id], [job_id], [task_id], [duration]]
        # task_seq.append(action)
        task_seq.append([[action], [machine_id], [job_id], [task_id_in_job], [processing_time]])

        # 主要就是确定任务开始时间
        # if len(machine_process[machine_id]) == 0:
        #     if task_id_in_job == 0:
        #         start_time[job_id, task_id_in_job] = 0
        #     else:
        #         start_time[job_id, task_id_in_job] = end_time[job_id, task_id_in_job-1]   # 同一任务前置任务结束时可开始
        # else:
        #     pre_task_in_machine = machine_process[machine_id][-1]  # job_id 0, task_id 1
        #     if task_id_in_job == 0:
        #         start_time[job_id, task_id_in_job] = end_time[pre_task_in_machine[0], pre_task_in_machine[1]]
        #     else:
        #         start_time[job_id, task_id_in_job] = max(end_time[job_id, task_id_in_job-1],
        #                                                  end_time[pre_task_in_machine[0], pre_task_in_machine[1]])
        #
        # end_time[job_id, task_id_in_job] = start_time[job_id, task_id_in_job] + processing_time
        # machine_process[machine_id].append(p)

        # 更新candidate
        if task_id_in_job < machine_num-1:
            next_in_job = task_id_in_job+1
            candidates[job_id] = (job_id, next_in_job, order[job_id, next_in_job]-1, time[job_id, next_in_job])
        else:
            candidates[job_id] = (job_id, task_id_in_job+1, np.inf, np.inf)

    return task_seq


def mwr(instance, job_num, machine_num):
    """
    for non-flexible job shop
    最长剩余工作时间优先.

    :param instance: [time, order]
    :param job_num:
    :param machine_num:
    :return:
    """

    time = instance[0]
    order = instance[1]

    candidates = []  # 记录可被调度的任务信息 (job_id, task_id_in_job, machine id, processing time，remain work time this job)

    for i in range(job_num):
        candidates.append((i, 0, order[i, 0] - 1, time[i, 0], np.sum(time[i])))

    task_seq = []

    while True:
        candidates_s = sorted(candidates, key=lambda x: x[-1])
        p = candidates_s[-1]
        # for p in candidates_sorted:
        if p[-1] == -np.inf:
            break

        job_id = p[0]
        task_id_in_job = p[1]
        machine_id = p[2]
        processing_time = p[3]
        remain_time = p[4]

        action = job_id * machine_num + task_id_in_job
        # task_seq.append(action)
        task_seq.append([[action], [machine_id], [job_id], [task_id_in_job], [processing_time]])

        if task_id_in_job < machine_num-1:
            next_in_job = task_id_in_job+1
            candidates[job_id] = (job_id, next_in_job, order[job_id, next_in_job]-1, time[job_id, next_in_job],
                                  remain_time)
        else:
            candidates[job_id] = (job_id, task_id_in_job+1, -np.inf, -np.inf, -np.inf)

    return task_seq


def mor(instance, job_num, machine_num):
    """
    for non-flexible job shop
    最多剩余工序优先

    :param instance: [time, order]
    :param job_num:
    :param machine_num:
    :return:
    """

    time = instance[0]
    order = instance[1]

    candidates = []

    for i in range(job_num):
        candidates.append((i, 0, order[i, 0] - 1, time[i, 0], machine_num))

    task_seq = []

    while True:
        c = candidates.copy()
        np.random.shuffle(c)
        candidates_s = sorted(c, key=lambda x: x[-1])
        p = candidates_s[-1]
        # for p in candidates_sorted:
        if p[-1] == -np.inf:
            break

        job_id = p[0]
        task_id_in_job = p[1]
        machine_id = p[2]
        processing_time = p[3]
        remain_task_num = p[4]

        action = job_id * machine_num + task_id_in_job
        # task_seq.append(action)
        task_seq.append([[action], [machine_id], [job_id], [task_id_in_job], [processing_time]])

        if task_id_in_job < machine_num-1:
            next_in_job = task_id_in_job+1
            candidates[job_id] = (job_id, next_in_job, order[job_id, next_in_job]-1, time[job_id, next_in_job],
                                  remain_task_num)
        else:
            candidates[job_id] = (job_id, task_id_in_job+1, -np.inf, -np.inf, -np.inf)

    return task_seq


cnames = ['#F0F8FF', '#FAEBD7', '#00FFFF', '#7FFFD4', '#F0FFFF', '#F5F5DC', '#FFE4C4',
          '#000000', '#FFEBCD', '#0000FF', '#8A2BE2', '#A52A2A', '#DEB887', '#5F9EA0',
          '#7FFF00', '#D2691E', '#FF7F50', '#6495ED', '#FFF8DC', '#DC143C', '#00FFFF',
          '#00008B', '#008B8B', '#B8860B', '#A9A9A9', '#006400', '#BDB76B', '#8B008B',
          '#556B2F', '#FF8C00', '#9932CC', '#8B0000', '#E9967A', '#8FBC8F', '#483D8B',
          '#2F4F4F', '#00CED1', '#9400D3', '#FF1493', '#00BFFF', '#696969', '#1E90FF',
          '#B22222', '#FFFAF0', '#228B22', '#FF00FF', '#DCDCDC', '#F8F8FF', '#FFD700',
          '#DAA520', '#808080', '#008000', '#ADFF2F', '#F0FFF0', '#FF69B4', '#CD5C5C',
          '#4B0082', '#FFFFF0', '#F0E68C', '#E6E6FA', '#FFF0F5', '#7CFC00', '#FFFACD',
          '#ADD8E6', '#F08080', '#E0FFFF', '#FAFAD2', '#90EE90', '#D3D3D3', '#FFB6C1',
          '#FFA07A', '#20B2AA', '#87CEFA', '#778899', '#B0C4DE', '#FFFFE0', '#00FF00',
          '#32CD32', '#FAF0E6', '#FF00FF', '#800000', '#66CDAA', '#0000CD', '#BA55D3',
          '#9370DB', '#3CB371', '#7B68EE', '#00FA9A', '#48D1CC', '#C71585', '#191970',
          '#F5FFFA', '#FFE4E1', '#FFE4B5', '#FFDEAD', '#000080', '#FDF5E6', '#808000',
          '#6B8E23', '#FFA500', '#FF4500', '#DA70D6', '#EEE8AA', '#98FB98', '#AFEEEE',
          '#DB7093', '#FFEFD5', '#FFDAB9', '#CD853F', '#FFC0CB', '#DDA0DD', '#B0E0E6',
          '#800080', '#FF0000', '#BC8F8F', '#4169E1', '#8B4513', '#FA8072', '#FAA460',
          '#2E8B57', '#FFF5EE', '#A0522D', '#C0C0C0', '#87CEEB', '#6A5ACD', '#708090',
          '#FFFAFA', '#00FF7F', '#4682B4', '#D2B48C', '#008080', '#D8BFD8', '#FF6347',
          '#40E0D0', '#EE82EE', '#F5DEB3', '#FFFFFF', '#F5F5F5', '#FFFF00', '#9ACD32']


def draw_gantt(start_time_on_machine, id_on_machine, duration, name):
    """
    绘制调度结果的甘特图，pyplot.barh()
    :param start_time_on_machine: 每个机器上各任务的开始时间
    :param id_on_machine: 每个机器上各任务的id
    :param duration: 每个任务的加工时间
    :return:
    """

    machine_num, job_num = start_time_on_machine.shape

    # duration_on_machine = np.empty(start_time_on_machine.shape)
    duration = duration.flatten()
    # for m in range(machine_num):
    #     for j in range(job_num):
    #         cur_id = id_on_machine[m, j]
    #         duration_on_machine[m, j] = duration[cur_id]

    y_ticks = ["M"+str(i+1) for i in range(machine_num)]
    # y_ticks = [''].append(y_ticks)
    # 依据任务开始时间与加工持续时间绘制甘特图
    import matplotlib.pyplot as plt
    cmap = matplotlib.cm.get_cmap('tab20', job_num)
    plt.figure(figsize=(12.8, 9.6))
    for j in range(job_num):
        plt.barh(y=0, width=0, left=0, height=0, color=cmap(j), label="Job "+str(j+1))

    for m in range(machine_num):
        for j in range(job_num):
            job_id = int(id_on_machine[m, j] / machine_num)
            plt.barh(y=m, width=duration[id_on_machine[m, j]], left=start_time_on_machine[m, j], color=cmap(job_id))

    plt.xticks(fontsize=20)
    plt.yticks(ticks=np.arange(machine_num), labels=y_ticks, fontsize=20)
    plt.title(name, fontsize=20)
    plt.legend(loc=2, bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=12)
    plt.xlabel("Time", fontsize=24)
    plt.ylabel("Machines", fontsize=24)

    plt.show()
    pass


def heuristicSPT(num_jobs, num_mc, machines, durations):
    machines_ = np.array(machines)
    tmp = np.zeros((num_jobs, num_mc + 1), dtype=int)
    tmp[:, :-1] = machines_
    machines_ = tmp

    durations_ = np.array(durations)
    tmp = np.zeros((num_jobs, num_mc + 1), dtype=int)
    tmp[:, :-1] = durations_
    durations_ = tmp

    indices = np.zeros([num_jobs], dtype=int)

    # Internal variables
    previousTaskReadyTime = np.zeros([num_jobs], dtype=int)
    machineReadyTime = np.zeros([num_mc], dtype=int)

    placements = [[] for _ in range(num_mc)]

    # While...
    while (not np.array_equal(indices, np.ones([num_jobs], dtype=int) * num_mc)):

        machines_Idx = machines_[range(num_jobs), indices]
        durations_Idx = durations_[range(num_jobs), indices]

        # 1: Check previous Task and machine availability
        mask = np.zeros([num_jobs], dtype=bool)

        for j in range(num_jobs):

            if previousTaskReadyTime[j] == 0 and machineReadyTime[machines_Idx[j]] == 0 and indices[j] < num_mc:
                mask[j] = True

        # 2: Competition SPT

        for m in range(num_mc):

            job = None
            duration = 99999

            for j in range(num_jobs):

                if machines_Idx[j] == m and durations_Idx[j] < duration and mask[j]:
                    job = j
                    duration = durations_Idx[j]

            if job != None:
                placements[m].append([job, indices[job]])

                previousTaskReadyTime[job] += durations_Idx[job]
                machineReadyTime[m] += durations_Idx[job]

                indices[job] += 1

        # time +1

        previousTaskReadyTime = np.maximum(previousTaskReadyTime - 1, np.zeros([num_jobs], dtype=int))
        machineReadyTime = np.maximum(machineReadyTime - 1, np.zeros([num_mc], dtype=int))

    return placements

import os 

def heuristic_test():
    # instances = np.load("../Instances/Benchmark/ta/J30_M20.npy")
    # from Env.SJSSPEnv import SJsspEnv
    # env = SJsspEnv(15, 15)

    # task_seq = spt(instances, 15, 15)
    # makespan, end_times = task_seq_to_makespan(None, task_seq, 15, 15)

    # method = spt
    # save_path = f"../TestLog/gen/{method.__name__}"  # 1
    is_dynamic = False
    print(os.getcwd())
    # (job_num, machine_num, high_bound)
    # params = [(6, 6, 100), (10, 10, 100), (15, 10, 100), (40, 30, 100), (50, 30, 100), (100, 30, 100), (20, 15, 50),
    #           (20, 15, 150), (20, 15, 300)]

    set_names = ['ta', 'dmu']
    set_sizes = {'ta': [(15, 15), (20, 15), (20, 20), (30, 15), (30, 20), (50, 15), (50, 20), (100, 20)],
                 'dmu': [(20, 15), (20, 20), (30, 15), (30, 20), (40, 15), (40, 20), (50, 15), (50, 20)]}

    set_dmu = [(20, 15), (20, 20), (30, 15), (30, 20), (40, 15), (40, 20), (50, 15), (50, 20)]

    for method in (spt, mwr, mor):
        print(method.__name__)
        save_path = f"./TestLog/{method.__name__}"
        # record = {}
        # for size in [(15, 15), (20, 15), (20, 20), (30, 15), (30, 20), (50, 15), (50, 20), (100, 20)]:   # 4
        for param in set_dmu:
            job_num = param[0]
            machine_num = param[1]
            # high_bound = param[2]

            instances = np.load(f"./J{job_num}_M{machine_num}.npy")  # 5
            ins = np.load('J20_M15.npy')

            num = instances.shape[0]
            objs = []
            times = []
            orders = []
            for i in range(num):
                instance = instances[i]

                start = time.time()
                order = method(instance, job_num, machine_num)  # 2
                obj, _ = task_seq_to_makespan(None, order, job_num, machine_num)
                end = time.time()

                times.append(end - start)
                orders.append(order)

                objs.append(obj)

            print(f"gen J{job_num} M{machine_num} done.")
            print(np.mean(times))
            np.save(f"{save_path}/J{job_num}_M{machine_num}_time.npy", times)
            print()

    # for sigma in [10, 20]:
    #     print(sigma)
    #     for method in (spt, mwr, mor):
    #         print(method.__name__)
    #         for name in set_names:
    #             save_path = f"../TestLog/{name}/dynamic/{sigma}/{method.__name__}"
    #             # record = {}
    #             # for size in [(15, 15), (20, 15), (20, 20), (30, 15), (30, 20), (50, 15), (50, 20), (100, 20)]:   # 4
    #             for param in set_sizes[name]:
    #                 job_num = param[0]
    #                 machine_num = param[1]
    #                 # high_bound = param[2]
    
    #                 # name = f"gen_J{job_num}_M{machine_num}_H{high_bound}"
    #                 instances = np.load(f"../Instances/Benchmark/{name}/J{job_num}_M{machine_num}.npy")  # 5
    #                 # instances = np.load("../Instances/Benchmark/ta/J30_M20.npy")
    #                 # job_num = 30
    #                 # machine_num = 20
    
    #                 num = instances.shape[0]
    #                 if is_dynamic:
    #                     biases = np.load(f"../Instances/Benchmark/{name}/bias/{sigma}/J{job_num}_M{machine_num}.npy")     # 6
    
    #                 objs = []
    #                 times = []
    #                 orders = []
    #                 for i in range(num):
    #                     instance = instances[i]
    
    #                     if is_dynamic:
    #                         bias = biases[i]
    
    #                     start = time.time()
    #                     order = method(instance, job_num, machine_num)  # 2
    #                     obj, _ = task_seq_to_makespan(None, order, job_num, machine_num)
    #                     end = time.time()
    
    #                     times.append(end - start)
    #                     orders.append(order)
    
    #                     # env = SJsspEnv(job_num, machine_num)
    #                     # env.reset(instance)
    #                     # for o in order:
    #                     #     env.step(o)
    #                     #     # end_time.append(env.max_end_time)
    #                     #
    #                     # obj = env.max_end_time
    #                     if is_dynamic:
    #                         obj += np.sum(biases[i])
    #                     objs.append(obj)
    
    #                 # record[name] = np.mean(objs)
    
    #                 print(name)
    #                 # print(objs)
    #                 print(np.mean(times))
    #                 # np.save(f"{save_path}/{name}_solu.npy", objs)
    #                 np.save(f"{save_path}/J{job_num}_M{machine_num}_time.npy", times)
    #                 # np.save(f"../Instances/DataGen/solu/{method.__name__}/{name}.npy", orders)  # 3
    #                 #
    #                 print(name + f"J{job_num} M{machine_num} sigma {sigma} done.")
    #                 print()
    
    #                 # obj = heuristicSPT(30, 15, instance[1]-1, instance[0])
    
    #             # np.save(f"gen_{method.__name__}.npy", record)
    #

def solution_draw():
    from Env.SJSSPEnv import SJsspEnv
    instance = np.load("../Instances/Benchmark/ft06.npy")

    order = np.load("../Instances/Benchmark/ft06_actions.npy")[0]

    env = SJsspEnv(6, 6)
    env.reset(instance)

    for o in order:
        env.step(o)

    start_time_on_machine = env.task_start_time_on_machine
    id_on_machine = env.task_id_on_machine
    duration = env.tasks_duration
    name = "ft06 6x6 with optimal value 55"

    draw_gantt(start_time_on_machine, id_on_machine, duration, name)


def fjs_parse(job_num, machine_num, job_info):
    """
    解析fjs文件，输出为list，or tools可以接收的形式
    :param job_num:
    :param machine_num:
    :param job_info:
    :return:
    """
    assert len(job_info) == job_num, "jobs info length should equal to job num"
    jobs = []
    for i, info in enumerate(job_info):
        job = []
        info = info.split()
        tasks_num = int(info[0])
        i = 1
        while i < len(info):
            # tasks = []
            optional_machine_num = int(info[i])
            i += 1
            # j = 0
            tasks = []
            while optional_machine_num > 0:
                tasks.append((int(info[i+1]), int(info[i])-1))      # (processing time, machine id)
                i += 2
                optional_machine_num -= 1

            job.append(tasks)

        assert len(job) == tasks_num, "job should contain tasks of given num"

        jobs.append(job)

    return jobs


def spt_flex(instances, job_num, machine_num):
    candidates = []
    # job_num = len(instances)

    for i in range(job_num):
        candidates.append((i, 0, sorted(instances[i][0], key=lambda x: x[0])))

    task_seq = []
    while True:
        candidates_sorted = sorted(candidates, key=lambda x: x[-1][0])
        p = candidates_sorted[0]

        if p[1] == np.inf:
            break

        idx = np.random.randint(len(p[-1]))

        job_id = p[0]
        task_id = p[1]
        machine_id = p[-1][idx][1]
        duration = p[-1][idx][0]

        action = job_id * machine_num + task_id
        task_seq.append([[action], [machine_id], [job_id], [task_id], [duration]])     # task id, machine id, job id, task order in jobs, duration

        if task_id < machine_num-1:
            next_in_job = task_id+1
            candidates[job_id] = (job_id, next_in_job, sorted(instances[job_id][next_in_job], key=lambda x: x[0]))
        else:
            candidates[job_id] = (job_id, np.inf, [(np.inf, np.inf)])

    return np.array(task_seq)


def mwr_flex(instances, job_num, machine_num):
    candidates = []
    # total_times = []

    for i in range(job_num):
        rest_time = 0
        for j in range(len(instances[i])):
            rest_time += instances[i][j][0][0]    # for la instance in erv data
        candidates.append((i, 0, instances[i][0], rest_time))

    task_seq = []

    while True:
        candidates_sorted = sorted(candidates, key=lambda x: x[-1])
        p = candidates_sorted[-1]

        if p[-1] == -np.inf:
            break

        idx = np.random.randint(len(p[2]))

        job_id = p[0]
        task_id = p[1]
        machine_id = p[2][idx][1]
        duration = p[2][idx][0]
        rest_time = p[3]

        action = job_id * machine_num + task_id
        task_seq.append([[action], [machine_id], [job_id], [task_id], [duration]])

        if task_id < machine_num-1:
            next_in_job = task_id+1
            candidates[job_id] = (job_id, next_in_job, instances[job_id][next_in_job], rest_time-duration)
        else:
            candidates[job_id] = (job_id, np.inf, [(np.inf, np.inf)], -np.inf)

    return np.array(task_seq)
    pass


def mor_flex(instances, job_num, machine_num):
    candidates = []

    for i in range(job_num):
        candidates.append((i, 0, instances[i][0], len(instances[i])))

    task_seq = []
    while True:
        candidates_sort = sorted(candidates, key=lambda x: x[-1])

        p = candidates_sort[-1]

        if p[-1] == -np.inf:
            break

        idx = np.random.randint(len(p[2]))

        job_id = p[0]
        task_id = p[1]
        machine_id = p[2][idx][1]
        duration = p[2][idx][0]
        rest_tasks = p[3]

        action = job_id * machine_num + task_id
        task_seq.append([[action], [machine_id], [job_id], [task_id],
                         [duration]])  # task id, machine id, job id, task order in jobs, duration

        if task_id < machine_num-1:
            next_in_job = task_id+1
            candidates[job_id] = (job_id, next_in_job, instances[job_id][next_in_job], rest_tasks-1)
        else:
            candidates[job_id] = (job_id, np.inf, [(np.inf, np.inf)], -np.inf)

    return np.array(task_seq)


def task_seq_to_makespan(jobs, task_seq, job_num, machine_num):
    machine_record = [[] for _ in range(machine_num)]
    end_time = np.zeros((job_num, machine_num))

    for task in task_seq:
        task_id = task[0][0]
        machine_id = task[1][0]
        job_id = task[2][0]
        task_order_in_job = task[3][0]
        duration = task[4][0]

        # task_order_in_job = task_id % machine_num
        # 计算该task的开始时间
        # 先从前置任务计算开始时间的下界
        if task_order_in_job == 0:
            start_time = 0
        else:
            start_time = end_time[job_id, task_order_in_job-1]

        machine_status = machine_record[machine_id]
        idx = 0
        for record in machine_status:
            if start_time >= record[0]:
                # 该任务开始时间晚于当前任务的开始时间
                start_time = max(start_time, record[1])     # 则该任务开始时间的下界不早于当前任务的结束时间
                idx += 1
                continue
            elif start_time+duration <= record[0]:
                # 此处可插入
                break
            else:
                # start time早于当前任务开始时间，但是空隙不够插入，则开始时间下界为当前任务结束时间
                start_time = record[1]
                idx += 1
                continue

        machine_status.insert(idx, (start_time, start_time+duration))
        end_time[job_id, task_order_in_job] = start_time+duration

    makespan = np.max(end_time[:, -1])
    return makespan, machine_record


def flexible_test():
    # 'edata', 'rdata',
    instance_sets = ['edata', 'rdata', 'vdata']
    sizes = [(10, 5), (15, 5), (20, 5), (10, 10), (15, 10), (20, 10)]
    methods = [spt_flex, mwr_flex, mor_flex]

    for set_name in instance_sets:
        for size in sizes:
            job_num = size[0]
            machine_num = size[1]
            dir_path = f"../Instances/Benchmark/Hurink/{set_name}/J{job_num}_M{machine_num}/"

            instances_files = os.listdir(dir_path)
            instances_files.sort()
            solutions = []
            for method in methods:
                # print(method.__name__)
                all_times = []
                for file in instances_files:
                    instance_path = dir_path + file
                    with open(instance_path) as f:
                        info = f.readline()
                        instance = f.readlines()

                    jobs = fjs_parse(job_num, machine_num, instance)
                    makespans = {}

                    start = time.time()
                    task_seq = method(jobs, job_num, machine_num)
                    makespan, _ = task_seq_to_makespan(jobs, task_seq, job_num, machine_num)
                    end = time.time()
                    # makespans[method.__name__] = makespan
                    all_times.append(end-start)

                print(f"{method.__name__} {set_name} J{job_num}_M{machine_num} time {np.mean(all_times)}")

                # solutions.append(makespans)
                # print(f"{set_name} J{job_num}_M{machine_num} {file} {makespans}")

                np.save(f"../TestLog/Hurink/{set_name}_J{job_num}_M{machine_num}_{method.__name__}_time.npy", all_times)

            print()
    # instance_path = "../Instances/Benchmark/Hurink/edata/J10_M5/la01.fjs"
    #
    # with open(instance_path) as file:
    #     info = file.readline()
    #     instance = file.readlines()
    #
    # info = info.split()
    # job_num = int(info[0])
    # machine_num = int(info[1])
    # avg_optional_machine = float(info[2])
    #
    # jobs = fjs_parse(job_num, machine_num, instance)
    # task_seq = mwr_flex(jobs, job_num, machine_num)
    #
    # # task_seq = torch.from_numpy(task_seq)
    # make_span, start_end_time_on_machine = task_seq_to_makespan(jobs, task_seq, job_num, machine_num)
    # print(make_span)
    # from Env.FJSSPEnv import FJSPEnv
    # env_args = {
    #     "show_mode": "print",
    #     "batch_size": 1,
    #     "num_jobs": job_num,
    #     "num_mas": machine_num,
    #     "device": "cpu",
    #     "ope_feat_dim": 6,
    #     "ma_feat_dim": 3,
    # }
    #
    # env = FJSPEnv([instance_path], env_args, 'file')
    #
    # for action in task_seq:
    #     env.step(action)
    #
    # print(env.makespan_batch)


if __name__ == "__main__":
    heuristic_test()
    # flexible_test()

"""静态JSSP环境
主要是从JSSP实例转化为环境中的状态表示，以及给定动作之后状态的转移，奖励值的设计
"""
from typing import Optional, Union, List

import gym
import numpy as np
# from gym.core import RenderFrame


class SJsspEnv(gym.Env):
    # def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
    #     pass

    def __init__(self, job_num=15, machine_num=20, args=None):
        super(SJsspEnv, self).__init__()
        self.step_count = 0

        self.num_of_jobs = job_num
        self.num_of_machines = machine_num
        self.num_of_tasks = self.num_of_machines * self.num_of_jobs  # 任务总数，也是析取图中节点数量

        tasks_per_job = np.arange(start=0, stop=self.num_of_tasks, step=1).reshape(self.num_of_jobs, -1)
        self.first_col = tasks_per_job[:, 0]  # 每个job开始的任务序号
        self.last_col = tasks_per_job[:, -1]  # 每个job结束的任务序号

        # 超参数
        self.feature_normalize_factor = getattr(args, "feature_normalize_factor", 1000)
        self.high_bound = getattr(args, "high_bound", 99)  # task工作时长的上界
        self.low_bound = getattr(args, "low_bound", 1)  # task工作时长的下界
        self.reward_scale = getattr(args, "reward_scale", 0)  # 为了正奖励的奖励常数
        self.env_num = getattr(args, "env_num", 1)
        self.env_name = getattr(args, "env_name", 'SJsspEnv')
        self.max_step = min(getattr(args, "max_step", 10000), 8 * self.num_of_tasks)
        self.fea_dim = getattr(args, "fea_dim", 3)
        self.state_dim = self.num_of_tasks + self.fea_dim  # TODO need modify (self.num_of_tasks, self.num_of_tasks + self.fea_dim)
        self.state_dim = (self.num_of_tasks, self.num_of_tasks + self.fea_dim)
        self.action_dim = getattr(args, "action_dim", 1)
        self.if_discrete = getattr(args, "if_discrete", True)
        self.target_return = getattr(args, "target_return", self.low_bound * self.num_of_machines)
        self.horizon_len = 4 * self.num_of_tasks
        self.gat_args = getattr(args, "gat_args", {
            "n_layers": 3,
            "n_head_per_layers": [8, 8, 1],
            "in_fea_dim": 3,
            "out_fea_dim": 64,
            "n_features_per_layers": [3, 64, 128, 64],
            "jobs_num": self.num_of_jobs,
            "tasks_num": self.num_of_tasks
        })

        # 实例环境的基本设置
        self.machine_order = None  # 每个job访问machine的顺序，即不同task的顺序
        self.tasks_duration = None  # 每个task的执行时长
        self.adj = None  # 图的邻接矩阵的实时表示，主要记录task的完成情况
        self.fea = None  # 节点的特征向量
        self.complete_time_lower_bound = None
        self.init_quality = None
        self.max_end_time = None
        self.finished_mark = np.zeros((self.num_of_jobs, self.num_of_machines), dtype=np.single)
        self.candidate_task = self.first_col.astype(np.int64)  # 当前候选的task
        self.mask = np.full(shape=self.num_of_jobs, fill_value=0, dtype=bool)  # job是否完成

        # 辅助变量
        self.partial_solution_seq = []  # 部分解序列

        self.duration_temp = None
        self.duration_copy = None  # ...
        self.flags = []  # ...
        self.pos_rewards = 0  #

        self.task_start_time_on_machine = None  # 每个machine中各task的开始时间 n_m * n_j
        self.task_id_on_machine = None  # 每个machine中执行的各taskid   n_m * n_j

    def reset(self, instance=None):
        """
        随机生成一个JSP实例，使用JSP实例初始化环境，instance = [task加工时间，task访问的机器]    [n_j X n_m X 2]
        :return: init state: 图的邻接矩阵 n_t * n_t，节点的特征向量 n_t * f
        """
        if instance is None:
            instance = np.array(uni_instance_gen(self.num_of_jobs, self.num_of_machines, self.low_bound, self.high_bound))
        assert instance.shape == (2, self.num_of_jobs,
                                  self.num_of_machines), f"Illegal Instance Shape. Instance Shape Should be {(2, self.num_of_jobs, self.num_of_machines)}"
        self.step_count = 0
        self.machine_order = instance[-1]
        self.tasks_duration = instance[0]

        self.duration_copy = np.copy(self.tasks_duration)
        self.duration_temp = np.zeros_like(self.tasks_duration, dtype=np.single)

        # 基本环境设置、辅助变量重置
        self.finished_mark = np.zeros((self.num_of_jobs, self.num_of_machines), dtype=np.single)
        self.finished_mark[:, 0] = 1
        self.candidate_task = self.first_col.astype(np.int64)  # 当前候选的task
        self.mask = np.full(shape=self.num_of_jobs, fill_value=0, dtype=bool)  # job是否完成

        self.partial_solution_seq = []  # 部分解序列

        self.flags = []  # ...
        self.pos_rewards = 0  #

        # 初始化析取图邻接矩阵，包括了每个节点以及可选择的边
        conj_nei_up_stream = np.eye(self.num_of_tasks, k=-1, dtype=np.single)
        # conj_nei_low_stream = np.eye(self.num_of_tasks, k=1, dtype=np.single)

        conj_nei_up_stream[self.first_col] = 0  # 每个job的第一个task没有前驱节点
        # conj_nei_low_stream[self.last_col] = 0      # 每个job的最后的task没有后继节点

        self_as_nei = np.eye(self.num_of_tasks, dtype=np.single)  # 每个task可以指向自身，表示当前task不往前推进
        self.adj = conj_nei_up_stream + self_as_nei  # 后继和前驱是对称等价的，所以只用记录其中一个即可

        # 初始化节点的特征向量：task节点完成时间的下界, TODO 也许可以扩充特征的表示
        self.complete_time_lower_bound = np.cumsum(self.tasks_duration, axis=1,
                                                   dtype=np.single)  # 每个task完成时间的下界，即前置task加工时间之和
        self.init_quality = self.complete_time_lower_bound.max()
        self.max_end_time = self.init_quality
        # 节点特征 = [完成时间的下界, 是否完成状态]，后续也可以加上machine id，扩充节点状态、、、
        nodes_features = np.concatenate((self.adj,
                                         self.complete_time_lower_bound.reshape(-1, 1) / self.feature_normalize_factor,
                                         self.tasks_duration.reshape(-1, 1), self.finished_mark.reshape(-1, 1),),
                                        axis=1)

        self.task_start_time_on_machine = -self.high_bound * np.ones_like(self.tasks_duration.transpose(),
                                                                          dtype=np.int32)
        self.task_id_on_machine = -self.num_of_jobs * np.ones_like(self.tasks_duration.transpose(), dtype=np.int32)

        # return self.adj, nodes_features, self.candidate_task, self.mask      # TODO 需要修改，reset返回状态，也许只需要[adj, node_features]
        return nodes_features

    def step(self, action):
        """
        环境根据输入的动作，进行状态转移，输出下一个状态、奖励、是否完成、...
        :param action: agent提供的动作, 是从candidate_task中选择的一个task节点 0~num_of_tasks
        :return: 下一个状态、奖励、dones、...
        """
        if action in self.candidate_task and action not in self.partial_solution_seq:
            # 如果当前action是未选择过的action
            # 从action中获取到task在矩阵中的位置
            job_id = action // self.num_of_machines  # 该task(action)所在的job
            task_in_job = action % self.num_of_machines  # 该task在job中的顺序
            self.step_count += 1  # TODO step 计数应该移到外面来
            self.finished_mark[job_id, task_in_job] = 2  # 标识该task调度结束, 0：task未完成，且不可调度，1：task可调度，前置任务完成，2：task完成

            duration = self.tasks_duration[job_id, task_in_job]
            self.partial_solution_seq.append(action)

            # 更新状态
            start_time, flag = self.permissible_left_shift(action)
            self.flags.append(flag)

            if action not in self.last_col:
                self.candidate_task[action // self.num_of_machines] += 1  # 当前job的候选task为下一个task
                self.finished_mark[job_id, task_in_job + 1] = 1
            else:
                self.mask[action // self.num_of_machines] = 1  # 如果是最后一个task，则该job顺利完成

            self.duration_temp[job_id, task_in_job] = start_time + duration
            self.complete_time_lower_bound = self.get_end_time_lower_bound()

            # 更新图矩阵
            predecessor, successor = self.get_actions_neighborhoods(action)
            self.adj[action] = 0  # job上task的边
            self.adj[action, action] = 1
            if action not in self.first_col:
                self.adj[action, action - 1] = 1
            self.adj[action, predecessor] = 1  # machine上task的边
            self.adj[action, successor] = 1

            if flag and predecessor != action and successor != action:  # 插入调度，删除旧边
                self.adj[successor, predecessor] = 0

        nodes_features = np.concatenate((self.adj,
                                         self.complete_time_lower_bound.reshape(-1, 1) / self.feature_normalize_factor,
                                         self.tasks_duration.reshape(-1, 1), self.finished_mark.reshape(-1, 1)), axis=1)
        # TODO 奖励的设计
        reward = - (self.complete_time_lower_bound.max() - self.max_end_time)
        if reward == 0:  # 对时间没有产生任何影响，设置为预设值
            reward = self.reward_scale
            self.pos_rewards += reward

        self.max_end_time = self.complete_time_lower_bound.max()

        # return self.adj, nodes_features, reward, self.done(), self.candidate_task, self.mask
        return nodes_features, reward, self.done(), self.mask

    def done(self):
        if len(self.partial_solution_seq) == self.num_of_tasks:
            return True

        return False

    def get_end_time_lower_bound(self):
        x, y = last_nonzero(self.duration_temp, 1, invalid_val=-1)
        self.duration_copy[np.where(self.duration_temp != 0)] = 0

        self.duration_copy[x, y] = self.duration_temp[x, y]
        temp = np.cumsum(self.duration_copy, axis=1)
        temp[np.where(self.duration_temp != 0)] = 0
        ret = self.duration_temp + temp
        return ret

    def get_actions_neighborhoods(self, action):
        # 获取在machine上的邻居，主要是为了确定在同一个machine上各task的调度顺序
        action_pos = np.where(self.task_id_on_machine == action)
        predecessor = self.task_id_on_machine[
            action_pos[0], action_pos[1] - 1 if action_pos[1].item() > 0 else action_pos[1]].item()
        successor_temp = self.task_id_on_machine[
            action_pos[0], action_pos[1] + 1 if action_pos[1].item() + 1 < self.task_id_on_machine.shape[-1] else
            action_pos[1]].item()
        successor = action if successor_temp < 0 else successor_temp

        return predecessor, successor
        pass

    def permissible_left_shift(self, action):
        """

        :param action:
        :return:
        """
        machine_id = np.take(self.machine_order, action) - 1  # task所在的machine下标，从0开始
        job_ready_time, machine_ready_time = self.calculate_task_machine_ready_time(action,
                                                                                    machine_id)  # action对应的task在job中、在所在machine中就绪时间
        task_duration = np.take(self.tasks_duration, action)  # task的加工时间

        start_time_of_machine = self.task_start_time_on_machine[machine_id]  # machine上各task的开始时间
        task_on_machine = self.task_id_on_machine[machine_id]  # machine上各task的id
        flag = False

        possible_pos = np.where(job_ready_time < start_time_of_machine)[
            0]  # 如果当前task就绪时间可以在machine当前已分配task之前插入分配，则先尝试插入分配
        if len(possible_pos) == 0:
            start_time = self.put_to_end(action, job_ready_time, machine_ready_time, start_time_of_machine,
                                         task_on_machine)
        else:
            # 存在可以插入的位置，计算位置的合法性，即插入后不会影响已经分配的任务开始
            idx_legal_pos, legal_insert_pos, end_time_for_possible_pos = self.calculate_legal_pos(task_duration,
                                                                                                  job_ready_time,
                                                                                                  possible_pos,
                                                                                                  start_time_of_machine,
                                                                                                  task_on_machine)
            if len(legal_insert_pos) == 0:
                start_time = self.put_to_end(action, job_ready_time, machine_ready_time, start_time_of_machine,
                                             task_on_machine)
            else:
                flag = True
                start_time = self.insert_in_between(action, idx_legal_pos, legal_insert_pos, end_time_for_possible_pos,
                                                    start_time_of_machine, task_on_machine)

        return start_time, flag

    def calculate_task_machine_ready_time(self, action, machine_id):
        """

        :param action:
        :param machine_id:
        :return: task就绪时间，以及所在machine就绪时间
        """
        # machine_id = np.take(self.machine_order, action) - 1    # action对应task所在的machine下标，machine_order中从1开始，下标从0开始

        # 计算task就绪时间，同一个job中前驱task完成
        predecessor_task_in_job = action - 1 if action % self.machine_order.shape[
            1] != 0 else None  # task的前驱任务，第一个task无前驱任务
        if predecessor_task_in_job is not None:
            pre_task_duration = np.take(self.tasks_duration, predecessor_task_in_job)  # 前驱任务的执行时间
            pre_task_machine_id = np.take(self.machine_order, predecessor_task_in_job) - 1  # 前驱任务所在的machine下标
            # 当前task就绪的时间是前驱task就绪时间加上加工时间
            task_ready_time = (self.task_start_time_on_machine[pre_task_machine_id][np.where(
                self.task_id_on_machine[pre_task_machine_id] == predecessor_task_in_job)] + pre_task_duration).item()
        else:
            task_ready_time = 0

        # 计算machine就绪时间，同一个machine上分配前置的task完成
        predecessor_task_on_machine = \
            self.task_id_on_machine[machine_id][np.where(self.task_id_on_machine[machine_id] >= 0)][-1] if len(
                np.where(self.task_id_on_machine[machine_id] >= 0)[0]) != 0 else None
        if predecessor_task_on_machine is not None:
            task_duration_pre_machine = np.take(self.tasks_duration, predecessor_task_on_machine)
            machine_ready_time = (self.task_start_time_on_machine[machine_id][
                                      np.where(self.task_start_time_on_machine[machine_id] >= 0)][
                                      -1] + task_duration_pre_machine)
        else:
            machine_ready_time = 0

        return task_ready_time, machine_ready_time

    def calculate_legal_pos(self, duration, job_ready_time, possible_pos, start_time_on_machine, task_on_machine):
        """

        :param task_on_machine:
        :param start_time_on_machine:
        :param duration:
        :param job_ready_time:
        :param possible_pos:
        :return:
        """

        start_time_of_possible_pos = start_time_on_machine[possible_pos]  # 可以插入位置上task的开始时间
        task_duration_of_possible_pos = np.take(self.tasks_duration, task_on_machine[possible_pos])  # 可以插入位置上task的加工时间
        earliest_start = max(job_ready_time, start_time_on_machine[possible_pos[0] - 1] + np.take(self.tasks_duration, [
            task_on_machine[possible_pos[0] - 1]]))
        end_time_for_possible_pos = np.append(earliest_start,
                                              (start_time_of_possible_pos + task_duration_of_possible_pos))[:-1]
        possible_gap = start_time_of_possible_pos - end_time_for_possible_pos

        idx_legal_pos = np.where(duration <= possible_gap)[0]
        legal_pos = np.take(possible_pos, idx_legal_pos)
        return idx_legal_pos, legal_pos, end_time_for_possible_pos

    def put_to_end(self, action, job_ready_time, machine_ready_time, start_time_on_machine, task_on_machine):
        """

        :param action:
        :param job_ready_time:
        :param machine_ready_time:
        :param start_time_on_machine:
        :param task_on_machine:
        :return:
        """
        index = np.where(start_time_on_machine == -self.high_bound)[0][0]
        start_time = max(job_ready_time, machine_ready_time)
        start_time_on_machine[index] = start_time
        task_on_machine[index] = action

        return start_time

    def insert_in_between(self, action, idx_legal_pos, legal_pos, end_time_for_possible_pos, start_time_on_machine,
                          task_on_machine):
        """

        :param action:
        :param idx_legal_pos:
        :param legal_pos:
        :param end_time_for_possible_pos:
        :param start_time_on_machine:
        :param task_on_machine
        :return:
        """
        earliest_idx = idx_legal_pos[0]
        earliest_pos = legal_pos[0]

        start_time = end_time_for_possible_pos[earliest_idx]

        start_time_on_machine[:] = np.insert(start_time_on_machine, earliest_pos, start_time)[:-1]
        task_on_machine[:] = np.insert(task_on_machine, earliest_pos, action)[:-1]
        return start_time


def last_nonzero(arr, axis, invalid_val=-1):
    mask = arr != 0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    yAxis = np.where(mask.any(axis=axis), val, invalid_val)
    xAxis = np.arange(arr.shape[0], dtype=np.int64)
    xRet = xAxis[yAxis >= 0]
    yRet = yAxis[yAxis >= 0]
    return xRet, yRet


def permute_rows(x):
    '''
    x is a np array
    '''
    ix_i = np.tile(np.arange(x.shape[0]), (x.shape[1], 1)).T
    ix_j = np.random.sample(x.shape).argsort(axis=1)
    return x[ix_i, ix_j]


def uni_instance_gen(n_j, n_m, low, high):
    times = np.random.randint(low=low, high=high, size=(n_j, n_m))
    machines = np.expand_dims(np.arange(1, n_m + 1), axis=0).repeat(repeats=n_j, axis=0)
    machines = permute_rows(machines)
    return times, machines


#  issac gym 并行环境版本


"""动态JSSP环境
在静态JSSP基础上，实例的状态(运行时间)具有随机性
"""

"""柔性JSSP环境"""

if __name__ == "__main__":
    # from util.uniform_instance_gen import uni_instance_gen

    # instance = np.array(uni_instance_gen(3, 4, 1, 99))
    env = SJsspEnv(15, 15)
    ta01 = np.load("../Instances/Benchmark/ta/J15_M15.npy")[0]
    state = env.reset(ta01)

    order = [30, 135, 90, 195, 75, 105, 165, 196, 60, 106, 120, 150, 136, 166, 107, 91, 0, 61, 92, 93, 137, 121, 62,
             197, 210, 76, 45, 211, 108, 1, 167, 2, 138, 151, 63, 46, 94, 198, 152, 122, 180, 139, 153, 109, 47, 168,
             64, 199, 140, 3, 110, 48, 65, 4, 212, 181, 123, 15, 169, 49, 170, 124, 141, 154, 213, 16, 66, 77, 31, 5,
             214, 171, 32, 111, 125, 50, 142, 182, 78, 215, 183, 172, 200, 155, 79, 173, 184, 80, 17, 51, 143, 112, 201,
             95, 67, 126, 96, 113, 33, 185, 174, 6, 216, 81, 97, 202, 144, 52, 114, 186, 115, 127, 68, 53, 18, 156, 175,
             145, 7, 187, 203, 116, 157, 117, 34, 98, 69, 19, 8, 176, 82, 54, 146, 204, 35, 118, 188, 119, 158, 36, 55,
             37, 177, 20, 21, 9, 159, 83, 22, 70, 128, 84, 189, 38, 56, 178, 71, 99, 217, 57, 218, 219, 190, 160, 85,
             23, 220, 129, 10, 39, 100, 147, 24, 86, 191, 161, 130, 221, 179, 162, 148, 11, 40, 25, 101, 131, 41, 163,
             102, 72, 26, 192, 205, 42, 222, 27, 58, 73, 12, 206, 132, 43, 149, 223, 103, 164, 193, 13, 44, 87, 28, 88,
             74, 207, 194, 224, 133, 208, 59, 89, 134, 104, 29, 209, 14]
    reward = 0
    end_time = []
    for o in order:
        reward += env.step(o)[1]
        end_time.append(env.max_end_time)

    print("reward: " + str(reward))
    print("span time: " + str(env.init_quality - reward - env.pos_rewards))
    print(end_time)
    # env.step(45)
    # env.step(150)
    # env.step(105)
    # env.step(210)
    # env.step(90)
    # env.step(120)
    # env.step(180)
    # env.step(211)
    # env.step(75)
    # env.step(121)
    # env.step(135)
    # env.step(165)
    # env.step(151)
    # env.step(181)
    # env.step(122)
    # env.step(106)
    # env.step(0)
    # env.step(76)
    # env.step(107)
    # env.step(108)
    # env.step(152)
    # env.step(136)
    # env.step(212)
    # env.step(225)

    print("Done")

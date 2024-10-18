import numpy as np
from Env.SJSSPEnv import SJsspEnv, uni_instance_gen


class DJsspEnv(SJsspEnv):
    def __init__(self, job_num, machine_num, sigma, lamb, theta, args=None):
        super(DJsspEnv, self).__init__(job_num, machine_num, args)
        self.sigma = sigma
        self.lamb = lamb
        self.theta = theta

        self.conf_matrix = np.ones((job_num, machine_num))
        for m in range(machine_num):
            self.conf_matrix[:, m] = np.clip(lamb**(m+1), a_min=self.theta, a_max=1.0)

        self.bias_ub = None
        self.fixed_bias = None
        self.conf_duration = None
        # self.bias = sigma * self.tasks_duration
        #
        # self.conf_duration = self.tasks_duration + (1-self.conf_matrix) * self.bias

    def reset(self, instance=None, bias=None):
        """
        随机生成一个JSP实例，使用JSP实例初始化环境，instance = [task加工时间，task访问的机器]    [n_j X n_m X 2]
        :return: init state: 图的邻接矩阵 n_t * n_t，节点的特征向量 n_t * f
        """
        if instance is None:
            instance = np.array(uni_instance_gen(self.num_of_jobs, self.num_of_machines, self.low_bound, self.high_bound))
        assert instance.shape == (2, self.num_of_jobs,
                                  self.num_of_machines), f"Illegal Instance Shape. Instance Shape Should be {(2, self.num_of_jobs, self.num_of_machines)}"
        self.step_count = 0
        self.machine_order = np.copy(instance[1])
        self.tasks_duration = np.copy(instance[0])

        # dynamic add
        self.fixed_bias = bias

        self.bias_ub = self.sigma * self.tasks_duration

        self.conf_duration = self.tasks_duration + (1 - self.conf_matrix) * self.bias_ub

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
                                         self.conf_duration.reshape(-1, 1), self.finished_mark.reshape(-1, 1),),
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
                                         self.conf_duration.reshape(-1, 1), self.finished_mark.reshape(-1, 1)), axis=1)
        # TODO 奖励的设计
        reward = - (self.complete_time_lower_bound.max() - self.max_end_time)
        if reward == 0:  # 对时间没有产生任何影响，设置为预设值
            reward = self.reward_scale
            self.pos_rewards += reward

        self.max_end_time = self.complete_time_lower_bound.max()

        # return self.adj, nodes_features, reward, self.done(), self.candidate_task, self.mask
        return nodes_features, reward, self.done(), self.mask

    def permissible_left_shift(self, action):
        """

        :param action:
        :return:
        """
        machine_id = np.take(self.machine_order, action) - 1  # task所在的machine下标，从0开始

        job_ready_time, machine_ready_time = self.calculate_task_machine_ready_time(action,
                                                                                    machine_id)  # action对应的task在job中、在所在machine中就绪时间

        # dynamic add
        job_id = int(action / self.num_of_machines)
        task_id_in_job = action % self.num_of_machines
        task_duration = np.take(self.tasks_duration, action)  # task的加工时间 TODO dynamic 更改

        if self.fixed_bias is None:
            task_duration += np.random.randint(0, max(1, self.sigma*task_duration))
        else:
            task_duration += np.take(self.fixed_bias, action)

        self.tasks_duration[job_id, task_id_in_job] = task_duration

        self.conf_matrix[job_id] = np.clip(self.conf_matrix[job_id] / self.lamb, a_min=self.theta, a_max=1.0)

        self.conf_duration = self.tasks_duration + (1 - self.conf_matrix) * self.bias_ub
        # end
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


def get_bias_matrix(instances, sigma=0.1):
    np.random.seed(114514)
    # 依据benchmark instance生成浮动矩阵
    _, _, job_num, machine_num = instances.shape
    dynamic_instances = []
    bias_instances = []
    for instance in instances:
        # job_num, machine_num = instance.shape
        time = instance[0]
        order = instance[1]

        bias = np.zeros((job_num, machine_num), dtype=np.int32)

        for i in range(job_num):
            for j in range(machine_num):
                h = max(1, sigma*time[i, j])
                bias[i, j] = np.random.randint(0, h)

        new_time = time + bias
        d_instance = [new_time, order]
        dynamic_instances.append(d_instance)
        bias_instances.append(bias)

    return np.array(dynamic_instances), np.array(bias_instances)


def gen_dynamic():
    sizes = {"dmu": [(20, 15), (20, 20), (30, 15), (30, 20), (40, 15), (40, 20), (50, 15), (50, 20)],
             "ta": [(15, 15), (20, 15), (20, 20), (30, 15), (30, 20), (50, 15), (50, 20), (100, 20)]}

    set_name = ""

    for set_name in ["ta", "dmu"]:
        for size in sizes[set_name]:
            job_num = size[0]
            machine_num = size[1]

            name = f"J{job_num}_M{machine_num}"
            instances = np.load(f"../Instances/Benchmark/{set_name}/{name}.npy")
            sigma = 20  # 1
            dynamic_instances, bias = get_bias_matrix(instances, sigma / 100.0)

            np.save(f"../Instances/Benchmark/{set_name}/dynamic/{sigma}/{name}.npy", dynamic_instances)  # 2
            np.save(f"../Instances/Benchmark/{set_name}/bias/{sigma}/{name}.npy", bias)  # 3

            print(set_name + " " + name + " done.")


if __name__ == "__main__":
    env = DJsspEnv(20, 15, 0.1, 0.9, 0.45)
    node_feature = env.reset()

    env.step(0)
    env.step(15)
    env.step(16)
    env.step(30)
    pass

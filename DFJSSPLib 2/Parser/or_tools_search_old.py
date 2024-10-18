import os
import numpy as np
import collections
from ortools.sat.python import cp_model


def taillard2jobs(instance, job_num, machine_num):
    time_matrix = instance[0]
    machine_matrix = instance[1]

    jobs = []
    for i in range(job_num):
        tasks = []
        for j in range(machine_num):
            tasks.append((machine_matrix[i][j] - 1, time_matrix[i][j]))

        jobs.append(tasks)

    return jobs


def ortools_search(jobs, job_num, machine_num):
    all_machines = range(machine_num)
    horizon = sum(task[1] for job in jobs for task in job)

    model = cp_model.CpModel()

    # Named tuple to store information about created variables.
    task_type = collections.namedtuple('task_type', 'start end interval')
    # Named tuple to manipulate solution information.
    assigned_task_type = collections.namedtuple('assigned_task_type',
                                                'start job index duration')

    # Creates job intervals and add to the corresponding machine lists.
    all_tasks = {}
    machine_to_intervals = collections.defaultdict(list)

    for job_id, job in enumerate(jobs):
        for task_id, task in enumerate(job):
            machine = task[0]
            duration = task[1]
            suffix = '_%i_%i' % (job_id, task_id)
            start_var = model.NewIntVar(0, horizon, 'start' + suffix)
            end_var = model.NewIntVar(0, horizon, 'end' + suffix)
            interval_var = model.NewIntervalVar(start_var, duration, end_var,
                                                'interval' + suffix)
            all_tasks[job_id, task_id] = task_type(start=start_var,
                                                   end=end_var,
                                                   interval=interval_var)
            machine_to_intervals[machine].append(interval_var)

    # Create and add disjunctive constraints.
    for machine in all_machines:
        model.AddNoOverlap(machine_to_intervals[machine])

    # Precedences inside a job.
    for job_id, job in enumerate(jobs):
        for task_id in range(len(job) - 1):
            model.Add(all_tasks[job_id, task_id + 1].start >= all_tasks[job_id, task_id].end)

    # Makespan objective.
    obj_var = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(obj_var, [
        all_tasks[job_id, len(job) - 1].end
        for job_id, job in enumerate(jobs)
    ])
    model.Minimize(obj_var)

    # 调用求解器
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 1800.0  # 求解器30min = 1800s的求解限制
    status = solver.Solve(model)

    # 打印解
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print('Solution:')
        # Create one list of assigned tasks per machine.
        assigned_jobs = collections.defaultdict(list)
        for job_id, job in enumerate(jobs):
            for task_id, task in enumerate(job):
                machine = task[0]
                assigned_jobs[machine].append(
                    assigned_task_type(start=solver.Value(
                        all_tasks[job_id, task_id].start),
                        job=job_id,
                        index=task_id,
                        duration=task[1]))
        # 可以依据assigned task的start time排序获得task seq
        print(f'Span Time: {solver.ObjectiveValue()}')

        tasks_seq = []
        tasks = sum(assigned_jobs.values(), [])
        tasks.sort(key=lambda x: x.start)
        for t in tasks:
            task_id = t.job * machine_num + t.index
            tasks_seq.append(task_id)

        return tasks_seq
        # Create per machine output lines.
        # output = ''
        # for machine in all_machines:
        #     # Sort by starting time.
        #     assigned_jobs[machine].sort()
        #     sol_line_tasks = 'Machine ' + str(machine) + ': '
        #     sol_line = '           '
        #
        #     for assigned_task in assigned_jobs[machine]:
        #         name = 'job_%i_task_%i' % (assigned_task.job,
        #                                    assigned_task.index)
        #         # Add spaces to output to align columns.
        #         sol_line_tasks += '%-15s' % name
        #
        #         start = assigned_task.start
        #         duration = assigned_task.duration
        #         sol_tmp = '[%i,%i]' % (start, start + duration)
        #         # Add spaces to output to align columns.
        #         sol_line += '%-15s' % sol_tmp
        #
        #     sol_line += '\n'
        #     sol_line_tasks += '\n'
        #     output += sol_line_tasks
        #     output += sol_line

        # Finally print the solution found.

        # print(output)
    else:
        print('No solution found.')
        return None


# 筛选
# opt_path = '../Instances/Optimals'
ins_path = '../Instances/OrlibNpy'
#
# all_instances = os.listdir(ins_path)
# opt_files = os.listdir(opt_path)
#
# opt_instances = [fn.split('_')[0] for fn in opt_files]
#
# search_instances = [ins for ins in all_instances if ins not in opt_instances and 'dmu' not in ins and 'ta' not in ins]  # 删去有最优解的实例以及dmu ta作为测试样例

search_instances = np.load('../Instances/search_list.npy')

for name in search_instances:
    dir = ins_path + '/' + name

    instance = np.load(dir+'/'+name+'.npy')
    size = np.load(dir+'/size.npy')

    print(name)
    print(size)
    # job_num = size[0]
    # machine_num = size[1]
    #
    # jobs_data = taillard2jobs(instance, job_num, machine_num)
    #
    # actions = []
    # for _ in range(2):
    #     task_seq = ortools_search(jobs_data, job_num, machine_num)
    #     actions.append(task_seq)

    # np.save(dir + '/actions.npy', actions)



# ortools_search(jobs_data, 3)







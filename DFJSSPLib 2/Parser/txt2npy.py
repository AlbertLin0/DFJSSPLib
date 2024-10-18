import numpy as np
import os

txt_path = "../Instances/Txt"
npy_path = "../Instances/OrlibNpy"

files = os.listdir(txt_path)

for f in files:
    name = f.split('.')[0]
    instance_path = npy_path + '/' + name
    if not os.path.exists(instance_path):
        os.makedirs(instance_path)

    file_path = txt_path + '/' + f
    with open(file_path, 'r') as txt_instance:
        size = txt_instance.readline().split()

        job_num = int(size[0])
        machine_num = int(size[1])
        # np.save(instance_path+'/'+'size', (job_num, machine_num))
        time_matrix = np.zeros([job_num, machine_num], dtype=np.int32)
        order_matrix = np.zeros([job_num, machine_num], dtype=np.int32)

        lines = txt_instance.readlines()
        times = lines[:job_num]
        orders = lines[job_num:]

        for i, job_time in enumerate(times):
            t = list(map(np.int32, job_time.split()))
            time_matrix[i] = t

        for i, job_order in enumerate(orders):
            o = list(map(np.int32, job_order.split()))
            order_matrix[i] = o

        data = (time_matrix, order_matrix)
        np.save(instance_path+'/'+name+'.npy', data)

        # load_data = np.load(instance_path+'/'+name+'.npy')[1]

    print(name + ' Done.')


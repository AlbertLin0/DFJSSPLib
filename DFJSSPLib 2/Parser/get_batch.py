import numpy as np
import pandas as pd

dataset_info = pd.read_csv('../Instances/dataset_info', sep='\t')

path = '../Instances/OrlibNpy'
batch_path = '../Instances/Batch'

dataset_info.sort_values(['job_num', 'machine_num'], inplace=True)

info_list = dataset_info.values.tolist()

size = (info_list[0][1], info_list[0][2])
batch_data = []
batch_actions = []
info = {}
num = 0

for instance in info_list:
    name = instance[0]
    cur_size = (instance[1], instance[2])

    # data = np.load(path+'/'+name+'/data.npy')
    # data = data.reshape(-1, data.shape[-2], data.shape[-1])
    # actions = np.load(path+'/'+name+'/actions.npy')
    # actions = actions.reshape(-1)
    if size == cur_size:
        # batch_data.extend(data)
        # batch_actions.extend(actions)
        num += instance[3]
    else:
        # batch_name = str(size[0]) + '_' + str(size[1])
        # np.save(batch_path + '/' + batch_name+'_data.npy', batch_data)
        # batch_data = []
        # np.save(batch_path + '/' + batch_name+'_actions.npy', batch_actions)
        # batch_actions = []
        info[size] = num
        num = 0

        size = cur_size
        num += instance[3]
        # batch_data.extend(data)
        # batch_actions.extend(actions)

print(info)






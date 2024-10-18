import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
import torch
import torch.nn as nn
from Env.SJSSPEnv import SJsspEnv

from RL.baseModel import ActorGAT
# 环境安装
# dataset、dataloader实现

device = torch.device("cuda:0")


class BatchDataset(Dataset):
    # 每次采样相同规模的(step, task)
    def __init__(self, dir_path, size):
        # instance数据路径列表，分size存放
        j_num = int(size.split('_')[0])
        m_num = int(size.split('_')[1])
        self.task_num = j_num * m_num * 1.0
        self.steps = np.load(dir_path + '/' + size + '_data.npy')
        self.tasks = np.load(dir_path + '/' + size + '_actions.npy').reshape(-1, 1)

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        # 将label转化到0-10之间
        return self.steps[idx], self.tasks[idx]


class InstanceDataset(Dataset):
    # 每次采样一个instance
    def __init__(self):
        # 将所有instances的解构建成列表
        self.path = '../Instances/OrlibNpy/'
        dataset_info = pd.read_csv('../Instances/filter_info', sep='\t').values.tolist()
        self.instance_list = []
        for item in dataset_info:
            self.instance_list.extend([(item[0], item[1], item[2], i) for i in range(item[3])])

    def __len__(self):
        return len(self.instance_list)

    def __getitem__(self, idx):
        instance = self.instance_list[idx]
        name = instance[0]
        i = instance[3]
        states = np.load(self.path+name+'/data.npy')
        actions = np.load(self.path+name+'/actions.npy')
        return states[i], actions[i]


class BatchModel(nn.Module):
    # actorgat
    # 每次将同规模的step-action数据
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class InstanceModel():
    pass


class FusionModel():
    pass


def batch_train(LR=0.1):
    EPOCHES = 1000
    # BATCH_SIZE = 4
    BATCH_SIZE = 64
    DIR_PATH = "../Instances/Batch"
    #LR = 0.1
    # LR = 2e-4

    # ['6_6', '10_5', '10_10', '15_5', '15_10', '15_15', '20_5', '20_10', '30_10', '50_10']
    # ['6_6', '10_5', '15_5', '15_10', '15_15', '20_10']
    data_sizes = ['20_5']

    datasets = [BatchDataset(DIR_PATH, size) for size in data_sizes]
    loaders = [DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True) for dataset in datasets]

    eval_dataset = BatchDataset(DIR_PATH, '20_10')
    eval_loaders = DataLoader(dataset=eval_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)

    gat_args = {
        "n_layers": 4,
        "n_head_per_layers": [8, 16, 16, 1],
        "in_fea_dim": 3,
        "out_fea_dim": 64,
        "n_features_per_layers": [3, 64, 256, 256, 64],
        "jobs_num": 0,
        "tasks_num": 0
    }

    model = ActorGAT(device, gat_args).to(device)
    cross_entropy = nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), LR, weight_decay=0.002)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2500, gamma=0.5)
    train_loss_recorder = []
    eval_recorder = []
    min_eval_loss = np.inf
    for i in range(EPOCHES):
        loss_sum = 0.0
        count = 0
        model.train()
        for loader in loaders:
            # node_num = loader.dataset.task_num
            for states, actions in loader:

                optimizer.zero_grad()

                states = states.to(device, non_blocking=True)
                states.requires_grad = True
                actions = actions.to(device, non_blocking=True, dtype=torch.long).squeeze(-1)
                # actions = actions.type(torch.float32)
                # cur_size = actions.shape[0]
                #
                # loss = 0.0
                candidates_prob = model(states)

                # for j in range(cur_size):
                #     cand = candidates[j].detach().cpu().numpy()
                #     label = torch.tensor(np.argwhere(cand == actions[j].item()), device=device).squeeze(0)
                #     loss += cross_entropy(candidates_prob[j].unsqueeze(0), label)

                # nn.CrossEntropyLoss()
                # preds = preds.reshape(-1, 1)

                # preds = preds.type(torch.float32)

                # loss = mse(candidates, actions)

                # loss.requires_grad = True

                # print("====Before Update=====")
                # for name, param in model.named_parameters():
                #     print("---->Name:", name)
                #     print("---->Para:", param)
                #     print("---->grad_require:", param.requires_grad)
                #     print("---->grad_values:", param.grad)
                loss = cross_entropy(candidates_prob, actions)

                loss.backward()
                # clip_grad_norm_(optimizer.param_groups[0]["params"], max_norm=3.0)
                optimizer.step()

                # print("====After Update=====")
                # for name, param in model.named_parameters():
                #     print("---->Name:", name)
                #     print("---->Para:", param)
                #     print("---->grad_require:", param.requires_grad)
                #     print("---->grad_values:", param.grad)

                # print(optimizer)

                torch.cuda.synchronize()
                scheduler.step()

                loss_sum += loss.item()
                count += 1

        train_loss_recorder.append(loss_sum / count)
        # print(f"Epoch {i+1}: Avg CrossEntropy Loss {loss_sum / count}")

        if (i+1) % 10 == 0:
            model.eval()

            with torch.no_grad():
                eval_loss_sum = 0.0
                eval_count = 0.0
                for states, actions in eval_loaders:
                    states = states.to(device, non_blocking=True)
                    states.requires_grad = True
                    actions = actions.to(device, non_blocking=True, dtype=torch.long).squeeze(-1)

                    prob = model(states)
                    loss = cross_entropy(prob, actions)

                    eval_loss_sum += loss.item()
                    eval_count += 1

            mean_loss = eval_loss_sum / eval_count
            if mean_loss < min_eval_loss:
                torch.save(model.state_dict(), f'../TrainLog/learning_rate_{LR}_actor_epoch_{i}_evalloss_{mean_loss}.pth')
                min_eval_loss = mean_loss
                print(min_eval_loss)
            # print(f"Validate Mean CrossEntropy Loss {mean_loss}")
            eval_recorder.append([mean_loss])

    np.save(f'../TrainLog/train_loss_learning_rate_{LR}.npy', train_loss_recorder)
    np.save(f'../TrainLog//eval_result_{LR}.npy', eval_recorder)
    torch.save(model.state_dict(), f'../learning_rate_{LR}_actor.pth')

def instance_train():
    EPOCHES = 10000
    # BATCH_SIZE = 4
    BATCH_SIZE = 1
    LR = 0.2

    dataset = InstanceDataset()
    loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    gat_args = {
        "n_layers": 4,
        "n_head_per_layers": [8, 16, 16, 1],
        "in_fea_dim": 3,
        "out_fea_dim": 64,
        "n_features_per_layers": [3, 64, 256, 256, 64],
        "jobs_num": 0,
        "tasks_num": 0
    }

    model = ActorGAT(device, gat_args).to(device)
    mse = nn.MSELoss(reduction='mean')
    cross_entropy = nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), LR, weight_decay=0.002)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2500, gamma=0.5)

    model.train()
    for i in range(EPOCHES):
        loss_sum = 0.0
        count = 0
        for states, actions in loader:

            optimizer.zero_grad()

            states = states.squeeze(0).to(device)
            actions = actions.squeeze(0).to(device)

            cur_size = actions.shape[0]

            loss = 0.0
            candidates_prob, candidates = model(states)

            for j in range(cur_size):
                cand = candidates[j].detach().cpu().numpy()
                label = torch.tensor(np.argwhere(cand == actions[j].item()), device=device).squeeze(0)
                loss += cross_entropy(candidates_prob[j].unsqueeze(0), label)

            # preds, _, _ = model.get_action(states)
            # preds = preds.type(torch.float32)
            # # preds = []
            # # for s in states[0]:
            # #     pred, _, _ = model.get_action(s)
            # #     preds.append(pred)
            # #
            # # preds = torch.Tensor(pred).to(device)
            # loss = mse(preds, actions)
            # loss_sum += loss.item()
            # count += 1

            loss /= cur_size

            loss.backward()
            clip_grad_norm_(optimizer.param_groups[0]["params"], max_norm=3.0)
            optimizer.step()
            torch.cuda.synchronize()
            scheduler.step()

            loss_sum += loss.item()
            count += 1

        print(f"Epoch {i}: Avg MSELoss {loss_sum/count}")


if __name__ == "__main__":
    LRs = [0.05, 0.025]
    # instance_train()
    for LR in LRs:
        batch_train(LR)

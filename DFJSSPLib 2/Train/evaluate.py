import os, time
import torch
import numpy as np
from Train.config import Arguments


class Evaluator:
    def __init__(self, cwd, agent_id, eval_env, args):
        self.recorder = []
        self.recorder_path = f"{cwd}/recorder.npy"

        self.cwd = cwd
        self.agent_id = agent_id
        self.eval_env = eval_env
        self.eval_gap = args.eval_gap
        self.eval_time1 = max(1, int(args.eval_times / np.e))
        self.eval_time2 = max(0, int(args.eval_times - self.eval_time1))
        self.target_return = args.target_return

        self.r_max = -np.inf
        self.eval_time = 0
        self.used_time = 0
        self.total_step = 0
        self.epoch = 0
        self.start_time = time.time()

        print(f"{'#' * 80}\n"
              f"{'ID':<3}{'Epoch':>10}{'Step':>10}{'maxR':>12} |"
              f"{'avgR':>10}{'stdR':>7}{'avgS':>7}{'stdS':>6} |"
              f"{'expR':>10}{'e_r_avg':>10}{'objC':>10}{'objA':>10}{'etc.':>10}")

    def evaluate_save_and_plot(self, act, steps, r_exp, explore_reward_avg, log_tuple):
        self.total_step += steps
        self.epoch += 1

        if time.time() - self.eval_time < self.eval_gap:
            if_reach_goal = False
            if_save = False
        else:
            self.eval_time = time.time()

            '''第一次验证'''
            rewards_steps_list = [get_cumulative_returns_and_step(self.eval_env, act) for _ in range(self.eval_time1)]
            rewards_steps_ary = np.array(rewards_steps_list, dtype=np.float32)

            r_avg, s_avg = rewards_steps_ary.mean(axis=0)  # 一个episode的平均reward和平均步数

            '''第二次验证'''
            if r_avg > self.r_max:
                rewards_steps_list.extend(
                    [get_cumulative_returns_and_step(self.eval_env, act) for _ in range(self.eval_time2)])
                rewards_steps_ary = np.array(rewards_steps_list, dtype=np.float32)
                r_avg, s_avg = rewards_steps_ary.mean(axis=0)

            r_std, s_std = rewards_steps_ary.std(axis=0)  # 一个episode的reward标准差和步数标准差

            if_save = r_avg > self.r_max  # 如果平均奖励更大则保存网络
            if if_save:
                self.r_max = r_avg

                act_path = f"{self.cwd}/actor_{self.total_step:012}_{self.r_max:09.3f}.pth"
                torch.save(act.state_dict(), act_path)  # 将策略网络即actor保存在act_path

                print(f"{self.agent_id:<3}{self.total_step:20.2e}{self.r_max:10.2f} |")

            self.recorder.append((self.total_step, r_avg, r_std, r_exp, explore_reward_avg, *log_tuple))

            if_reach_goal = bool(self.r_max > self.target_return)
            if if_reach_goal and self.used_time is None:
                self.used_time = int(time.time() - self.start_time)
                print(f"{'ID':<3}{'Epoch':>10}{'Step':>10}{'TargetR':>10} |"
                      f"{'avgR':>10}{'stdR':>7}{'avgS':>7}{'stdS':>6} |"
                      f"{'UsedTime':>8} #######\n"
                      f"{self.agent_id:<3}{self.epoch:10d}{self.total_step:10.2e}{self.r_max:10.2f} |"
                      f"{r_avg:10.2f}{r_std:7.1f}{s_avg:7.0f}{s_std:6.0f} |"
                      f"{self.used_time:>8} ######")

            print(f"{self.agent_id:<3}{self.epoch:10.1f}{self.total_step:10.2e}{self.r_max:12.2f} |"
                  f"{r_avg:10.2f}{r_std:7.1f}{s_avg:7.0f}{s_std:6.0f} |"
                  f"{r_exp:10.2f}{explore_reward_avg:10.2f}{''.join(f'{n:10.4f}' for n in log_tuple)}")

            # if hasattr(self.eval_env, 'curriculum_learning_for_evaluator'):
            #     self.eval_env.curriculum_learning_for_evaluator(r_avg)

            '''plot learning curve'''
            if len(self.recorder) == 0:
                print("| save_npy_draw_plot() WARNING: len(self.recorder) == 0")
                return None

            np.save(self.recorder_path, self.recorder)

            train_time = int(time.time() - self.start_time)
            total_step = int(self.recorder[-1][0])
            save_title = f"Step_time_maxR_{int(total_step)}_{int(train_time)}_{self.r_max:.3f}"
            save_learning_curve(self.recorder, self.cwd, save_title)
        return if_reach_goal, if_save

    def save_or_load_recoder(self, if_save):
        if if_save:
            np.save(self.recorder_path, self.recorder)
        elif os.path.exists(self.recorder_path):
            recorder = np.load(self.recorder_path)
            self.recorder = [tuple(i) for i in recorder]
            self.total_step = self.recorder[-1][0]


def get_cumulative_returns_and_step(env, act):
    """

    :param env:
    :param act:
    :return:
    """
    max_step = env.max_step
    if_discrete = env.if_discrete
    device = next(act.parameters()).device
    # print("############")
    state = env.reset()
    steps = None
    returns = 0.0  # sum rewards in an episode
    for steps in range(max_step):
        tensor_state = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        # tensor_action = act(tensor_state).argmax(dim=1) if if_discrete else act(tensor_state)
        tensor_action, _, _ = act.get_action(tensor_state)
        action = tensor_action.detach().cpu().numpy().item()
        # print(action)
        state, reward, done, _ = env.step(action)

        returns += reward
        if done:
            break

    returns = getattr(env, 'cumulative_returns', returns)
    steps += 1
    return returns, steps


def save_learning_curve(recorder, path, save_title):
    # TODO 需要实现
    pass

import numpy as np
import torch

from RL.Agent import Agent
from RL.baseModel import ActorPPO, CriticPPO, ActorGAT, ActorGATBatch, CriticGAT


class AgentPPO(Agent):
    """
    Proximal Policy Optimization Algorithm
    """
    def __init__(self, net_dim, state_dim, action_dim, gpu_id, args):
        self.if_off_policy = False
        self.act_class = getattr(self, "act_class", ActorGAT)
        self.cri_class = getattr(self, "cri_class", CriticGAT)
        self.if_cri_target = getattr(args, "if_cri_target", False)
        Agent.__init__(self, net_dim, state_dim, action_dim, gpu_id, args)

        self.ratio_clip = getattr(args, "ratio_clip", 0.2)             # 0.0 ~ 0.5
        self.lambda_entropy = getattr(args, "lambda_entropy", 0.005)     # 0.00 ~ 0.10
        self.lambda_gae_adv = getattr(args, "lambda_gae_adv", 0.98)     # 0.95 ~ 0.99

        if getattr(args, "if_use_gae", False):
            self.get_reward_sum = self.get_reward_sum_gae
        else:
            self.get_reward_sum = self.get_reward_sum_raw

    def explore_one_env(self, env, target_step):
        states = torch.zeros((target_step, *self.state_dim), dtype=torch.float32).to(self.device)
        actions = torch.zeros((target_step, self.action_dim), dtype=torch.float32).to(self.device)
        log_probs = torch.zeros(target_step, dtype=torch.float32).to(self.device)
        actions_id = torch.zeros(target_step, dtype=torch.float32).to(self.device)
        rewards = torch.zeros(target_step, dtype=torch.float32).to(self.device)
        dones = torch.zeros(target_step, dtype=torch.bool).to(self.device)
        # cumulative_rewards = []
        # c_r = 0
        ary_state = self.states[0]

        get_action = self.actor.get_action
        convert = self.actor.convert_action_for_env
        for i in range(target_step):
            state = torch.as_tensor(ary_state, dtype=torch.float32, device=self.device)
            action, log_prob, action_id = [t.squeeze() for t in get_action(state.unsqueeze(0))]     # changed, 删除了unsqueeze
            ary_action = convert(action).detach().cpu().numpy().item()
            ary_state, reward, done, _ = env.step(ary_action)

            # c_r += reward
            if done:
                ary_state = env.reset()
                # cumulative_rewards.append(c_r)
                # c_r = 0

            states[i] = state
            actions[i] = action
            log_probs[i] = log_prob
            actions_id[i] = action_id
            rewards[i] = reward
            dones[i] = done

        self.states[0] = ary_state
        rewards = (rewards * self.reward_scale).unsqueeze(1)
        undones = (1 - dones.type(torch.float32)).unsqueeze(1)
        return states, actions, log_probs, rewards, undones, actions_id

    def explore_vec_env(self, env, target_step, random_exploration=None):
        traj_list = []
        last_dones = torch.zeros(self.env_num, dtype=torch.int, device=self.device)
        state = self.states

        step_i = 0
        dones = torch.zeros(self.env_num, dtype=torch.int, device=self.device)
        get_action = self.actor.get_action
        get_a_to_e = self.actor.convert_action_for_env
        while step_i < target_step or not any(dones):
            action, log_prob = get_action(state)
            next_state, rewards, dones, _ = env.step(get_a_to_e(action))

            traj_list.append(
                (state.clone(), rewards.clone(), dones.clone(), action, log_prob)
            )

            step_i += 1
            last_dones[torch.where(dones)[0]] = step_i
            state = next_state

        self.states = state
        return self.convert_trajectory(traj_list, last_dones)

    def update_net(self, buffer):
        """
        从经验池中采样数据，更新网络
        :param buffer: 经验池，其中保存了探索的路径(state, reward, dones, actions, log_probs)
        :return: 平均loss：critic avg loss, actor avg loss, action 标准差
        """
        with torch.no_grad():
            states, actions, log_probs, rewards, undones, actions_id = buffer
            buffer_size = states.shape[0]

            '''计算动作的优势'''
            batch_size = self.batch_size
            values = [self.critic(states[i: i+batch_size]) for i in range(0, buffer_size, batch_size)]
            values = torch.cat(values, dim=0).squeeze(1)
            advantages = self.get_advantages(rewards, undones, values)
            reward_sums = advantages * values

            del rewards, undones, values

            advantages = (advantages - advantages.mean()) / (advantages.std(dim=0) + 1e-5)
        assert log_probs.shape[0] == advantages.shape[0] == reward_sums.shape[0] == buffer_size

        '''开始更新网络参数'''
        obj_critics = 0.0
        obj_actors = 0.0

        update_times = int(buffer_size * self.repeat_times / self.batch_size)
        assert update_times >= 1
        for _ in range(update_times):
            indices = torch.randint(buffer_size, size=(self.batch_size, ), requires_grad=False)
            state = states[indices]
            action = actions[indices]
            action_id = actions_id[indices]
            log_prob = log_probs[indices]
            advantage = advantages[indices]
            reward_sum = reward_sums[indices]

            value = self.critic(state).squeeze(1)
            obj_critic = self.criterion(value, reward_sum)
            self.optimizer_update_amp(self.cri_optimizer, self.cri_optim_scheduler, obj_critic)
            # if self.if_cri_target:
            #     self.soft_update(self.cri_target, self.critic, self.soft_update_tau)

            new_log_prob, obj_entropy = self.actor.get_logprob_entropy(state, action_id)
            ratio = (new_log_prob - log_prob.detach()).exp()
            surrogate1 = advantage * ratio
            surrogate2 = advantage * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = torch.min(surrogate1, surrogate2).mean()

            obj_actor = obj_surrogate + obj_entropy.mean() * self.lambda_entropy
            self.optimizer_update_amp(self.act_optimizer, self.act_optim_scheduler, -obj_actor)

            obj_critics += obj_critic.item()
            obj_actors += obj_actor.item()

        action_std = getattr(self.actor, 'action_std_log', torch.zeros(1)).exp().mean()
        return obj_critics / update_times, obj_actors / update_times, action_std.item()

    def get_reward_sum_raw(self):
        pass

    def get_reward_sum_gae(self):
        pass

    def get_advantages(self, rewards, undones, values):
        advantages = torch.empty_like(values)
        masks = undones * self.gamma
        target_step = rewards.shape[0]

        next_state = torch.tensor(np.array(self.states), dtype=torch.float32).to(self.device)
        next_value = self.critic(next_state).detach()[0, 0]

        advantage = 0
        for t in range(target_step - 1, -1, -1):
            delta = rewards[t] + masks[t] * next_value - values[t]
            advantages[t] = advantage = delta + masks[t] * self.lambda_gae_adv * advantage
            next_value = values[t]

        return advantages

import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.nmix import Mixer
from modules.mixers.vdn import VDNMixer
from modules.mixers.qatten import QattenMixer
from envs.matrix_game import print_matrix_status
from utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets
import torch as th
from torch.optim import RMSprop, Adam
import numpy as np
from utils.th_utils import get_parameters_num
from utils.attack_util import logits_margin, attack, get_diff, noise_atk, get_max_diff
from torch.autograd import Variable
from utils.pareto import MinNormSolver, gradient_normalizers

class NQLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        
        self.last_target_update_episode = 0
        self.device = th.device('cuda' if args.use_cuda  else 'cpu')
        self.params = list(mac.parameters())

        if args.mixer == "qatten":
            self.mixer = QattenMixer(args)
        elif args.mixer == "vdn":
            self.mixer = VDNMixer()
        elif args.mixer == "qmix":
            self.mixer = Mixer(args)
        else:
            raise "mixer error"
        self.target_mixer = copy.deepcopy(self.mixer)
        self.params += list(self.mixer.parameters())

        print('Mixer Size: ')
        print(get_parameters_num(self.mixer.parameters()))
        
        if self.args.weight_td:
            self.log_var_a = th.zeros((1,), requires_grad=True, device=self.device)
            self.log_var_b = th.zeros((1,), requires_grad=True, device=self.device)
            self.log_var_b.data.fill_(self.args.robust_lambda)
            self.params = ([p for p in self.params] + [self.log_var_a] + [self.log_var_b])
            
        elif self.args.weight_adv_loss:
            self.log_var_b = th.zeros((1,), requires_grad=True, device=self.device)
            self.log_var_b.data.fill_(self.args.robust_lambda) 
            self.numerator = self.args.robust_lambda // 10
            self.params = ([p for p in self.params]+ [self.log_var_b])
        

        if self.args.optimizer == 'adam':
            self.optimiser = Adam(params=self.params,  lr=args.lr, weight_decay=getattr(args, "weight_decay", 0))
        else:
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.train_t = 0

        # priority replay
        self.use_per = getattr(self.args, 'use_per', False)
        self.return_priority = getattr(self.args, "return_priority", False)
        if self.use_per:
            self.priority_max = float('-inf')
            self.priority_min = float('inf')
        
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, per_weight=None):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1] # the total reward
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1]) # whether the sample is valid
        avail_actions = batch["avail_actions"]
        
        # just for robust regularizer
        adv_terminated = batch["terminated"].float()
        adv_mask = batch["filled"].float()
        adv_mask[:, 1:] = adv_mask[:, 1:] * (1 - adv_terminated[:, :-1])
        
        
        # Calculate estimated Q-Values
        self.mac.agent.train() # train mode
        mac_out = []
        adv_margin = []
        self.mac.init_hidden(batch.batch_size)
        
                
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t) # normal Q-values
            mac_out.append(agent_outs)
            
        # if t_env >= self.args.t_max // 4:        
            if self.args.diff_regular or self.args.pareto:
                adv_batch = noise_atk(batch, self.args)
                adv_agent_out = self.mac.forward(adv_batch, t=t)    
                adv_margin.append(get_max_diff(agent_outs, adv_agent_out))
                
        mac_out = th.stack(mac_out, dim=1)  # Concat over time. (batch, seq_len, n_agents, n_actions)
        # Pick the Q-Values for the actions taken by each agent
        # (batch, seq_len - 1, n_agents)
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        with th.no_grad():
            self.target_mac.agent.train()
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)

            # We don't need the first timesteps Q-Value estimate for calculating targets
            target_mac_out = th.stack(target_mac_out, dim=1)  # Concat across time

            # Max over target Q-Values/ Double q learning
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            
            # Calculate n-step Q-Learning targets
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"]) # (batch, seq_len, 1)

            if getattr(self.args, 'q_lambda', False):
                qvals = th.gather(target_mac_out, 3, batch["actions"]).squeeze(3)
                qvals = self.target_mixer(qvals, batch["state"])

                targets = build_q_lambda_targets(rewards, terminated, mask, target_max_qvals, qvals,
                                    self.args.gamma, self.args.td_lambda)
            else: # (batch, seq_len - 1, n_agents) the last timestep is removed
                targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals, 
                                                    self.args.n_agents, self.args.gamma, self.args.td_lambda)

        # Mixer
        # (batch, seq_len - 1, num_agents) -> (batch, seq_len - 1, 1)
        # change individual q-values to joint q-values
        chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])

        td_error = (chosen_action_qvals - targets.detach())
        td_error2 = 0.5 * td_error.pow(2) # (batch, seq_len - 1, 1)

        mask = mask.expand_as(td_error2) # (batch, seq_len - 1, 1)
        masked_td_error = td_error2 * mask # (batch, seq_len - 1, 1)


        # important sampling for PER
        if self.use_per:
            per_weight = th.from_numpy(per_weight).unsqueeze(-1).to(device=self.device)
            masked_td_error = masked_td_error.sum(1) * per_weight


        # TODO: here need to add robust term
        loss = L_td = masked_td_error.sum() / mask.sum()
        

    # if t_env >= self.args.t_max // 4:        
        
        if self.args.robust_regular or self.args.diff_regular or self.args.pareto: 
            
            adv_margin = th.stack(adv_margin, dim=0).transpose(0, 1) * adv_mask.squeeze(2) # (batch, seq_len)
            adv_loss = adv_margin.sum() / adv_mask.sum()
            if self.args.weight_td:
                loss = th.exp(-self.log_var_a) * loss + self.log_var_a
                loss += (adv_loss * th.exp(-self.log_var_b) + self.log_var_b)
            elif self.args.weight_adv_loss:
                loss = loss + adv_loss * (-self.log_var_b) + (float(self.args.robust_lambda) / self.log_var_b)
            elif self.args.pareto:
                
                # lossa = masked_td_error
                # lossb = adv_margin[:, :-1].unsqueeze(2)
                # if th.matmul(lossa.transpose(1,2), lossb).sum() >= th.matmul(lossa.transpose(1,2), lossa).sum():
                #     gamma = 1
                # elif th.matmul(lossa.transpose(1,2), lossb).sum() >= th.matmul(lossb.transpose(1,2), lossb).sum():
                #     gamma = 0
                # else:
                #     gamma = th.matmul((lossb - lossa).transpose(1,2), lossb).mean() / th.norm(lossa - lossb, p=2) ** 2
                # lossa = lossa.sum() / mask.sum()
                # lossb = lossb.sum() / adv_mask.sum()
                # loss = gamma * lossa + (1 - gamma) * lossb
                
                # compute grad of td
                loss_data = {}
                grads = {}
                scale = {}
                tasks = ['td', 'adv']
                loss_data['td'] = loss.data
                loss_data['adv'] = adv_loss.data
                self.optimiser.zero_grad()
                loss.backward(retain_graph=True)
                grads['td'] = []
                for p in self.params:
                    if p.grad is not None:
                        grads['td'].append(Variable(p.grad.data.clone(), requires_grad=False))
                    
                # compute grad of adv    
                self.optimiser.zero_grad()
                adv_loss.backward()
                grads['adv'] = []
                for p in self.params:
                    if p.grad is not None:
                        grads['adv'].append(Variable(p.grad.data.clone(), requires_grad=False))
                        
                        
                # normalize
                gn = gradient_normalizers(grads, loss_data, self.args.normalization_type)
                for t in tasks:
                    for gr_i in range(len(grads[t])):
                        grads[t][gr_i] = grads[t][gr_i] / gn[t]
                        
                # use F-W algorithm to compute pareto optimal
                sol, _ = MinNormSolver.find_min_norm_element([grads[t] for t in tasks])
                for i, t in enumerate(tasks):
                    scale[t] = float(sol[i])
                    
                    
                self.optimiser.zero_grad()
                
                td_loss, adv_loss = self.compute_td_and_adv(batch, t_env, episode_num, per_weight) 
                
                # scale['adv'] = min(0.01, scale['adv'])
                # scale['td'] = 1 - scale['adv']
                                    
                loss = scale['td'] * td_loss + scale['adv'] * adv_loss
                        
            else: # fixed lambda
                loss += adv_loss * float(self.args.robust_lambda)
            
                

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_td", L_td.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            
            if self.args.robust_regular or self.args.diff_regular: 
                self.logger.log_stat("adv_margin", adv_margin.mean().item(), t_env)
                self.logger.log_stat("loss", loss.item(), t_env)
                if self.args.weight_td:
                    self.logger.log_stat("log_var_a", self.log_var_a.item(), t_env)
                    self.logger.log_stat("log_var_b", self.log_var_b.item(), t_env)
                elif self.args.weight_adv_loss:
                    self.logger.log_stat("log_var_b", self.log_var_b.item(), t_env)
                elif self.args.pareto:
                    self.logger.log_stat("td_weight", scale['td'], t_env)
                    self.logger.log_stat("adv_weight", scale['adv'], t_env)
            
            self.log_stats_t = t_env
            
            # print estimated matrix
            if self.args.env == "one_step_matrix_game":
                print_matrix_status(batch, self.mixer, mac_out)

        # return info
        info = {}
        # calculate priority
        if self.use_per:
            if self.return_priority:
                info["td_errors_abs"] = rewards.sum(1).detach().to('cpu')
                # normalize to [0, 1]
                self.priority_max = max(th.max(info["td_errors_abs"]).item(), self.priority_max)
                self.priority_min = min(th.min(info["td_errors_abs"]).item(), self.priority_min)
                info["td_errors_abs"] = (info["td_errors_abs"] - self.priority_min) \
                                / (self.priority_max - self.priority_min + 1e-5)
            else:
                info["td_errors_abs"] = ((td_error.abs() * mask).sum(1) \
                                / th.sqrt(mask.sum(1))).detach().to('cpu')
        return info

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
            
    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))


    def compute_td_and_adv(self, batch: EpisodeBatch, t_env: int, episode_num: int, per_weight=None):
       # Get the relevant quantities
        rewards = batch["reward"][:, :-1] # the total reward
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1]) # whether the sample is valid
        avail_actions = batch["avail_actions"]
        
        # just for robust regularizer
        adv_terminated = batch["terminated"].float()
        adv_mask = batch["filled"].float()
        adv_mask[:, 1:] = adv_mask[:, 1:] * (1 - adv_terminated[:, :-1])
        
        
        # Calculate estimated Q-Values
        self.mac.agent.train() # train mode
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        adv_margin = []

                
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t) # normal Q-values
            mac_out.append(agent_outs)
            
            
            if self.args.robust_regular: 
                envs_not_terminated = [i for i in range(batch.batch_size) if adv_mask[i][t]]
                adv_batch = attack(self.mac, batch, self.args, t, bs=envs_not_terminated)
                with th.no_grad():
                    adv_agent_out = self.mac.forward(adv_batch, t=t) # (batch, n_agents, n_actions) adv Q-values
                label = th.argmax(agent_outs, dim=2).clone().detach() # (batch, n_agents)
                adv_margin.append(logits_margin(adv_agent_out[envs_not_terminated], label[envs_not_terminated], avail_actions[envs_not_terminated][:, t]))
            
            else:
                adv_batch = noise_atk(batch, self.args)
                adv_agent_out = self.mac.forward(adv_batch, t=t)    
                adv_margin.append(get_max_diff(agent_outs, adv_agent_out))
                
        mac_out = th.stack(mac_out, dim=1)  # Concat over time. (batch, seq_len, n_agents, n_actions)
        # Pick the Q-Values for the actions taken by each agent
        # (batch, seq_len - 1, n_agents)
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        with th.no_grad():
            self.target_mac.agent.train()
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)

            # We don't need the first timesteps Q-Value estimate for calculating targets
            target_mac_out = th.stack(target_mac_out, dim=1)  # Concat across time

            # Max over target Q-Values/ Double q learning
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            
            # Calculate n-step Q-Learning targets
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"]) # (batch, seq_len, 1)

            if getattr(self.args, 'q_lambda', False):
                qvals = th.gather(target_mac_out, 3, batch["actions"]).squeeze(3)
                qvals = self.target_mixer(qvals, batch["state"])

                targets = build_q_lambda_targets(rewards, terminated, mask, target_max_qvals, qvals,
                                    self.args.gamma, self.args.td_lambda)
            else: # (batch, seq_len - 1, n_agents) the last timestep is removed
                targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals, 
                                                    self.args.n_agents, self.args.gamma, self.args.td_lambda)

        # Mixer
        # (batch, seq_len - 1, num_agents) -> (batch, seq_len - 1, 1)
        # change individual q-values to joint q-values
        chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])

        td_error = (chosen_action_qvals - targets.detach())
        td_error2 = 0.5 * td_error.pow(2) # (batch, seq_len - 1, 1)

        mask = mask.expand_as(td_error2) # (batch, seq_len - 1, 1)
        masked_td_error = td_error2 * mask # (batch, seq_len - 1, 1)


        # important sampling for PER
        if self.use_per:
            per_weight = th.from_numpy(per_weight).unsqueeze(-1).to(device=self.device)
            masked_td_error = masked_td_error.sum(1) * per_weight


        # TODO: here need to add robust term
        loss = masked_td_error.sum() / mask.sum()
        
        adv_margin = th.stack(adv_margin, dim=0).transpose(0, 1) * adv_mask.squeeze(2) # (batch, seq_len)
        adv_loss = adv_margin.sum() / adv_mask.sum()
        
        return loss, adv_loss
    
    # def compute_adv(self, batch: EpisodeBatch, t_env: int, episode_num: int, agent_outs, per_weight=None):
    #     # Get the relevant quantities
    #     rewards = batch["reward"][:, :-1] # the total reward
    #     actions = batch["actions"][:, :-1]
    #     terminated = batch["terminated"][:, :-1].float()
    #     mask = batch["filled"][:, :-1].float()
    #     mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1]) # whether the sample is valid
    #     avail_actions = batch["avail_actions"]
        
    #     # just for robust regularizer
    #     adv_terminated = batch["terminated"].float()
    #     adv_mask = batch["filled"].float()
    #     adv_mask[:, 1:] = adv_mask[:, 1:] * (1 - adv_terminated[:, :-1])
        
    #     for t in range(batch.max_seq_length):

        
        
        
        
    #     adv_margin = th.stack(adv_margin, dim=0).transpose(0, 1) * adv_mask.squeeze(2) # (batch, seq_len)
    #     adv_loss = adv_margin.sum() / adv_mask.sum()
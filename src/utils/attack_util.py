import torch
from torch import autograd
import torch.nn as nn
import torch.optim as optim
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
TARGET_MULT = 10000.0

def get_diff(normal_Q, perturbed_Q): # Q: (batch_size, num_agents, n_actions)
    return torch.norm(normal_Q[:] - perturbed_Q[:], p="fro", dim=tuple(range(1, normal_Q.dim())))

def get_max_diff(normal_Q, perturbed_Q): # Q: (batch_size, num_agents, n_actions)
    element_wise_diff = torch.norm(normal_Q - perturbed_Q, p="fro", dim=-1)
    maxdiff, _ = torch.max(element_wise_diff, dim=1)
    return maxdiff

def logits_margin(logits, y, avail_actions):
    
    # logits is the Q value just calculated (batch_size, num_agents, n_actions)
    # y is the target action (batch_size, num_agents)
    comp_logits = logits - torch.zeros_like(logits).scatter(2, torch.unsqueeze(y, 2), 1e10) # mask the Q value of target action
    
    # # Create a mask where avail_actions is 0
    # mask = (avail_actions.detach().clone() == 0).float()

    # # Subtract 1e10 from the positions where avail_actions is 0
    # comp_logits = comp_logits - mask * 1e10
    
    sec_logits, _ = torch.max(comp_logits, dim=2)
    margin = sec_logits - torch.gather(logits, 2, torch.unsqueeze(y, 2)).squeeze(2)
    margin = margin.sum()  # (batch_size, num_agents) -> 1
    return margin

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

def to_one_hot(y, num_actions):
    """
    Take a batch of label y with n dims and convert it to
    1-hot representation with n+1 dims.
    Link: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/24
    """
    # y = y.detach().clone().view(-1, 1)
    # y_onehot = y.new_zeros((y.size()[0], num_classes)).scatter_(1, y, 1)
    y_onehot = torch.FloatTensor(1, num_actions)
    y_onehot.zero_()
    y_onehot.scatter_(1, torch.tensor([[y]]), 1)
    return Variable(y_onehot)



def noise_atk(batch, args):
    epsilon = getattr(args, 'epsilon', 0.2)
    adv_batch = deepcopy(batch)
    adv_obs = torch.normal(batch['obs'], torch.ones_like(batch['obs']) * epsilon)
    adv_batch['obs'].data = adv_obs
    return adv_batch


def noise_atk_on_one_agent(model, batch, y, t_ep, t_env, bs, test_mode, verbose, args):
    epsilon = getattr(args, 'epsilon', 0.2)
    adv_batch = deepcopy(batch)
    if batch['obs'][bs,t_ep,0].numel() == 0:
        return adv_batch
    adv_obs = torch.normal(batch['obs'][bs,t_ep,0], torch.ones_like(batch['obs'][bs,t_ep,0]) * epsilon)
    adv_batch['obs'][bs,t_ep,0].data = adv_obs
    return adv_batch

def pgd(model, batch, y, t_ep, t_env, bs, test_mode, verbose=False, args={}):
    epsilon = getattr(args, 'epsilon', 0.2)
    niters = getattr(args, 'niters', 4)
    loss_func = logits_margin
    step_size = epsilon * 1.0 / niters
    y = Variable(torch.tensor(y))
    if verbose:
        print('epislon: {}, step size: {}, target label: {}'.format(epsilon, step_size, y))
        
    adv_batch = deepcopy(batch)
    rand = getattr(args, 'random_start', True)
    if rand:
        noise = 2 * epsilon * torch.rand(batch['obs'].data.size()) - epsilon # (-epislon, epsilon)
        if USE_CUDA:
            noise = noise.cuda()
        adv_obs = adv_batch['obs'].data + noise
        adv_obs = Variable(adv_obs.data, requires_grad=False)
        if verbose:
            print('max diff after adding noise: ', np.max(abs(adv_obs.data.cpu().numpy()-batch['obs'].data.cpu().numpy())))
    else:
        adv_obs = Variable(adv_batch['obs'], requires_grad=False)
    adv_data = {'obs': adv_obs} # create a dict for update.
    adv_batch.update(adv_data)
    adv_agent_inputs = model._build_inputs(adv_batch, t_ep) 
    adv_agent_inputs = Variable(adv_agent_inputs, requires_grad=True)
    obs_dim = batch['obs'].shape[-1]
    model.agent.eval()
    for i in range(niters):
        
        logits, model.hidden_states = model.agent.forward(adv_agent_inputs, model.hidden_states)
        loss = loss_func(logits[bs], y, batch['avail_actions'][bs,t_ep]) # logits lost grad! with torch.no_grad causes this problem.
        if verbose:
            print('current loss: ', loss.data.cpu().numpy())
        
        grad = torch.autograd.grad(loss, adv_agent_inputs,
                            retain_graph=False, create_graph=False)[0]
        eta = step_size * grad.data.sign()
        # adv_data['obs'] = Variable(adv_data['obs'][:,t_ep].data + eta, requires_grad=True)
        # adjust the (perturbed_obs - original_obs) to be within [-epsilon, epsilon]
        eta = torch.clamp(eta, -epsilon, epsilon)
        adv_agent_inputs = adv_agent_inputs.data + eta        
        clamped_update = torch.clamp(adv_agent_inputs[...,:obs_dim].data - batch['obs'][:,t_ep].data, -epsilon, epsilon)        
        with torch.no_grad():
            adv_agent_inputs[..., :obs_dim] = clamped_update + batch['obs'][:, t_ep]
              
        adv_agent_inputs = Variable(adv_agent_inputs, requires_grad=True)
        # adv_data['obs'][:,t_ep].data = adv_agent_inputs.data[...,:adv_data['obs'].shape[-1]]
        # adv_batch.update(adv_data)
        # adv_agent_inputs = model._build_inputs(adv_batch, t_ep) 
        # adv_agent_inputs = Variable(adv_agent_inputs, requires_grad=True)
        if verbose:
            print('linf diff after clamp: ',torch.max(abs(adv_agent_inputs[...,:obs_dim].data-batch['obs'][:,t_ep].data)))
    with torch.no_grad():
        adv_batch['obs'][:,t_ep] = adv_agent_inputs[...,:adv_data['obs'].shape[-1]]
    return adv_batch


def fgsm(model, batch, y, t_ep, t_env, bs, test_mode, verbose, args):
    epsilon=getattr(args, 'epsilon', 0.2)
    adv_batch = deepcopy(batch)
    
    adv_obs = Variable(adv_batch['obs'].data, requires_grad=True)
    logits = model.forward(adv_obs, t_ep)
    loss = F.nll_loss(logits, y)
    model.agent.zero_grad()
    loss.backward()
    eta = epsilon * adv_obs.grad.data.sign()
    adv_obs = Variable(adv_obs.data + eta, requires_grad=True)
    return adv_obs.data


def cw(model, batch, y, t_ep, t_env, bs, test_mode, verbose, args):
    C = getattr(args, 'C', 0.0001)
    niters = getattr(args, 'niters', 50)
    step_size = getattr(args, 'step_size', 0.01)
    confidence = getattr(args, 'confidence', 0.0001)

    
    adv_batch = deepcopy(batch)
    adv_obs = Variable(adv_batch['obs'], requires_grad=True)
    
    y_onehot = to_one_hot(y, model.num_actions).float()
    optimizer = optim.Adam([adv_obs], lr=step_size)
    for i in range(niters):
        logits = model.forward()
        real = (y_onehot * logits).sum(dim=1)
        other = ((1.0 - y_onehot) * logits - (y_onehot * TARGET_MULT)).max(1)[0]
        loss1 = torch.clamp(real - other + confidence, min=0.)
        loss2 = torch.sum((batch - adv_obs).pow(2), dim=[1,2,3])
        loss = loss1 + loss2 * C

        optimizer.zero_grad()
        model.features.zero_grad()
        loss.backward()
        optimizer.step()
        if verbose:
           print('loss1: {}, loss2: {}'.format(loss1, loss2))
    return adv_batch


def attack(model, batch, args, t_ep, bs,  t_env=0, test_mode=False): # here bs is envs_not_terminated, may be not equal to batch_size.
    # under robust regularization, loss_func is logits_margin
    method = getattr(args, 'attack_method', 'pgd')
    verbose = getattr(args, 'verbose', False)
    y = model.select_actions(batch, t_ep, t_env, bs, test_mode) # actions. (batch_size, n_agents)
    if method == 'cw':
        atk = cw
    elif method == 'fgsm':
        atk = fgsm
    elif method == 'pgd':
        atk = pgd
    elif method == 'noise_atk_on_one_agent':
        atk = noise_atk_on_one_agent
    else:
        atk = noise_atk

    # if args.env == 'sc2':
    #     atk = noise_atk_on_one_agent
    if method == 'noise':
        adv_batch = atk(batch, args)
    else:
        adv_batch = atk(model, batch, y, t_ep, t_env, bs, test_mode, verbose=verbose, args=args)
    if verbose:
        print('obs: {}->{},'.format(batch['obs'], adv_batch['obs']))
    return adv_batch


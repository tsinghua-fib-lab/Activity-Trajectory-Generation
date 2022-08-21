import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint
import math
from torch.autograd import Variable
import utils
import pandas as pd

class MLP(nn.Module):

    def __init__(self, dim_in,  dim_hidden, dim_out, num_hidden=0, activation=nn.Tanh()):
        super(MLP, self).__init__()

        if num_hidden == 0:
            self.linears = nn.ModuleList([nn.Linear(dim_in, dim_out)])
        elif num_hidden >= 1:
            self.linears = nn.ModuleList()
            self.linears.append(nn.Linear(dim_in, dim_hidden))
            self.linears.extend([nn.Linear(dim_hidden, dim_hidden) for _ in range(num_hidden-1)])
            self.linears.append(nn.Linear(dim_hidden, dim_out))
        else:
            raise Exception('number of hidden layers must be positive')

        for m in self.linears:
            nn.init.normal_(m.weight, mean=0, std=0.1)
            nn.init.uniform_(m.bias, a=-0.1, b=0.1)

        self.activation = activation

    def forward(self, x):
        for m in self.linears[:-1]:
            x = self.activation(m(x))

        return self.linears[-1](x)


def state2tensor(state, device): # state = (time, loc, (Lambda, init_state))

    time, loc, (Lambda, init_state) = zip(*state)

    time = torch.tensor(time).to(device)

    loc = torch.tensor(loc).to(device)

    (Lambda, init_state) = (Lambda.to(device), init_state.to(device))

    return time, loc, (Lambda, init_state)


class Policy_net(nn.Module):
    def __init__(self,args,device,STPP_model,generator):
        super(Policy_net, self).__init__()
        self.args = args
        self.device = device
        self.STPP_model = STPP_model
        self.generator = generator


    def act(self, batch):
        self.generator.sample_seqs(self.STPP_model, batch)
        seqs = (self.generator.event_times, self.generator.event_types, [[s.detach().cpu().numpy().tolist() for s in seq[1:]] for seq in self.generator.spatial_locations], self.generator.state_save, self.generator.init_state_save.detach().cpu())
        self.generator.state_init()
        return seqs

    def get_value(self, state, value_net):
        value = value_net(state)
        return value

    def evaluate_actions(self, inputs): 

        event_times, event_types, spatial_locations, init_state_save, mask = inputs

        N, T = event_times.shape[0], event_times.shape[1]
       
        t0, t1 = map(lambda x: utils.cast(x, self.device), utils.get_t0_t1(self.args.data))

        t1 = None

        if torch.isnan(init_state_save).any() or torch.isnan(event_times).any() or torch.isnan(event_types).any() or torch.isnan(spatial_locations).any():
            print('nan error')
            exit()

        if not ((event_times >=0).all() & (event_times <=self.args.t_max).all()):
            print('time error')
            exit()
        
        space_loglik, _, _, prob_event, _, (intensities, Lambdas) = self.STPP_model(init_state_save, event_times.squeeze(dim=2), spatial_locations,event_types.squeeze(dim=2), mask, t0, t1)

        delta_lambda = Lambdas[:,1:]-Lambdas[:,:-1]

        if not (delta_lambda>=0).all():
            print('lambda negtive error')
            exit()

        N, T = mask.shape

        index = torch.arange(N*T).long()
        prob_event = prob_event.reshape(N*T,-1)
        event_types = event_types.reshape(N*T).long()
        log_prob_event = torch.log(prob_event[index,event_types].reshape(N,T))

        time_loglik = (torch.log(intensities) + log_prob_event - delta_lambda) * mask

        space_loglik = space_loglik * mask

        dist = torch.distributions.Categorical(prob_event.reshape(N*T,-1))

        action_log_probs = (space_loglik + time_loglik) * mask

        dist_entropy = dist.entropy() * mask.reshape(N*T)

        return action_log_probs[:,1:], dist_entropy.reshape(N, T)[:,1:]


class Base(nn.Module):
    def __init__(self,args,hidden_dim):
        super(Base, self).__init__()
        self.args = args
        self.hidden_size = self.args.hidden_size

        self.week_emb = nn.Embedding(
            num_embeddings=7, embedding_dim=args.week_dim
        )
        self.hour_emb = nn.Embedding(
            num_embeddings=24, embedding_dim=args.hour_dim
        )

        self.GRU = nn.GRU(1 + self.args.week_dim + self.args.hour_dim + self.args.type_dim + 2 + hidden_dim, self.args.hidden_size, 1, batch_first=True)

    def forward(self, inputs):

        x_emb = torch.cat(inputs,dim=-1)

        x_rnn, _ = self.GRU(x_emb)

        return x_rnn[:,-1:,:]


class Value_net(nn.Module):
    def __init__(self,args,device,STPP_model, Base):
        super(Value_net, self).__init__()
        self.args = args
        self.device = device
        self.STPP_model = STPP_model

        self.week_emb = Base.week_emb
        self.hour_emb = Base.hour_emb

        self.hidden_size = self.args.hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.state_encoder = Base

        self.critic_linear = MLP(self.hidden_size, self.hidden_size, 1, 1)

    def forward(self, state):
        event_times, event_types, spatial_locations, tpp_state = state

        N, T = event_times.shape[0], event_times.shape[1]

        week = (event_times % 7).long()
        hour = ((event_times % 1) * 24).long()

        type_emb = self.STPP_model.type_emb(event_types).squeeze(dim=2)

        if self.args.weekhour:

            week_emb = self.STPP_model.week_emb(week).squeeze(dim=2)
            hour_emb = self.STPP_model.hour_emb(hour).squeeze(dim=2)

        else:
            week_emb = self.week_emb(week).squeeze(dim=2)
            hour_emb = self.hour_emb(hour).squeeze(dim=2)

        time_emb = [event_times[:,:i+1] for i in range(event_times.shape[1])]

        week_emb = [week_emb[:,:i+1] for i in range(week_emb.shape[1])]

        hour_emb = [hour_emb[:,:i+1] for i in range(hour_emb.shape[1])]

        type_emb = [type_emb[:,:i+1] for i in range(type_emb.shape[1])]

        loc_emb = [spatial_locations[:,:i+1] for i in range(spatial_locations.shape[1])]

        tpp_state = [tpp_state[:,:i+1] for i in range(tpp_state.shape[1])]

        state_emb = []

        for i in range(len(time_emb)):
            inputs = [time_emb[i],week_emb[i],hour_emb[i],type_emb[i],loc_emb[i], tpp_state[i]]
            state_emb.append(self.state_encoder(inputs))

        state_emb = torch.cat(state_emb,dim=1)

        value = self.critic_linear(state_emb)

        return value.reshape(N,T)


class Discriminator(nn.Module):
    def __init__(self,args,device,STPP_model, Base):
        super(Discriminator, self).__init__()

        self.args = args
        self.device = device
        self.STPP_model = STPP_model
        self.week_emb = Base.week_emb
        self.hour_emb = Base.hour_emb

        self.hidden_size = self.args.hidden_size

        self.state_encoder = Base

        self.disc = nn.Sequential(MLP(self.hidden_size + self.args.type_dim + 2 + 1 + 1, self.hidden_size, 1, 1),nn.Sigmoid())

    def forward(self, state, action):

        event_times, event_types, spatial_locations, tpp_state = state

        N, T = event_times.shape[0], event_times.shape[1]

        week = (event_times % 7).long()
        hour = ((event_times % 1) * 24).long()

        type_emb = self.STPP_model.type_emb(event_types).squeeze(dim=2)

        if self.args.weekhour:
            week_emb = self.STPP_model.week_emb(week).squeeze(dim=2)
            hour_emb = self.STPP_model.hour_emb(hour).squeeze(dim=2)

        else:
            week_emb = self.week_emb(week).squeeze(dim=2)
            hour_emb = self.hour_emb(hour).squeeze(dim=2)

        # state encoding
        time_emb = [event_times[:,:i+1] for i in range(event_times.shape[1])]

        week_emb = [week_emb[:,:i+1] for i in range(week_emb.shape[1])]

        hour_emb = [hour_emb[:,:i+1] for i in range(hour_emb.shape[1])]

        type_emb = [type_emb[:,:i+1] for i in range(type_emb.shape[1])]

        loc_emb = [spatial_locations[:,:i+1] for i in range(spatial_locations.shape[1])]

        tpp_state = [tpp_state[:,:i+1] for i in range(tpp_state.shape[1])]

        state_emb = []

        for i in range(len(time_emb)):
            inputs = [time_emb[i],week_emb[i],hour_emb[i],type_emb[i],loc_emb[i], tpp_state[i]]
            state_emb.append(self.state_encoder(inputs))

        state_emb = torch.cat(state_emb,dim=1)

        Interval_action, Type_action, Loc_action = action

        geodistance = torch.sqrt(torch.sum((spatial_locations - Loc_action).pow(2),dim=2)).float()

        type_action_emb = self.STPP_model.type_emb(Type_action).squeeze(dim=2)

        inputs = torch.cat((state_emb.reshape(N*T,-1), Interval_action.reshape(N*T,-1), type_action_emb.reshape(N*T,-1), Loc_action.reshape(N*T,-1), geodistance.reshape(N*T,-1)),dim=1)

        reward = self.disc(inputs)

        return reward.reshape(N,T)
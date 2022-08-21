from torch.distributions import Categorical
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint

import numpy as np
import math
import copy
import time
import pandas as pd

import utils
from .spatial.cnf import TimeVariableCNF, build_fc_odefunc, max_rms_norm

class SeqGen(object):
    '''
    here is the sequence generator using neural spatio-temporal point process
    '''
    def __init__(self, args, device):
        self.args = args
        self.NN = None
        self.device = device
        self.D = 2


    def compute_intensity_given_past(self,current_time,model,nlinspace=1):

        last_time = torch.tensor(0.0).expand(self.NN).to(self.device)

        if self.state_update is not None:
            last_time = self.state_update[0]
            state = self.state_update[-1]

        else:
            init_state = (torch.randn([self.NN,model.temporal_model.hidden_dims[0]]) / math.sqrt(model.temporal_model.hidden_dims[0])).to(self.device) 

            self.init_state_save = init_state

            init_state = model.temporal_model.init_transform(init_state)

            state = [
                torch.zeros(1).to(self.device).expand(self.NN),  # Lambda(t_0)
                init_state.to(self.device),
            ]

            self.prejump_hidden_states.append(init_state.detach().cpu())

        state_traj = model.temporal_model.ode_solver.integrate(last_time,current_time, state, nlinspace = nlinspace, method=self.args.ode_method)

        hiddens = state_traj[1] # (1 + nlinspace, N, D);
        
        hiddens = hiddens[1:, :, :]

        state = [s[-1] for s in state_traj] 

        Lambda, tpp_state = state 

        self.state_update = [current_time, state]

        self.intensity = model.temporal_model.get_intensity(tpp_state)

        self.intensity_event = model.temporal_model.get_intensity_event(tpp_state)

        return hiddens, tpp_state, Lambda


    def sample_time_type(self, model,mask_t_max=None):

        #gprejump_hidden_states, tpp_state, Lambda = [], [], []

        time_current = torch.tensor(0.0).to(self.device).expand(self.NN).clone()

        if len(self.event_times[0])>0:
            time_current = torch.tensor([i[-1] for i in self.event_times]).to(self.device)

        mask = time_current>=self.args.t_max

        if mask_t_max is None:
            mask_t_max = time_current>=self.args.t_max

        u = torch.tensor([1.5 for _ in range(self.NN)]).to(self.device)

        self.compute_intensity_given_past(time_current.clone().detach(),model)

        state_save = [self.state_update[1][0].detach().cpu(), self.state_update[1][1].detach().cpu()]

        intensity_hazard = self.intensity.clone().squeeze()
    
        while (1-mask.float()).bool().any():

            Exp = torch.distributions.Exponential(torch.tensor(1.0))
            E = torch.tensor([Exp.sample()for _ in range(self.NN)]).to(self.device)
            Uni = torch.distributions.uniform.Uniform(torch.tensor(0.0),torch.tensor(1.0))
            U = torch.tensor([Uni.sample()for _ in range(self.NN)]).to(self.device)

            interval = E/intensity_hazard

            mask_t_max = mask_t_max | (time_current + interval>=self.args.t_max)

            mask = mask | mask_t_max

            time_current += interval * (1-mask.float())
            
            hiddens_t, tpp_state_t, Lambda_t = self.compute_intensity_given_past(time_current.clone().detach(), model)

            u = (U * intensity_hazard / self.intensity)

            # this snippet below is for adaptive thining
            # it can speed things up
            # by decreasing upper bound
            # but it is toggled off when data is randomly generated at the beginning of this project
            
            intensity_hazard = self.intensity.squeeze()
            mask = mask | (u<1)

        loc_update = torch.zeros(self.NN).to(self.device)

        flag= 0 

        for i in range(self.NN):
            if len(self.event_times[i])==0 or time_current[i].item() != self.event_times[i][-1] and time_current[i].item()<self.args.t_max: 
                flag=1
                loc_update[i] += 1
                dist = Categorical(self.intensity_event[i])
                event_type = dist.sample().item()
                self.event_types[i].append(event_type)
                self.event_times[i].append(time_current[i].item())
                self.state_save[i].append([state_save[0][i].item(),state_save[1][i].detach().cpu().numpy().tolist()])
        
        prejump_hidden_states = hiddens_t.transpose(0, 1)
        
        if flag==1 and isinstance(self.prejump_hidden_states,list):
            self.prejump_hidden_states.append(prejump_hidden_states.detach())

        return prejump_hidden_states, tpp_state_t, Lambda_t, mask_t_max, loc_update.bool()


    def hidden_update(self,model, tpp_state, Lambda):

        cond_type = model.type_emb(torch.tensor([i[-1] for i in self.event_types]).to(self.device))

        cond_spatial = torch.cat([i[-1].unsqueeze(0) for i in self.spatial_locations],dim=0).to(self.device)

        if self.args.weekhour:

            if self.args.data=='Tencent_deploy':

                cond_week = model.week_emb((torch.tensor([i[-1] for i in self.event_times]).to(self.device) / 24).long())

                cond_hour = model.hour_emb(torch.tensor([i[-1] for i in self.event_times]).to(self.device).long())

            else:

                cond_week = model.week_emb((torch.tensor([i[-1] for i in self.event_times]).to(self.device) % 7).long())

                cond_hour = model.hour_emb(((torch.tensor([i[-1] for i in self.event_times]).to(self.device) % 1) * 24).long())

            cond = torch.cat((cond_spatial, cond_type, cond_week, cond_hour),dim=1)

        else:
            cond = torch.cat((cond_spatial,cond_type), dim=1)

        event_time = torch.tensor([i[-1] for i in self.event_times]).to(self.device)

        updated_tpp_state = model.temporal_model.hidden_state_dynamics.update_state(event_time, tpp_state, cond=cond.float())

        self.state_update[1][1] = updated_tpp_state

    def sample_one_event_attn(self, model,flag=True):
        if len(self.event_times[0])==0:
            prejump_hidden_states, tpp_state, Lambda, mask, loc_mask = self.sample_time_type(model)
            self.mask_t_max = mask.clone()
        else:
            if flag:
                prejump_hidden_states, tpp_state, Lambda, mask, loc_mask = self.sample_time_type(model,self.mask_t_max.clone())
                self.mask_t_max = mask.clone()
            else:
                prejump_hidden_states, tpp_state, Lambda, mask, loc_mask = self.sample_time_type(model,None)
        if self.x_base is None:
            x = []
            for i in range(self.NN):
                xs = utils.gaussian_sample(model.spatial_model.z_mean,model.spatial_model.z_logstd)
                x.append(xs)
            x = torch.cat(x,dim=0).to(self.device)
            t0 = torch.zeros([x.shape[0],1]).to(self.device)
            t1 = torch.zeros([x.shape[0],1]).to(self.device) + model.spatial_model.time_offset

            self.x_base, _ = model.spatial_model.base_cnf.integrate(t0,t1,x,torch.zeros_like(t0).to(self.device)) # N * 2
        
        if not loc_mask.any():
            return mask

        x = self.sample_location_attn(model)

        for i in range(self.NN):
            if loc_mask[i]:
                self.spatial_locations[i].append(x[i].cpu().detach())
        self.hidden_update(model,tpp_state,Lambda)

        return mask

        
    def sample_location_attn(self, model):
        if len(self.spatial_locations[0]) == 0:
            spatial_locations = self.x_base.unsqueeze(dim=1).clone().to(self.device)
            event_times = torch.zeros([self.NN,1]).to(self.device)
            aux_state = self.prejump_hidden_states[0].unsqueeze(dim=1).to(self.device)
            t_embed = model.spatial_model.t_embedding(event_times) / math.sqrt(model.spatial_model.t_embedding_dim)
            aux_type = torch.zeros([self.NN,1,self.args.type_dim]).to(self.device)
            if self.args.weekhour:
                if self.args.data=='Tencent_deploy':
                    aux_week = model.week_emb((event_times / 24).long())
                    aux_hour = model.hour_emb(event_times.long())
                else:
                    aux_week = model.week_emb((event_times % 7).long())
                    aux_hour = model.hour_emb(((event_times % 1) * 24).long())
                inputs = [spatial_locations, aux_state, t_embed, aux_week, aux_hour, aux_type]
            else:
                inputs = [spatial_locations, aux_state, t_embed]
        else:
            maxlen = max([len(i) for i in self.spatial_locations])
            if isinstance(self.prejump_hidden_states,list):
                aux_state = torch.cat(self.prejump_hidden_states,dim=1).to(self.device) 
            else:
                aux_state = self.prejump_hidden_states[:].float()
                self.prejump_hidden_states = [self.prejump_hidden_states[:,index:index+1].to(self.device) for index in range(self.prejump_hidden_states.shape[1])]

            loc = [torch.cat([j.unsqueeze(dim=0) for j in i]+[i[-1].unsqueeze(dim=0) for _ in range(maxlen-len(i))], dim=0).unsqueeze(dim=0) for i in self.spatial_locations]
            spatial_locations = torch.cat(loc, dim=0).to(self.device)
            event_times = torch.tensor([i[:-1]+[i[-1] for _ in range(maxlen-len(i[:-1]))] for i in self.event_times]).to(self.device)
            t_embed = model.spatial_model.t_embedding(event_times) / math.sqrt(model.spatial_model.t_embedding_dim)
            event_types =  torch.tensor([i[:-1]+[i[-1] for _ in range(maxlen-len(i[:-1]))] for i in self.event_types]).to(self.device)
            if self.args.weekhour:
                if self.args.data=='Tencent_deploy':
                    aux_week = model.week_emb((event_times / 24).long())
                    aux_hour = model.hour_emb(event_times.long())
                else:
                    aux_week = model.week_emb((event_times % 7).long())
                    aux_hour = model.hour_emb(((event_times % 1) * 24).long())
                aux_type = model.type_emb(event_types)
                inputs = [spatial_locations, aux_state, t_embed, aux_week, aux_hour, aux_type]
            else:
                inputs = [spatial_locations, aux_state, t_embed]
        
        # attention layer uses (T, N, D) ordering.
        inputs = [inp.transpose(0, 1) for inp in inputs]
        norm_fn = max_rms_norm([a.shape for a in inputs])

        x = torch.cat(inputs, dim=-1)

        model.spatial_model.odefunc.set_shape(x.shape)

        N, T, D = spatial_locations.shape

        x = x.reshape(T * N, -1).to(self.device).float()

        t0 = event_times.to(self.device).transpose(0,1).expand(T,N).reshape(-1) + model.spatial_model.time_offset
        t1 = torch.tensor([[i[-1]] for i in self.event_times]).to(self.device).transpose(0,1).expand(T,N).reshape(-1) + model.spatial_model.time_offset

        z, _ = model.spatial_model.cnf.integrate(t0, t1, x, torch.zeros_like(t0), norm=norm_fn)

        z = z[:, :model.spatial_model.dim]  # (T * N, D)

        z = z.reshape(T,N,D)[-1,:,:]

        return z

    def state_init(self, model, event_times, spatial_locations, event_types):

        self.event_times, self.spatial_locations, self.event_types,  self.state_update, self.intensity, self.intensity_event, self.state_save, self.init_state_save, self.x_base, self.prejump_hidden_states = event_times, spatial_locations, event_types,  [[] for _ in range(self.NN)], None, None, [[] for _ in range(self.NN)], None, None, []

        self.x_base = 1
        
        input_mask = torch.ones([len(event_times),1]).to(self.device)

        init_state = (torch.randn([len(event_times),model.temporal_model.hidden_dims[0]]) / math.sqrt(model.temporal_model.hidden_dims[0])).to(self.device)

        t0 = torch.tensor(0.0).to(self.device)
        t1 = None

        type_emb = model.type_emb(torch.tensor(event_types).long().to(self.device))

        intensities, Lambda, hidden_states, prob_event, tpp_states, _ = model.temporal_model.integrate_lambda(init_state, torch.tensor(event_times).to(self.device), torch.cat([i[0].unsqueeze(dim=0).unsqueeze(dim=0) for i in spatial_locations],dim=0).to(self.device), type_emb, input_mask, t0, t1, model.week_emb, model.hour_emb)

        hidden_states = hidden_states[:, 1:]

        self.prejump_hidden_states = hidden_states

        state = [Lambda, tpp_states[:,-1,:]]

        self.state_update = [torch.tensor(event_times).squeeze(dim=1).to(self.device), state]

        self.intensity = model.temporal_model.get_intensity(tpp_states[:,-1,:])

        self.intensity_event = model.temporal_model.get_intensity_event(tpp_states[:,-1,:])

        self.mask_t_max = torch.tensor(0.0).to(self.device).expand(self.NN).clone()

        time_current = torch.tensor([i[-1] for i in self.event_times]).to(self.device)

        self.mask_t_max = time_current>=self.args.t_max


    def sample_seqs(self, model, seq_num, event_times, spatial_locations, event_types):

        self.NN = seq_num

        self.state_init(model, event_times, spatial_locations, event_types)
        
        t_temp = True

        while t_temp:
            temp = self.sample_one_event_attn(model, True)
            t_temp = False if temp.all() else True

        seqs = [[(self.event_times[i][j], self.event_types[i][j], self.spatial_locations[i][j].detach().cpu().numpy().tolist()) for j in range(len(self.event_times[i])) if self.event_times[i][j]<self.args.t_max] for i in range(self.NN)]


        return seqs

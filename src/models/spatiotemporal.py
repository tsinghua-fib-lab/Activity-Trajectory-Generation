# Copyright (c) Facebook, Inc. and its affiliates.

from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn
from models.spatial import SelfAttentiveCNF,JumpCNF
from models.temporal import NeuralPointProcess
from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint_event
import math


class SpatiotemporalModel(nn.Module, metaclass=ABCMeta):

    @abstractmethod
    def forward(self, event_times, spatial_locations, input_mask, t0, t1):
        """
        Args:
            event_times: (N, T)
            spatial_locations: (N, T, D)
            input_mask: (N, T)
            t0: () or (N,)
            t1: () or (N,)
        """
        pass

    @abstractmethod
    def spatial_conditional_logprob_fn(self, t, event_times, spatial_locations, t0, t1):
        pass


class CombinedSpatiotemporalModel(SpatiotemporalModel):

    def __init__(self, spatial_model, temporal_model):
        super().__init__()
        self.spatial_model = spatial_model
        self.temporal_model = temporal_model

    def forward(self, event_times, spatial_locations, input_mask, t0, t1):
        space_loglik = self._spatial_logprob(event_times, spatial_locations, input_mask)
        time_loglik = self._temporal_logprob(event_times, spatial_locations, input_mask, t0, t1)
        return space_loglik, time_loglik

    def spatial_conditional_logprob_fn(self, t, event_times, spatial_locations, t0, t1):
        return self.spatial_model.spatial_conditional_logprob_fn(t, event_times, spatial_locations)

    def _spatial_logprob(self, event_times, spatial_locations, input_mask):
        return self.spatial_model.logprob(event_times, spatial_locations, input_mask)

    def _temporal_logprob(self, event_times, spatial_locations, input_mask, t0, t1):
        return self.temporal_model.logprob(event_times, spatial_locations, input_mask, t0, t1)


class SharedHiddenStateSpatiotemporalModel(SpatiotemporalModel, metaclass=ABCMeta):

    def __init__(self, dim=2, hidden_dims=[64, 64, 64], tpp_hidden_dims=[8, 20], tpp_cond=False, tpp_style="gru",
                 actfn="softplus", tpp_actfn="softplus", zero_init=True, share_hidden=False, solve_reverse=False, tpp_otreg_strength=0.0, tol=1e-6, args=None,**kwargs):
        super().__init__()
        if args.weekhour:
            dim = 2 + args.type_dim + args.week_dim + args.hour_dim
        else:
            dim = 2 + args.type_dim 
        self.hidden_dims = hidden_dims
        self.args = args
        tpp_hidden_dims = [h for h in tpp_hidden_dims]

        self.temporal_model = NeuralPointProcess(
            cond_dim=dim, hidden_dims=tpp_hidden_dims, cond=tpp_cond, style=tpp_style, actfn=tpp_actfn, hdim=tpp_hidden_dims[0] // 2,
            separate=2 if not share_hidden else 1, tol=args.tol, otreg_strength=tpp_otreg_strength,args=args)

        dim = 2
        self._build_spatial_model(dim, hidden_dims, actfn, zero_init, aux_dim=tpp_hidden_dims[0] // 2,
                                  aux_odefunc=self.temporal_model.hidden_state_dynamics if solve_reverse else zero_diffeq,
                                  tol=tol, args=args, **kwargs)

        self.type_emb = nn.Embedding(
                num_embeddings=self.args.num_event, embedding_dim=self.args.type_dim, scale_grad_by_freq=False, sparse=False)


        if args.weekhour:
            self.week_emb = nn.Embedding(
                    num_embeddings=7, embedding_dim=self.args.week_dim, scale_grad_by_freq=False, sparse=False)

            self.hour_emb = nn.Embedding(
                    num_embeddings=24, embedding_dim=self.args.hour_dim, scale_grad_by_freq=False, sparse=False)

        else:
            self.week_emb, self.hour_emb = None, None

    @abstractmethod
    def _build_spatial_model(self, dim, hidden_dims, actfn, zero_init, aux_dim, aux_odefunc, **kwargs):
        pass

    def forward(self, init_time, event_times, spatial_locations, event_types, input_mask, t0, t1):

        type_emb = self.type_emb(event_types.long())

        intensities, Lambda, hidden_states, prob_event, tpp_state, Lambdas = self.temporal_model.integrate_lambda(init_time, event_times, spatial_locations, type_emb, input_mask, t0, t1, self.week_emb, self.hour_emb, 1)

        #print('intensity',intensities.shape,Lambda.shape)
        
        time_loglik_NT = torch.sum(torch.log(intensities + 1e-8) * input_mask, dim=1) - Lambda

        time_loglik = time_loglik_NT.sum()

        #time_origin = time_loglik.clone()

        for i in range(event_times.shape[0]):
            for j in range(event_times.shape[1]):
                time_loglik += torch.log(prob_event[i,j,event_types[i,j].item()] + 1e-8) * input_mask[i,j]

        #print('temporal finished!')

        if t1 is not None:
            hidden_states = hidden_states[:, 1:-1]  # Remove first (t=t0) and last (t=t1) hidden states. 

        else:
            hidden_states = hidden_states[:, 1:] 

        space_loglik = self.spatial_model.logprob(event_times, spatial_locations, type_emb, input_mask, aux_state=hidden_states,week_emb = self.week_emb,hour_emb = self.hour_emb)
        #print('space finished!')

        return space_loglik, time_loglik, time_loglik_NT, prob_event, tpp_state, (intensities, Lambdas)
         #(N, T, num_event)

    def spatial_conditional_logprob_fn(self, t, event_times, spatial_locations, t0, t1):
        print('error!')
        hidden_state_times = torch.cat([event_times, torch.tensor(t).reshape(-1).to(event_times)]).reshape(1, -1)
        _, _, hidden_states = self.temporal_model.integrate_lambda(hidden_state_times, spatial_locations[None], input_mask=None, t0=t0, t1=None)
        hidden_states = hidden_states[:, 1:]  # Remove first (t=t0) hidden state.
        return self.spatial_model.spatial_conditional_logprob_fn(t, event_times, spatial_locations, aux_state=hidden_states)

    def vector_field_fn(self, t, event_times, spatial_locations, t0, t1):
        print('error!')
        hidden_state_times = torch.cat([event_times, torch.tensor(t).reshape(-1).to(event_times)]).reshape(1, -1)
        _, _, hidden_states = self.temporal_model.integrate_lambda(hidden_state_times, spatial_locations[None], input_mask=None, t0=t0, t1=None)
        hidden_states = hidden_states[0, 1:]  # Remove first (t=t0) hidden state.
        return self.spatial_model.vector_field_fn(t, event_times, spatial_locations, aux_state=hidden_states)

    def sample_spatial(self, nsamples, event_times, spatial_locations, input_mask, t0, t1):
        print('error!')
        intensities, Lambda, hidden_states = self.temporal_model.integrate_lambda(event_times, spatial_locations, input_mask, t0, t1)
        hidden_states = hidden_states[:, 1:-1]  # Remove first (t=t0) and last (t=t1) hidden states.
        samples = self.spatial_model.sample_spatial(nsamples, event_times, spatial_locations, input_mask, aux_state=hidden_states)
        return samples


class JumpCNFSpatiotemporalModel(SharedHiddenStateSpatiotemporalModel):

    def _build_spatial_model(self, dim, hidden_dims, actfn, zero_init, aux_dim, aux_odefunc, args, **kwargs):
        self.spatial_model = JumpCNF(
            dim=dim, hidden_dims=hidden_dims, actfn=actfn, zero_init=zero_init, aux_dim=aux_dim, aux_odefunc=aux_odefunc, args=args, **kwargs,
        )


class SelfAttentiveCNFSpatiotemporalModel(SharedHiddenStateSpatiotemporalModel):

    def _build_spatial_model(self, dim, hidden_dims, actfn, zero_init, aux_dim, aux_odefunc, args, **kwargs):
        self.spatial_model = SelfAttentiveCNF(dim=dim, hidden_dims=hidden_dims, actfn=actfn, zero_init=zero_init, aux_dim=aux_dim, args = args, **kwargs)


class SelfAttentiveCNFSpatiotemporalModel2(SharedHiddenStateSpatiotemporalModel):

    def _build_spatial_model(self, dim, hidden_dims, actfn, zero_init, aux_dim, aux_odefunc, args, **kwargs):
        self.spatial_model = SelfAttentiveCNF2(dim=dim, hidden_dims=hidden_dims, actfn=actfn, zero_init=zero_init, aux_dim=aux_dim, args = args, **kwargs)


class JumpGMMSpatiotemporalModel(SharedHiddenStateSpatiotemporalModel):

    def _build_spatial_model(self, dim, hidden_dims, actfn, zero_init, aux_dim, aux_odefunc, n_mixtures=5, **kwargs):
        self.spatial_model = ConditionalGMM(dim=dim, hidden_dims=hidden_dims, actfn=actfn, aux_dim=aux_dim, n_mixtures=n_mixtures)


def zero_diffeq(t, h):
    return torch.zeros_like(h)

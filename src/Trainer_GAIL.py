import os
import mlflow
import time
import itertools
import math
import json

import torch
import random

from tqdm import tqdm
from sklearn.metrics import accuracy_score

import datasets
from iterators import EpochBatchIterator
from models import CombinedSpatiotemporalModel, JumpCNFSpatiotemporalModel, SelfAttentiveCNFSpatiotemporalModel, SelfAttentiveCNFSpatiotemporalModel2, JumpGMMSpatiotemporalModel
from models.spatial import JumpCNF, SelfAttentiveCNF
from models.spatial.cnf import TimeVariableCNF
from models.temporal import NeuralPointProcess
from models.temporal.neural import ACTFNS as TPP_ACTFNS
from models.temporal.neural import TimeVariableODE
from models.GAIL import Policy_net, Value_net, Discriminator, RolloutStorage, Base
from models.generator import *
from Evaluation import *
from PreTrainer import PreTrainer

import utils


class Trainer_GAIL(object):
    def __init__(self, args,device,TIME, pretrainer = None):

        self.pretrainer = pretrainer

        self.args = args

        self.device = device

        self.model_path = './ModelSave/{}/'.format(TIME)

        if not os.path.exists('./ModelSave/'):
                os.mkdir('./ModelSave/')

        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)


    def model_init(self):

        if self.pretrainer is not None:
            self.pretrainer.model_init()

        self.policy_net = Policy_net(self.args, self.device, self.pretrainer.model, self.pretrainer.generator).to(self.device)

        BaseModel = Base(self.args,self.policy_net.STPP_model.temporal_model.hidden_dims[0])

        self.evaluation = Evaluation(self.args)

        self.value_net = Value_net(self.args,self.device,self.pretrainer.model, BaseModel).to(self.device)

        self.discriminator = Discriminator(self.args, self.device, self.pretrainer.model, BaseModel).to(self.device)

        self.buffer = RolloutStorage(self.args)

        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.args.policy_lr, weight_decay=self.args.weight_decay)

        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(),lr = 1e-3, weight_decay = self.args.weight_decay)

        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(),lr = self.args.value_lr,weight_decay = self.args.weight_decay)

        self.value_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.value_optimizer, mode='min', factor=0.1, patience=5, min_lr=3e-4)
    
        # loss
        self.dis_loss = nn.BCELoss(reduction='mean').to(self.device)


    def get_reward(self, state, action):

        d_reward = self.discriminator(state, action)

        log_reward = d_reward.log()

        if log_reward.shape[0]==1:
            return log_reward.detach().item() 
        else:
            return log_reward.cpu().detach().sum()
    
    def spatiotemporal_pad(self,event_times, spatial_locations, event_types, state_save):
        """Input is a list of tensors with shape (T, 1 + D)
            where T may be different for each tensor.

        Returns:
            event_times: (N, max_T)
            spatial_locations: (N, max_T, D)
            event_types: (N, max_T)
            ...
            mask: (N, max_T)
        """

        def pad(x,max_len,item=0):

            return [[s+ [[item for _ in range(len(s[0]))]] * (max_len-len(s))][0] if len(s)<max_len else s[:max_len] for s in x]

        lengths = [len(seq) for seq in event_times]
        max_len = min(max(lengths),15)
        event_times = pad(self.ListUnsqueeze(event_times),max_len,self.args.t_max)

        event_types = pad(self.ListUnsqueeze(event_types),max_len)

        spatial_locations = pad(spatial_locations,max_len,0.01)

        Lambda = [[[s[0]] for s in seq] for seq in state_save]
        tpp_state = [[s[1] for s in seq] for seq in state_save]

        Lambda = pad(Lambda,max_len)

        tpp_state = pad(tpp_state,max_len)
        state_save = [[[Lambda[i1][i2], tpp_state[i1][i2]] for i2 in range(max_len)] for i1 in range(len(Lambda))]

        mask = torch.stack([torch.cat([torch.ones(seq_len), torch.zeros(max_len - seq_len)], dim=0) if seq_len < max_len else torch.ones(max_len) for seq_len in lengths])

        return event_times, spatial_locations, event_types, state_save, mask

    def ListUnsqueeze(self, x):
        return [[[i] for i in seq] for seq in x]

    def simulation_validate(self,gen_data,test_data):
        distance_jsd, radius_jsd, duration_jsd, dailyloc_jsd, distance_step_jsd = self.evaluation.get_JSD(gen_data,test_data)

        print("############### Simulation and JSD Evaluation! ###############")
        print("distance_jsd: {}, radius_jsd: {}, distance_step_jsd: {}, duration_jsd: {}, dailyloc_jsd: {}".format(distance_jsd, radius_jsd, distance_step_jsd, duration_jsd, dailyloc_jsd))
    

    def simulate(self,model,num=200):

        model.eval()

        self.args.sim_batch_test = min(self.args.sim_batch_test, num)

        sim_data = []
        with torch.no_grad():
            for index in tqdm(range(int(num/self.args.sim_batch_test))):
                sim_data += self.policy_net.generator.sample_seqs(model, self.args.sim_batch_test)
        model.train()

        return sim_data


    def sim_test(self):

        self.policy_net.STPP_model.eval()
        self.policy_net.STPP_model.spatial_model.cnf.training = False
        self.policy_net.STPP_model.spatial_model.base_cnf.training = False

        with torch.no_grad():
            
            gen_data = self.simulate(self.policy_net.STPP_model,self.args.sim_num)
            gen_data = [[(i[0],i[1],[i[2][0]*self.args.S_std[0][0].item()+self.args.S_mean[0][0].item(),i[2][1]*self.args.S_std[0][1].item()+self.args.S_mean[0][1].item()]) for i in u[1:]] for u in gen_data if len(u[1:])>9] # !!!

            self.simulation_validate(gen_data,self.test_set)

        self.policy_net.STPP_model.train()
        self.policy_net.STPP_model.spatial_model.cnf.training = True
        self.policy_net.STPP_model.spatial_model.base_cnf.training = True

        return 0

    def Train(self):

        t0,t1,train_epoch_iter,val_loader,test_loader,train_set, test_set = self.pretrainer.data_loader()

        self.t0, self.t1 = t0, t1
        self.train_set = train_set

        t1 = None

        test_data_iter = torch.utils.data.DataLoader(
            test_set,
            batch_size=self.args.test_bsz,
            shuffle=False,
            collate_fn=datasets.spatiotemporal_events_collate_fn,
        )

        self.test_set = [[(i[0].item(),int(i[3]), [i[1].item()*self.args.S_std[0][0].item()+self.args.S_mean[0][0].item(),i[2].item()*self.args.S_std[0][1].item()+self.args.S_mean[0][1].item()]) for i in u[1:] if i[0].item()<self.args.t_max] for u in test_set]

        self.test_set = [u for u in self.test_set if len(u)>0]
        
        self.model_init()

        test_data_init_state = [(torch.randn([i[0].shape[0],self.pretrainer.model.temporal_model.hidden_dims[0]]) / math.sqrt(self.pretrainer.model.temporal_model.hidden_dims[0])) for i in test_data_iter]
        
        reward_save = []

        for itr in range(self.args.sim_iterations):

            with torch.no_grad():
                print('\nsimulate start!\n')

                self.policy_net.STPP_model.spatial_model.cnf.training = False
                self.policy_net.STPP_model.spatial_model.base_cnf.training = False

                event_times, event_types, spatial_locations, state_save, init_state_save = self.policy_net.act(self.args.sim_batch)

                num_min = 1

                state_save = [i for index, i in enumerate(state_save) if len(event_times[index])>num_min]
                init_state_save = torch.cat([i.unsqueeze(dim=0) for index, i in enumerate(init_state_save) if len(event_times[index])>num_min],dim=0).numpy()
                event_types = [i for index, i in enumerate(event_types) if len(event_times[index])>num_min]
                spatial_locations = [i for index, i in enumerate(spatial_locations) if len(event_times[index])>num_min]
                event_times = [i for index, i in enumerate(event_times) if len(i)>num_min]

                self.args.gen_batch = len(event_times)

                self.policy_net.STPP_model.spatial_model.cnf.training = True
                self.policy_net.STPP_model.spatial_model.base_cnf.training = True

                event_times, spatial_locations, event_types, state_save, mask = self.spatiotemporal_pad(event_times, spatial_locations, event_types, state_save)

                N, T = mask.shape

                Intervals = [[[s[0]-seq[index][0]] for index, s in enumerate(seq[1:])] for seq in event_times]

                tpp_state = [[s[1] for s in seq] for seq in state_save]

                state = [torch.tensor(event_times).float().to(self.device)[:,:-1], torch.tensor(event_types).long().to(self.device)[:,:-1], torch.tensor(spatial_locations).float().to(self.device)[:,:-1], torch.tensor(tpp_state).float().to(self.device)[:,:-1]]

                action = [torch.tensor(Intervals).float().to(self.device), torch.tensor(event_types).long().to(self.device)[:,1:], torch.tensor(spatial_locations).float().to(self.device)[:,1:]]

                values = (self.policy_net.get_value(state, self.value_net)*mask[:,:-1].to(self.device)).detach().cpu().numpy()

                reward = (self.discriminator(state,action).log()*mask[:,:-1].to(self.device)).detach().cpu().numpy()

                inputs = [torch.tensor(event_times).float().to(self.device), torch.tensor(event_types).long().to(self.device), torch.tensor(spatial_locations).float().to(self.device), torch.tensor(init_state_save).float().to(self.device), mask.to(self.device)]

                action_log_probs, dist_entropy = self.policy_net.evaluate_actions(inputs) 

                done = (1-mask[:,1:]).numpy() 

                if itr!=0:
                    reward_save += [i for u_index, u in enumerate(reward) for index, i in enumerate(u) if done[u_index][index]==0]

                state = [torch.tensor(event_times).float().to(self.device), torch.tensor(event_types).long().to(self.device), torch.tensor(spatial_locations).float().to(self.device), torch.tensor(tpp_state).float().to(self.device)] 

                state = [i.detach().cpu().numpy() for i in state]
                action = [i.detach().cpu().numpy() for i in action]

                num_max = 20

                state = [np.hstack((i, np.zeros([N,num_max-i.shape[1],i.shape[2]]))) for i in state]

                action = [np.hstack((i, np.zeros([N,num_max-i.shape[1],i.shape[2]]))) for i in action]

                reward = np.hstack((reward, np.zeros([N,num_max-reward.shape[1]])))

                done = np.hstack((done,np.ones([N,num_max-done.shape[1]]))) # ones!
                
                values = np.hstack((values,np.zeros([N,num_max-values.shape[1]])))

                action_log_probs = action_log_probs.detach().cpu().numpy()

                action_log_probs = np.hstack((action_log_probs, np.zeros([N, num_max-action_log_probs.shape[1]])))

                dist_entropy = dist_entropy.detach().cpu().numpy()

                dist_entropy = np.hstack((dist_entropy, np.zeros([N, num_max-dist_entropy.shape[1]])))
                
                self.buffer.store([state, action, init_state_save,reward, done, values, action_log_probs, dist_entropy])

            if itr % self.args.simfreq == 0:
                print('\n#################### Validate! ####################')
                self.sim_test()
                print('Model Save!')
                torch.save(self.policy_net.STPP_model.state_dict(), self.model_path+'model_{}.pkl'.format(itr))

            if itr % self.args.iter_disc == 0:
                print('\n#################### discriminator training! ####################')
                if itr==0:
                    self.discriminator_train(t0,t1,1)
                else:
                    self.discriminator_train(t0,t1,self.args.iter_disc)
                self.buffer.clear_dis()
                if itr==0:
                    self.discriminator_optimizer.param_groups[0]['lr'] = self.args.dis_lr

            if self.args.separate and itr % self.args.iter_enc==0:
                print('\n#################### encoder training! ####################')
                self.enc_train()

            if itr % self.args.iter_ppo == 0:
                print('\n#################### policy training! ####################')
                reward_save = []
                self.buffer.compute_returns()
                self.ppo_train()
                self.buffer.clear()


    def discriminator_train(self, t0, t1=None, iter_disc=-1):

        batch_seq = self.args.sim_batch * self.args.iter_disc

        train_set = random.sample(list(self.train_set), batch_seq)

        real_data = torch.utils.data.DataLoader(
            train_set,
            batch_size=10000,
            shuffle=False,
            collate_fn=datasets.spatiotemporal_events_collate_fn,
        )

        state_fake, action_fake, done_fake = self.buffer.dis_memory

        state_fake = [torch.tensor(state_fake[0]).float().to(self.device), torch.tensor(state_fake[1]).long().to(self.device), torch.tensor(state_fake[2]).float().to(self.device), torch.tensor(state_fake[3]).float().to(self.device)]

        action_fake = [torch.tensor(action_fake[0]).float().to(self.device), torch.tensor(action_fake[1]).long().to(self.device),torch.tensor(action_fake[2]).float().to(self.device)]
        

        fake_mask = torch.tensor(1-done_fake).to(self.device)

        N_fake,T_fake = fake_mask.shape

        mask = fake_mask.clone().reshape(N_fake,T_fake)
        max_len = int(torch.sum(mask,dim=1).max().item())
        fake_mask = mask[:,:max_len]

        N, T = fake_mask.shape

        fake_mask = fake_mask.reshape(N*T)

        labels_fake = torch.FloatTensor(N*T,1).fill_(0).to(self.device)

        state_fake = [i[:,:T].detach() for i in state_fake]
        action_fake = [i[:,:T].detach() for i in action_fake]

        for idx, batch in enumerate(real_data):

            init_state = (torch.randn([batch_seq,self.pretrainer.model.temporal_model.hidden_dims[0]]) / math.sqrt(self.pretrainer.model.temporal_model.hidden_dims[0])).to(self.device)

            event_times, spatial_locations, event_types, real_mask = map(lambda x: utils.cast(x, self.device), batch)

            event_types = event_types.long()
            _, _, _, _, tpp_state, _ = self.pretrainer.model(init_state, event_times, spatial_locations, event_types, real_mask, t0, t1)

            tpp_state = tpp_state.detach() 

            real_mask = real_mask[:,1:]
            N_real, T_real = real_mask.shape
            real_mask = real_mask.reshape(N_real*T_real)

            for _ in range(self.args.disc_epoch):

                state_real = [event_times[:,:-1].detach().unsqueeze(dim=2), event_types[:,:-1].detach(), spatial_locations[:,:-1].detach(), tpp_state.detach()]
                Intervals = (event_times[:,1:]-event_times[:,:-1]).detach()
                action_real = [Intervals.clone(), event_types[:,1:].detach(), spatial_locations[:,1:].detach()]

                reward_fake = self.discriminator(state_fake, action_fake).reshape(N*T) * fake_mask

                reward_real = self.discriminator(state_real, action_real).reshape(N_real*T_real) * real_mask

                labels_real = torch.FloatTensor(N_real*T_real,1).fill_(1).to(self.device)

                reward = torch.cat((reward_fake.float(),reward_real.float()),dim=0).unsqueeze(dim=1)
                labels = torch.cat((labels_fake.float(),labels_real.float()),dim=0)
                mask = torch.cat((fake_mask.float(),real_mask.float()),dim=0)

                reward_new = torch.cat([i for index, i in enumerate(reward) if mask[index].item()==1], dim=0)
                labels_new = torch.cat([i for index, i in enumerate(labels) if mask[index].item()==1], dim=0)

                dis_loss = self.dis_loss(reward_new, labels_new) 

                pred = np.round(reward_new.cpu().detach())
                target = np.round(labels_new.cpu().detach())

                acc = accuracy_score(pred, target)

                if acc>0.8:
                    break
                
                self.discriminator_optimizer.zero_grad()
                dis_loss.backward()
                self.discriminator_optimizer.step()

                print('accuracy:{}'.format(acc))


    def enc_train(self):

        train_set = random.sample(list(self.train_set), self.args.sim_batch * self.args.iter_enc)

        data = torch.utils.data.DataLoader(
            train_set,
            batch_size=64,
            shuffle=False,
            collate_fn=datasets.spatiotemporal_events_collate_fn,
        )

        for p in self.policy_net.STPP_model.temporal_model.ode_solver.func.intensity_fn.parameters():
            p.requires_grad = False

        if self.args.spatial_policy:
            for p in self.policy_net.STPP_model.spatial_model.parameters():
                p.requires_grad = False
        
        self.policy_optimizer.param_groups[0]['lr'] = self.args.enc_lr

        start = time.time()

        t0, t1 = self.t0.clone(), self.t1.clone()

        init_state = [(torch.randn([d[0].shape[0], self.pretrainer.model.temporal_model.hidden_dims[0]]) / math.sqrt(self.pretrainer.model.temporal_model.hidden_dims[0])) for d in data]

        for _ in tqdm(range(self.args.enc_epoch)):

            space_loglik_all = 0.0
            time_loglik_all = 0.0
            num_all = 0.0

            for bid, batch in enumerate(data):
                
                event_times, spatial_locations, event_types, input_mask = map(lambda x: utils.cast(x, self.device), batch)

                event_types = event_types.long()
                num_events = input_mask.sum()

                space_loglik, time_loglik, _,_ ,_,_= self.policy_net.STPP_model(init_state[bid].to(self.device), event_times, spatial_locations, event_types, input_mask, t0, t1)

                space_loglik_all += space_loglik.sum().item()

                time_loglik_all += time_loglik.sum().item()

                num_all += num_events.item()

                space_loglik = space_loglik.sum() / num_events
                time_loglik = time_loglik.sum() / num_events

                loglik= time_loglik + space_loglik
                loss = loglik.mul(-1.0).mean()

                self.policy_optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=self.args.gradclip).item()
                self.policy_optimizer.step()

            space_loglik = space_loglik_all / num_all
            time_loglik = time_loglik_all / num_all
            loss_all = space_loglik + time_loglik
            
        end = time.time()
        print('backward time: {}'.format(round((end-start)/60,1)))

        for p in self.policy_net.STPP_model.parameters(): # 只让intensity有梯度
            p.requires_grad = True

        self.policy_optimizer.param_groups[0]['lr'] = self.args.policy_lr


    def test_loglik_validate(self, model, init_state, test_loader, t0, t1):

        model.eval()

        space_loglik_meter = utils.AverageMeter()
        time_loglik_meter = utils.AverageMeter()
        time_origin_loglik_meter = utils.AverageMeter()

        with torch.no_grad():
            for bid, batch in enumerate(test_loader):
                event_times, spatial_locations, event_types, input_mask = map(lambda x: utils.cast(x, self.device), batch)
                event_types = event_types.long()
                num_events = input_mask.sum().item()

                space_loglik, time_loglik, _,_ ,_,_= model(init_state[bid].to(self.device), event_times, spatial_locations, event_types, input_mask, t0, t1)

                space_loglik = space_loglik.sum().item() / num_events
                time_loglik = time_loglik.sum().item() / num_events

                space_loglik_meter.update(space_loglik, num_events)
                time_loglik_meter.update(time_loglik, num_events)

        print('validate_time:{}'.format(time_loglik_meter.avg))
        print('validate_space:{}'.format(space_loglik_meter.avg))
        model.train()

    def ppo_train(self):

        state, action, init_state_save,reward, done, _, action_log_probs, _, returns,advantages = self.buffer.memory

        N_value, T_value = done.shape

        mask_value = torch.tensor(1.0-done).to(self.device).reshape(N_value*T_value)

        old_action_probs = torch.tensor(action_log_probs).float().to(self.device)

        returns = torch.tensor(returns).float().to(self.device).reshape(N_value*T_value).detach()

        advantages = torch.tensor(advantages).float().to(self.device).reshape(N_value*T_value,1).detach()

        advantage_non_zero = torch.cat([advantages[i] for i in range(N_value*T_value) if mask_value[i].item()==1],dim=0)

        if self.args.adv_norm==1:
            advantages = (advantages.squeeze(dim=1) - advantage_non_zero.mean()) / (advantage_non_zero.std() + 1e-10) * mask_value
        
        state_value = [torch.tensor(state[0]).float().to(self.device), torch.tensor(state[1]).long().to(self.device), torch.tensor(state[2]).float().to(self.device), torch.tensor(state[3]).float().to(self.device)]      


        mask = mask_value.clone().reshape(N_value,T_value)
        max_len = int(torch.sum(mask,dim=1).max().item())
        mask = mask[:,:max_len]

        advantages = advantages.reshape(N_value,T_value)

        N, T = mask.shape

        old_action_probs = old_action_probs[:,:T].reshape(N*T).detach()
        advantages = advantages[:,:T].reshape(N*T)
        inputs = [state_value[0].clone()[:,:T+1], state_value[1].clone()[:,:T+1], state_value[2].clone()[:,:T+1], torch.tensor(init_state_save).float().to(self.device), torch.cat((torch.ones([N, 1]).to(self.device), mask[:,:T]), dim=1)] # 这里N+1要注意！

        mask = mask[:,:T].reshape(N*T).detach()

        if self.args.separate:
            print('separate!')
            for p in self.policy_net.STPP_model.parameters():
                p.requires_grad = False

            for p in self.policy_net.STPP_model.temporal_model.ode_solver.func.intensity_fn.parameters():
                p.requires_grad = True

            if self.args.spatial_policy:
                for p in self.policy_net.STPP_model.spatial_model.parameters():
                    p.requires_grad = True

        for _ in tqdm(range(self.args.ppo_epoch)):

            values = self.value_net(state_value)
            value_loss = 0.5 * ((returns-values.reshape(N_value*T_value))*mask_value).pow(2).sum()/mask_value.sum() # value是n-1时刻

            action_log_probs, action_dist_entropy = self.policy_net.evaluate_actions(inputs)

            action_log_probs = action_log_probs.reshape(N*T)
            action_dist_entropy = action_dist_entropy.reshape(N*T)

            if action_log_probs.shape != old_action_probs.shape:
                print('shape error！')
                print(action_log_probs.shape,old_action_probs.shape)
                exit()

            if self.args.ratio_exp:
                ratio = torch.exp(torch.clamp((action_log_probs-old_action_probs) * mask, max=self.args.max_ratio))

            else:
                ratio = (action_log_probs-old_action_probs) * mask

            assert ratio.shape==advantages.shape

            surr2 = torch.clamp(ratio, 1. - self.args.clip_param, 1. + self.args.clip_param,) * advantages * mask
            surr1 = ratio * advantages * mask 

            action_loss = -torch.min(surr1, surr2).sum()/mask.sum()

            loss = action_loss + value_loss - (mask * action_dist_entropy).sum()/mask.sum() * self.args.entropy_coef

            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()

            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(),self.args.clip_grad)

            self.policy_optimizer.step()
            self.value_optimizer.step()

        for p in self.policy_net.STPP_model.parameters():
            p.requires_grad = True        
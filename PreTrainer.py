import os
import mlflow
import time
import itertools
import math
import json

import torch

from tqdm import tqdm

import datasets
from iterators import EpochBatchIterator
from models import CombinedSpatiotemporalModel, JumpCNFSpatiotemporalModel, SelfAttentiveCNFSpatiotemporalModel, SelfAttentiveCNFSpatiotemporalModel2, JumpGMMSpatiotemporalModel
from models.spatial import JumpCNF, SelfAttentiveCNF
from models.spatial.cnf import TimeVariableCNF
from models.temporal import HomogeneousPoissonPointProcess, HawkesPointProcess, SelfCorrectingPointProcess, NeuralPointProcess
from models.temporal.neural import ACTFNS as TPP_ACTFNS
from models.temporal.neural import TimeVariableODE
from models.generator import *
from Evaluation import *

import utils


def load_data(data, split="train",num=10000,args=None):
    if data =='Foursquare':
        return datasets.Foursquare(split=split,num=num, args=args)
    elif data == "Mobile":
        return datasets.Mobile(split=split,num=num, args=args)
    else:
        raise ValueError(f"Unknown data option {data}")

class PreTrainer(object):
    def __init__(self, args,device,TIME):

        self.args = args
        self.device = device
        self.model_path = './ModelSave/{}/'.format(TIME)

        self.pretrain_path = './PretrainModel/'

        self.sim_path = './ModelSave/{}/simulate_data/'.format(TIME)

        if not os.path.exists('./ModelSave/'):
            os.mkdir('./ModelSave/')

        if not os.path.exists(self.pretrain_path):
            os.mkdir(self.pretrain_path)

        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        if not os.path.exists(self.sim_path):
            os.mkdir(self.sim_path)


    def model_init(self):

        x_dim=2

        self.generator = SeqGen(self.args,self.device)

        self.evaluation = Evaluation(self.args)

        if self.args.model == "jumpcnf" and self.args.tpp == "neural":
            self.model = JumpCNFSpatiotemporalModel(dim=x_dim,
                hidden_dims=list(map(int, self.args.hdims.split("-"))),
                tpp_hidden_dims=list(map(int, self.args.tpp_hdims.split("-"))),
                actfn=self.args.actfn,
                tpp_cond=self.args.tpp_cond,
                tpp_style=self.args.tpp_style,
                tpp_actfn=self.args.tpp_actfn,
                share_hidden=self.args.share_hidden,
                solve_reverse=self.args.solve_reverse,
                tol=self.args.tol,
                otreg_strength=self.args.otreg_strength,
                tpp_otreg_strength=self.args.tpp_otreg_strength,
                layer_type=self.args.layer_type,
                args=self.args).to(self.device)

        if self.args.model == "attncnf" and self.args.tpp == "neural":
            self.model = SelfAttentiveCNFSpatiotemporalModel(dim=x_dim,
                hidden_dims=list(map(int, self.args.hdims.split("-"))),
                tpp_hidden_dims=list(map(int, self.args.tpp_hdims.split("-"))),
                actfn=self.args.actfn,
                tpp_cond=self.args.tpp_cond,
                tpp_style=self.args.tpp_style,
                tpp_actfn=self.args.tpp_actfn,
                share_hidden=self.args.share_hidden,
                solve_reverse=self.args.solve_reverse,
                l2_attn=self.args.l2_attn,
                tol=self.args.tol,
                otreg_strength=self.args.otreg_strength,
                tpp_otreg_strength=self.args.tpp_otreg_strength,
                layer_type=self.args.layer_type,
                lowvar_trace=not self.args.naive_hutch,
                args=self.args).to(self.device)

        if self.args.model == "attncnf2" and self.args.tpp == "neural":
            self.model = SelfAttentiveCNFSpatiotemporalModel2(dim=x_dim,
                hidden_dims=list(map(int, self.args.hdims.split("-"))),
                tpp_hidden_dims=list(map(int, self.args.tpp_hdims.split("-"))),
                actfn=self.args.actfn,
                tpp_cond=self.args.tpp_cond,
                tpp_style=self.args.tpp_style,
                tpp_actfn=self.args.tpp_actfn,
                share_hidden=self.args.share_hidden,
                solve_reverse=self.args.solve_reverse,
                l2_attn=self.args.l2_attn,
                tol=self.args.tol,
                otreg_strength=self.args.otreg_strength,
                tpp_otreg_strength=self.args.tpp_otreg_strength,
                layer_type=self.args.layer_type,
                lowvar_trace=not self.args.naive_hutch,
                args=self.args).to(self.device)

        params = []
        attn_params = []
        for name, p in self.model.named_parameters():
            if "self_attns" in name:
                attn_params.append(p)
            else:
                params.append(p)

        self.optimizer = torch.optim.AdamW([
            {"params": params},
            {"params": attn_params}
        ], lr=self.args.lr, weight_decay=self.args.weight_decay, betas=(0.9, 0.98))


    def validate(self, model, init_state, test_loader, t0, t1):

        model.eval()

        space_loglik_meter = utils.AverageMeter()
        time_loglik_meter = utils.AverageMeter()
        time_origin_loglik_meter = utils.AverageMeter()

        with torch.no_grad():
            for idx, batch in enumerate(test_loader):
                event_times, spatial_locations, event_types, input_mask = map(lambda x: utils.cast(x, self.device), batch)
                event_types = event_types.long()
                num_events = input_mask.sum()

                space_loglik, time_loglik, _,_ ,_,_= model(init_state[idx], event_times, spatial_locations, event_types, input_mask, t0, t1)

                space_loglik = space_loglik.sum() / num_events
                time_loglik = time_loglik.sum() / num_events

                space_loglik_meter.update(space_loglik.item(), num_events)
                time_loglik_meter.update(time_loglik.item(), num_events)

        model.train()

        return space_loglik_meter.avg, time_loglik_meter.avg, time_origin_loglik_meter.avg

    def data_loader(self,num=None):
        t0, t1 = map(lambda x: utils.cast(x, self.device), utils.get_t0_t1(self.args.data))

        train_set = load_data(self.args.data, split="train",num = self.args.num_data,args=self.args)
        val_set = load_data(self.args.data, split="val",num = self.args.num_data,args=self.args)
        test_set = load_data(self.args.data, split="test",num = self.args.num_data,args=self.args)

        train_epoch_iter = EpochBatchIterator(
        dataset=train_set,
        collate_fn=datasets.spatiotemporal_events_collate_fn,
        batch_sampler=train_set.batch_by_size(self.args.max_events),
        seed=self.args.seed,
        )

        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=self.args.test_bsz,
            shuffle=False,
            collate_fn=datasets.spatiotemporal_events_collate_fn,
        )

        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=self.args.test_bsz,
            shuffle=False,
            collate_fn=datasets.spatiotemporal_events_collate_fn,
        )

        return t0,t1,train_epoch_iter,val_loader,test_loader, train_set, test_set

    def cosine_decay(self, learning_rate, global_step, decay_steps, alpha=0.0):
        global_step = min(global_step, decay_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * global_step / decay_steps))
        decayed = (1 - alpha) * cosine_decay + alpha

        return learning_rate * decayed

    def learning_rate_schedule(self, global_step, warmup_steps, base_learning_rate, train_steps):
        warmup_steps = int(round(warmup_steps))
        scaled_lr = base_learning_rate

        if warmup_steps:
            learning_rate = global_step / warmup_steps * scaled_lr
        else:
            learning_rate = scaled_lr

        if global_step < warmup_steps:
            learning_rate = learning_rate
        else:
            learning_rate = self.cosine_decay(scaled_lr, global_step - warmup_steps, train_steps - warmup_steps)
    
        return learning_rate


    def set_learning_rate(self, optimizer, lr):
        for i, group in enumerate(optimizer.param_groups):
            group['lr'] = lr



    def PreTrain(self):

        t0, t1, train_epoch_iter, val_loader, test_loader, train_set, test_set  = self.data_loader()
        

        self.test_set = [[(i[0].item(), int(i[3]), [i[1].item()*self.args.S_std[0][0].item()+self.args.S_mean[0][0].item(),i[2].item()*self.args.S_std[0][1].item()+self.args.S_mean[0][1].item()]) for i in u if i[0].item()<self.args.t_max] for u in test_set]

        print('t0:{}, t1:{}'.format(t0.item(),t1.item()))

        test_set = [[(i[0].item(),int(i[3]), [i[1].item(),i[2].item()]) for i in u] for u in test_set]

        self.model_init()

        init_state_test = [(torch.randn([len(batch[0]),self.model.temporal_model.hidden_dims[0]]) / math.sqrt(self.model.temporal_model.hidden_dims[0])).to(self.device) for batch in test_loader]
        
        begin_itr = 0
        
        self.model.train()

        start_time = time.time()

        iteration_counter = itertools.count(begin_itr)
        begin_epoch = begin_itr // len(train_epoch_iter)

        early_stop_cnt = 0

        minor_increase = 0

        max_time_loglik, max_space_loglik = -1e9, -1e9

        batch_iter = train_epoch_iter.next_epoch_itr(shuffle=False)
        
        init_state = [(torch.randn([len(batch[0]),self.model.temporal_model.hidden_dims[0]]) / math.sqrt(self.model.temporal_model.hidden_dims[0])).to(self.device) for batch in batch_iter]

        grad_norm = 0.0


        for epoch in range(begin_epoch, math.ceil(self.args.num_iterations / len(train_epoch_iter))):

            start = time.time()

            space_loglik_meter = utils.AverageMeter()
            time_loglik_meter = utils.AverageMeter()
            time_origin_loglik_meter = utils.AverageMeter()
            gradnorm_meter = utils.AverageMeter()

            batch_iter = train_epoch_iter.next_epoch_itr(shuffle=False)

            if epoch == 0 and self.args.warmup_itrs:
                self.args.warmup_itrs = int(len(batch_iter) * 5) # warmup的步数
                print('warmup_iters: {}'.format(self.args.warmup_itrs))

            for idx, batch in enumerate(batch_iter):

                itr = next(iteration_counter)

                self.optimizer.zero_grad()
    
                event_times, spatial_locations, event_types, input_mask = map(lambda x: utils.cast(x, self.device), batch)
                event_types = event_types.long()
                num_events = input_mask.sum().double()

                N, T = input_mask.shape 

                if num_events == 0:
                    raise RuntimeError("Got batch with no observations.") 

                print('training start!')

                space_loglik, time_loglik, time_origin_loglik, _,_,_ = self.model(init_state[idx],event_times, spatial_locations, event_types, input_mask, t0, t1)

                space_loglik = space_loglik.sum() / num_events

                time_loglik = time_loglik.sum() / num_events

                time_loglik_meter.update(time_loglik.item())

                time_loglik *= self.args.time_coef

                loglik =  time_loglik + space_loglik

                space_loglik_meter.update(space_loglik.item())

                loss = loglik.mul(-1.0).mean() # maximize loglikelihood == minimize -loglikelihood

                print('backward start!')
                loss.backward()

                # Set learning rate
                if epoch <=5:
                    total_itrs = math.ceil(self.args.num_iterations / len(train_epoch_iter)) * len(train_epoch_iter)
                    lr = self.learning_rate_schedule(itr, self.args.warmup_itrs, self.args.lr, total_itrs)
                    self.set_learning_rate(self.optimizer, lr)
                
                params = []
                grads = []
                for param in self.model.parameters():
                    params.append(param.reshape(-1))

                params_norm = torch.norm(torch.cat(params,dim=0), 2).item()
                
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.gradclip).item()
                gradnorm_meter.update(grad_norm)

                self.optimizer.step()


                print(f"Iter {itr} | Epoch {epoch}"
                        f" | Temporal {time_loglik_meter.val:.4f}({time_loglik_meter.avg:.4f})"
                        f" | Spatial {space_loglik_meter.val:.4f}({space_loglik_meter.avg:.4f})"
                        f" | GradNorm {gradnorm_meter.val:.2f}({gradnorm_meter.avg:.2f})")

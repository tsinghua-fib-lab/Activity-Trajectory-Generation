from collections import deque
import random
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import mlflow
import copy

class RolloutStorage(object):
    def __init__(self, args):
        self.args = args
        self.memory = [None for _ in range(8)]
        self.dis_memory = [None for _ in range(3)]
    
    def store(self, inputs):

        #state, action, init_state_save,reward, done, values, action_log_probs, dist_entropy = inputs

        for i in range(len(inputs)):

            if self.memory[i] is None:
                self.memory[i] = copy.deepcopy(inputs[i])
            else:
                if i in [0,1]:
                    for j in range(len(self.memory[i])):
                        self.memory[i][j] = np.vstack((self.memory[i][j], inputs[i][j]))
                        
                else:
                    self.memory[i] = np.vstack((self.memory[i],inputs[i]))

    
            if i in [0,1]:
                if self.dis_memory[i] is None:
                    self.dis_memory[i] = copy.deepcopy(inputs[i])
                else:
                    for j in range(len(self.dis_memory[i])):
                        self.dis_memory[i][j] = np.vstack((self.dis_memory[i][j], inputs[i][j]))

            if i==4:
                if self.dis_memory[2] is None:
                    self.dis_memory[2] = copy.deepcopy(inputs[i])
                else:
                    self.dis_memory[2] = np.vstack((self.dis_memory[2],inputs[i]))

        #[state, init_state, action, reward, done, value, action_log_prob,action_dist_entropy]
        # action: event_times, event_types, spatial_locations
        # state: event_times, event_types, spatial_locations, tpp_state


    def compute_returns(self):

        N, T = self.memory[4].shape[0], self.memory[4].shape[1]

        self.memory.append(np.zeros([N, T])) # gae

        self.memory.append(np.zeros([N, T])) # returns

        for n_seq in range(N):

            value_previous = 0
            gae = 0

            for step in reversed(range(T)):

                reward = self.memory[3][n_seq][step]
                done = self.memory[4][n_seq][step]

                if step>0 and self.memory[4][n_seq][step]==1 and self.memory[4][n_seq][step-1]==1:
                    gae = 0
                    returns = 0
                    value = 0
                else:
                    mask = 1-done
                    value = self.memory[5][n_seq][step]
                    delta = reward + self.args.gamma * value_previous * mask - value
                    gae = delta + self.args.gamma * self.args.lmbda * mask * gae
                    returns = gae + value

                self.memory[8][n_seq][step] = gae
                self.memory[9][n_seq][step] = returns

                value_previous = value


    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory = [None for _ in range(8)]

    def clear_dis(self):
        self.dis_memory = [None for _ in range(3)]

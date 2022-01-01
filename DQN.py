from math import gamma
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import Sequential
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np
from utility import linear_annealing, exponential_annealing
from torch.utils.tensorboard import SummaryWriter
from config import *


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"{DEVICE = }")

class PolicyNetwork(nn.Module):
    def __init__(self, num_states, num_actions):
        super(PolicyNetwork, self).__init__()
        """
        :param num_states: Input size for the network
        :param num_actions: Output size for the network
        
        Define your neural network here
        This network will be used for both active and target networks
        Do not create over-sized networks
            *** You probably do not need more than 2 layers. 3 at most. 
            *** Your layers should probably only need 128 neurons at max. 
            *** Try to create a small network since more neurons means more processing power; increasing training time
            *** This is not a requirement, just a suggestion. You may create any network you want as long as it learns
        """
        # self.layer1 = None
        
        self.layer1 = nn.Linear(num_states, 32)         # hidden layer 1
        self.layer2 = nn.Linear(32, 128)                # hidden layer 2
        self.layer_out = nn.Linear(128, num_actions)    # output

        # self.layer1 = nn.Sequential(
        #     nn.Conv2d(1, 16, 3),
        #     nn.ReLU(),
        #     # nn.MaxPool2d(kernel_size=2, stride=2)
        # )
        # self.layer2 = nn.Sequential(
        #     nn.Conv2d(16, 64, 3),
        #     nn.ReLU(),
        #     # nn.MaxPool2d(kernel_size=2, stride=2)
        # )
        # out_dim = GRID_DIMS-2
        # self.layer_out = nn.Sequential(
        #     nn.Linear(out_dim * out_dim * 64, num_actions),
        #     nn.ReLU()
        # )


    def forward(self, x):
        """
        :param x: Input to the network
        :return: The action probabilities for each action

        This is the method that is called when you send the state to the network
        You send the input x (which is state) through the layers in order
        After each layer, do not forget to pass the output from an activation function (relu, tanh etc.)
        """
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer_out(x)

        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = x.view(x.size(0), -1)  # Flatten for FUlly Connected layer
        # x = self.layer_out(x)
        
        return x


class DQN:
    memory_count = 0        # Amount of data pushed into mem
    update_count = 0        # Number of updates done

    def __init__(self, HYPERPARAMETERS):
        super(DQN, self).__init__()
        self.num_states = HYPERPARAMETERS["number_of_states"]
        self.num_actions = HYPERPARAMETERS["number_of_actions"]
        self.capacity = HYPERPARAMETERS["replay_buffer_capacity"]   # Memory capacity
        self.learning_rate = HYPERPARAMETERS["learning_rate"]       # Alpha in DQN formula
        self.batch_size = HYPERPARAMETERS["batch_size"]             # Number of batches to process at each update
        self.gamma = HYPERPARAMETERS["gamma"]                       # Discount factor

        self.target_net = PolicyNetwork(self.num_states, self.num_actions)
        self.act_net = PolicyNetwork(self.num_states, self.num_actions)
        # self.target_net = PolicyNetwork(self.num_states, self.num_actions).cuda()
        # self.act_net = PolicyNetwork(self.num_states, self.num_actions).cuda()
        # self.target_net = PolicyNetwork(self.num_states, self.num_actions).to(DEVICE)
        # self.act_net = PolicyNetwork(self.num_states, self.num_actions).to(DEVICE)
        self.memory = [None] * self.capacity

        # The epsilon value for e-greedy action selection
        # At the start, the agent will select a random action with %90 probability
        # That value will drop down as we take action, until %10 (it is always good to have some randomness/noise).
        # Linearly or exponentially (your call)
        self.e = 0.9
        if HYPERPARAMETERS["epsilon_annealing"] == 'linear':
            self.epsilon = linear_annealing(
                self.e,
                0.1,
                # HYPERPARAMETERS["number_of_steps"]
                HYPERPARAMETERS["number_of_steps"] * 2/3
            )
        else:
            numsteps_str = str(int(HYPERPARAMETERS["number_of_steps"]))
            num_digits = len(numsteps_str)
            factor = int(numsteps_str[0])
            if factor == 1: 
                factor = 10
            # DECAY_RATIO = 1 - (factor * 10**-num_digits)
            DECAY_RATIO = HYPERPARAMETERS["decay"]
            self.epsilon = exponential_annealing(
                self.e,
                0.1,
                # HYPERPARAMETERS["number_of_steps"]
                DECAY_RATIO
            )

        # We will use Adam optimizer here
        self.optimizer = optim.Adam(self.act_net.parameters(),
                                    self.learning_rate)
        # Mean-squared error will be enough for this project
        self.loss_func = nn.MSELoss()
        # self.writer = SummaryWriter()
        self.update_steps = 0

    def select_action(self, state):
        # To select an action, we need to feed it to Neural Net
        # NN only accepts tensors, so we need to convert the state
        self.e = next(self.epsilon)
        if np.random.random() < self.e:
            # print(f"random action ({eps = })")
            return np.random.randint(0, self.num_actions)
        state = torch.tensor(state, dtype=torch.float32)
        state = state.unsqueeze(0).unsqueeze(0)
        # state = torch.tensor(state, dtype=torch.float32).cuda()
        # state = torch.tensor(state, dtype=torch.float32).to(DEVICE)
        res = self.act_net.forward(state)
        # print(f"{res = } {res.sum() = } ||| {torch.argmax(res) = }")
        return torch.argmax(res)
        
        # Here, the exploitation-exploration balance is handled
        # We get the next epsilon value based on the current step amount
        # raise NotImplementedError("You should write a function for action selection")

    def store_transition(self, transition):
        index = self.memory_count % self.capacity
        self.memory[index] = transition
        self.memory_count += 1
        return self.memory_count >= self.capacity

    def save(self, filename):
        import os
        os.makedirs(filename, exist_ok=True)
        torch.save(self.target_net.state_dict(), filename + "/target_Q.pt")

    def update(self):
        if self.memory_count >= self.capacity:
            self.update_steps += 1
            # Read state, action, reward, next_state from mem
            states, actions, rewards, next_states = [], [], [], []
            for t in self.memory:
                states.append(t.state)
                actions.append(t.action)
                rewards.append(t.reward)
                next_states.append(t.next_state)

            states = torch.tensor(states).float()
            actions = torch.LongTensor(actions).view(-1, 1).long()
            rewards = torch.tensor(rewards).float()
            next_states = torch.tensor(next_states).float()
            # states = torch.tensor(states).float().cuda()
            # actions = torch.LongTensor(actions).view(-1, 1).long().cuda()
            # rewards = torch.tensor(rewards).float().cuda()
            # next_states = torch.tensor(next_states).float().cuda()
            # states = torch.tensor(states).float().to(DEVICE)
            # actions = torch.LongTensor(actions).view(-1, 1).long().to(DEVICE)
            # rewards = torch.tensor(rewards).float().to(DEVICE)
            # next_states = torch.tensor(next_states).float().to(DEVICE)
            
            # The view method reshapes the tensor without any copy
            # operation. It is super fast and efficient

            rewards = (rewards - rewards.mean()) / (rewards+ 1e-7)
            # Take a look at this reward calculation. We have a tensor
            # (1D vector) of rewards, we are calculating mean and std
            # of this tensor, and subtract mean from all elements.
            # Then divide all elements by (std + some small value) to
            # prevent division by zero. What do we get? A normalized
            # tensor of rewards. This is a normalization technique. If
            # the rewards are too divergent, it will affect the training
            # negatively, thus we normalize them for each batch.

            # Calculate target_Q values by using Bellman equation
            # Note that we do not want to calculate gradients for this

            # Update...
            # Get a set of random indices to fetch them in memory
            for index in BatchSampler(SubsetRandomSampler(range(len(self.memory))), 
                                        batch_size=self.batch_size,
                                        drop_last=False):
                # print(f"darari: {index}", end="|", flush=True)
                
                # print(states, states.shape)
                # print(states[index], states[index].shape)

                # Get the current Q values
                # Notice we are using active network and
                # calculating gradients.

                # The optimization loop
                # Call zero_grad to clear previous grads
                # Then make back propagation
                # Then step
              
                # Update the target network every once in a while
                # raise NotImplementedError


                s_batch = states[index]
                a_batch = actions[index]
                r_batch = rewards[index]
                ns_batch = next_states[index]
                ## (Lecture 13) MIT DeepLearning L5, slide 31:
                ## L = E[ ||target_Q - predicted_Q||^2 ]
                ## target_Q = r + gamma * next state max Q values | predicted = (state,action) Q values
                ## target_Q: Bellman equation
                predicted = self.act_net(s_batch).gather(1, a_batch).squeeze() # calculate Q values 
                # print(f"{predicted.shape = }")
                # print(action.shape, action.unsqueeze(-1).shape)
                next_state_max_Qs, _ = self.target_net(ns_batch).max(1)  # _ : maximizing actions
                next_state_max_Qs = next_state_max_Qs.detach()
                # print(f"{next_state_max_Qs.shape = }")
                target = r_batch + self.gamma * next_state_max_Qs
                # print(f"{target.shape = }")
                self.loss = self.loss_func(target, predicted)
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()
                break

            
            if self.update_steps % 100 == 0: # Update the target network every once in a while
                self.target_net.load_state_dict(self.act_net.state_dict())
                
            # print()
            return True     # updated


        else:
            if self.memory_count % 100 == 0:
                print(f"Memory Buffer is too small ({self.memory_count})")
            return False     # not updated

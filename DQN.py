import torch
import torch.nn as nn
from numpy.random import random, randint
import torch.nn.functional as F
from numpy.random import choice
from random import sample


class DQN:
    def __init__(self, Net, param, config, n_action, state_shape):
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        # Création des NN d'éval et de target
        self.eval_model = Net(n_action).to(self.device)
        self.target_model = Net(n_action).to(self.device)

        # Tentative pour charger les poids depuis la dernière sauvegarde
        try:
            self.eval_model.load_state_dict(torch.load("Save/" + config["SAVE_LOC"], map_location=self.device))
            self.target_model.load_state_dict(torch.load("Save/" + config["SAVE_LOC"], map_location=self.device))
            print("Loaded from Memory ! ")
        except:
            pass

        self.param = param
        self.config = config

        self.memory = Buffer(self.param["BUFFER_SIZE"], state_shape, self.device)

        # Optimisation selon le Double DQN ou pas
        if self.config["DOUBLE_DQN"]:
            self.optimizer = torch.optim.Adam(self.eval_model.parameters(),
                                              lr=self.param["LR"], )
        else:
            self.optimizer = torch.optim.Adam(self.target_model.parameters(),
                                              lr=self.param["LR"], )

        self.criterion = nn.MSELoss().to(self.device)

        self.n_action = n_action
        self.step_counter = 0

    def get_action(self, state, test=False):
        """Renvoie une action en fonction de la politique choisie
        si test=True renvoie la meilleur action selon NN"""

        self.target_model.eval()
        state = torch.FloatTensor(state).to(self.device)
        Q = self.target_model(state).view([-1])

        #TEST
        if test:
            action = torch.argmax(Q).item()

        # EXPLORATION BOLTZMANN
        elif self.config["BOLTZMANN"]:
            proba = F.softmax(Q / self.param["TAU"], dim=0).detach()
            proba = proba.cpu().numpy().round(2)
            proba /= proba.sum()
            action = choice([k for k in range(self.n_action)], p=proba)

        # EPSILON-GREEDY
        else:
            if random() > self.param["EPSILON"] or test:
                action = torch.argmax(Q).item()
            else:
                action = randint(0, self.n_action)

        self.target_model.train()
        return action

    def store(self, state, action, next_state, reward, done):
        """Enregistre une expérience selon la mémoire de l'agent"""
        self.memory.append([state, action, next_state, reward, done])

    def learn(self):
        """Etape d'apprentisatige"""
        self.step_counter += 1

        if self.step_counter < self.param["START_TRAIN"]:
            return  # Attend que la mémoire soit suffisament remplie pour commencer


        state, action, next_state, reward, done = self.memory.get_batch(self.param["BATCH_SIZE"])

        # Mode double DQN
        if self.config["DOUBLE_DQN"]:
            # Evolution lente des poids
            eval_dict = self.eval_model.state_dict()
            target_dict = self.eval_model.state_dict()
            for weights in eval_dict:
                target_dict[weights] = (1 - self.param['ALPHA']) * target_dict[weights] + self.param['ALPHA'] * eval_dict[
                    weights]
                self.target_model.load_state_dict(target_dict)

            # Q valeurs d'évaluations
            Q_eval = self.eval_model(state).gather(1, action.long().unsqueeze(1))

        # Mode DQN Simple
        else:
            Q_eval = self.target_model(state).gather(1, action.long().unsqueeze(1))

        # Calcul de la target
        Q_eval = Q_eval.reshape([self.param["BATCH_SIZE"]])
        Q_next = self.target_model(next_state).detach()
        Q_target = reward + self.param["GAMMA"] * Q_next.max(1)[0].reshape([self.param["BATCH_SIZE"]])  # "*reward

        # Optimization
        self.optimizer.zero_grad()
        loss = self.criterion(Q_eval, Q_target)
        loss.backward()
        self.optimizer.step()

        # LOG
        if self.step_counter % 1000 == 0:
            print("Step ", self.step_counter, " : Loss = ", loss, ", Epsilon : ", self.param["EPSILON"])

        # EPSILON DECAY
        if self.param["EPSILON"] > self.param["EPSILON_MIN"]:
            self.param["EPSILON"] *= self.param["EPSILON_DECAY"]


class Buffer:
    def __init__(self, taille_buffer, state_shape, device):
        self.state = torch.empty([taille_buffer, *state_shape]).to(device)
        self.action = torch.empty([taille_buffer]).to(device)
        self.next_state = torch.empty([taille_buffer, *state_shape]).to(device)
        self.reward = torch.empty([taille_buffer]).to(device)
        self.done = torch.empty([taille_buffer]).to(device)

        self.content = []

        self.index = 0
        self.taille = taille_buffer

    def append(self, o):

        self.state[self.index % self.taille] = torch.FloatTensor(o[0])
        self.action[self.index % self.taille] = torch.LongTensor([o[1]])
        self.next_state[self.index % self.taille] = torch.FloatTensor(o[2])
        self.reward[self.index % self.taille] = torch.FloatTensor([o[3]])
        self.done[self.index % self.taille] = torch.BoolTensor([o[4]])

        self.index = (self.index + 1)

    def get_batch(self, batch_size):
        if self.index < self.taille:
            index = sample([k for k in range(self.index)], batch_size)
        else:
            index = sample([k for k in range(self.taille)], batch_size)
        return self.state[index, :], self.action[index], self.next_state[index, :], self.reward[index], self.done[index]

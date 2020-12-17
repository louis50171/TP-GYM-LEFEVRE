import gym
import matplotlib.pyplot as plt
from DeepNet import Net, Net_Dueling
import torch
import numpy as np
from DQN import DQN

torch.manual_seed(0)

config = {
    "TEST_MODE": True,      # Désactive l'entrainement et teste directement le réseau sauvegardé
    "DUELING_DQN": False,   # Active le dueling DQN
    "DOUBLE_DQN": True,     # Active le double DQN
    "BOLTZMANN": False,     # Exploration boltzmann (True), epsilon-greedy (False)
    "PLOTTING": True,       # Affichage du reward temps réel (lent)
    "RENDERING": False,     # Active l'affichage de l'env en temps réel (lent)
    "SAVE": False,          # Active la sauvegarde du DQN
    "SAVE_LOC": "eval_model_cartpool.data",  # Nom du fichier de sauvegarde
    "N_TEST": 10             # Nombre de tests à réaliser (moyenne des récompenses)
}

param = {
    "BUFFER_SIZE": 100000,
    "LR": 1e-4,
    "EPSILON": 1,
    "EPSILON_MIN": 0.1,
    "EPSILON_DECAY": 0.999,
    "BATCH_SIZE": 32,
    "GAMMA": 0.9,
    "ALPHA": 0.005,
    "N_EPISODE": 200,
    "N_STEP": 200,
    "START_TRAIN": 1000,
}


def test(env, dqn):
    """Lance un test avec le DQN passé en paramèter et envoie le score de sortie"""
    observation = env.reset()
    score = 0
    done = False
    while not done:
        env.render()
        action = dqn.get_action(observation, test=True)
        observation, reward, done, info = env.step(action)
        score += reward
    return score

def train(env, dqn):
    """Effectue un episode d'entrainement"""
    observation = env.reset()
    score = 0

    for k in range(param["N_STEP"]):
        if config["RENDERING"] : env.render()

        action = dqn.get_action(observation)  # Action selon la politique d'exploration
        observation_next, reward, done, info = env.step(action)  # Effectue l'action
        dqn.store(observation, action, observation_next, reward, done)  # Stocke l'expérience

        score += reward
        if dqn.memory.index > param["BATCH_SIZE"]:
            dqn.learn()  # Step d'apprentissage

        observation = observation_next
    return score, dqn

def cartpole_NN():
    """Lance l'entrainement/test d'un DQN sur Cartpole"""
    # Création de l'environnement
    env = gym.make('CartPole-v1')
    env = env.unwrapped

    # Dimension de l'espace d'entrée/sortie du NN
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    # Dueling
    if config["DUELING_DQN"]:
        net = Net_Dueling
    else:
        net = Net

    # Création du DQN
    dqn = DQN(net, param, config, action_space, [observation_space])
    scores_list = []

    # entrainement sur le nombre d'épisodes
    if not config["TEST_MODE"]:
        for episode in range(param["N_EPISODE"]):

            score, dqn = train(env, dqn)
            scores_list.append(score)

            print("Episode : ", episode, " | Steps : ", scores_list[-1])

            if episode % 20 == 0 and config["SAVE"]:  # Sauvegarde du DQN tout les 20 episodes
                print("Saved !")
                torch.save(dqn.eval_model.state_dict(), "Save/" + config["SAVE_LOC"])

            if config["PLOTTING"]: plot_evolution(scores_list)  # Affichage reward temps réel

        # Save du DQN
        if config["SAVE"] : torch.save(dqn.eval_model.state_dict(), "Save/" + config["SAVE_LOC"])

    # test
    score = []
    for k in range(config["N_TEST"]):
        s = test(env, dqn)
        score.append(s)
        print("Test Episode ", k + 1, " : ", s)

    print("moyenne : ", np.mean(score))
    print("Standard Deviation : ", np.std(score))
    env.close()


def plot_evolution(data):
    plt.figure(2)
    plt.clf()
    plt.title("Reward")
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.plot(data)
    plt.grid()
    plt.pause(0.001)


if __name__ == '__main__':
    cartpole_NN()
    plt.show()

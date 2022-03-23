import gym
import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from keras.layers import Activation
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from tensorflow.lite.tools import visualize
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects




env = gym.make("MountainCar-v0")
states = env.observation_space.shape[0]
actions = env.action_space.n
episodes = 10

def custom_activation(x):
    return 35*x*x*x
    
    
    
    #return (-3)*x*x*x

get_custom_objects().update({'custom_activation': Activation(custom_activation)})

def make_model(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape = (1,states)))
    model.add(Dense(24))
    model.add(Activation(custom_activation, name = 'SpecialActivation' ))
    model.add(Dense(actions, activation = "linear"))
    return model


def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit = 50000,window_length = 1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                   nb_actions = actions, nb_steps_warmup = 10, target_model_update = 1e-2)
    return dqn



##model = make_model(states,actions)
##dqn = build_agent(model, actions)
##dqn.compile(Adam(lr=0.001), metrics = ['mae'])
##dqn.fit(env, nb_steps = 100000, visualize=False, verbose=1)
##a = dqn.test(env, nb_episodes=20, visualize = True)

##dqn.save_weights("HillClimb.h5f", overwrite = True)

model = make_model(states, actions)
dqn = build_agent(model, actions)
dqn.compile(Adam(lr=0.001), metrics = ['mae'])
dqn.load_weights("HillClimb.h5f")
tes = dqn.test(env, nb_episodes=10, visualize = True)



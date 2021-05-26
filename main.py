import pygame
import sys

import torch
import torch.optim as optim
import torch.nn.functional as F

from game.pipe import Pipe
from game.square import Square
from game.wall import Wall
from game.mygame import MyGame
from game.grid import Grid

from AI.agent import Agent
from AI.dqn import DQN
#from AI.env_manager import CartPoleEnvManager
from AI.epsilon_greedy_strategy import EpsilonGreedyStrategy
from AI.experience import Experience
from AI.q_values import QValues
from AI.replay_memory import ReplayMemory
from AI.utils import plot, get_moving_average, extract_tensors
from AI.training_function import training_function


# Settings
window_width = 100
window_height = 100
square_side = 10
color_white = (255, 255, 255)
color_black = (0, 0, 0)
color_fuzzy = (255, 105, 180)

batch_size =    128 # 256 - number of replay memory experiences considered
gamma =         0.9 # 0.999 - discount factor in Bellman eqn
eps_start =     1 # 1 - exploration rate
eps_end =       0.01 # 0.01
eps_decay =     0.01 # 0.001
target_update = 10 # 10 - frequency for updating target network
memory_size =   100000 # 100000 - capacity of replay memory
lr =            0.001 # 0.001 - learning rate
num_episodes =  1000 # 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



if __name__ == "__main__":
    strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
    agent = Agent(strategy, 4, device)
    memory = ReplayMemory(memory_size)
    grid = Grid(window_width, window_height, square_side)
    policy_net = DQN(grid.size_x, grid.size_y).to(device)
    target_net = DQN(grid.size_x, grid.size_y).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)
    
    episode_scores = []
    for episode in range(num_episodes):
        grid = Grid(window_width, window_height, square_side)
        game = MyGame(window_width, window_height, square_side, grid)
        current_state = game.pipe.get_state()
        score = 0
        i = 0
        while not game.pipe.done:
            i +=1
            print(i)
            action = agent.select_action(current_state, policy_net)
            reward = game.time_step(int(action))
            next_state = game.pipe.get_state()
            #reward = -(length + illegal_moves)
            score += reward
            memory.push(Experience(current_state, action, next_state, reward))
            current_state = next_state

            if memory.can_provide_sample(batch_size):
                experiences = memory.sample(batch_size)
                # :( next line problem
                states, actions, rewards, next_states = extract_tensors(experiences)
                current_q_values = QValues.get_current(policy_net, states, actions)
                next_q_values = QValues.get_next(target_net, next_states)
                target_q_values = (next_q_values * gamma) + rewards

                loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if episode % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

            #game.state = game.pipe.update(game.game_window)
            #game.wall.update(game.game_window, grid)
            #game.pipe.draw(game.game_window)
        #length, fuck_ups = game.time_step()
        #score = -(length + fuck_ups)
        episode_scores.append(score)
        # add if memory.can_provide_sample....
        # add if episode % target_update...
        #plot(episode_scores, 100)



# Deep Lizard stuff
# for episode in range(num_episodes):
#          em.reset()
#          state = em.get_state()
#          for timestep in count():
#              action = agent.select_action(state, policy_net)
#              reward = em.take_action(action)
#              next_state = em.get_state()
#              memory.push(Experience(state, action, next_state, reward))
#              state = next_state
#
#       *       if memory.can_provide_sample(batch_size):
#       *           experiences = memory.sample(batch_size)
#       *           states, actions, rewards, next_states = extract_tensors(experiences)
#
#       *           current_q_values = QValues.get_current(policy_net, states, actions)
#       *           next_q_values = QValues.get_next(target_net, next_states)
#       *           target_q_values = (next_q_values * gamma) + rewards
#
#       *           loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
#       *           optimizer.zero_grad()
#       *           loss.backward()
#       *           optimizer.step()
#
#              if em.done:
#                  episode_durations.append(timestep)
#                  #plot(episode_durations, 100)
#                  break
#
#       *   if episode % target_update == 0:
#       *       target_net.load_state_dict(policy_net.state_dict())
#
#      plot(episode_durations, 100, config)
#      f = get_moving_average(100, episode_durations)
#      f = np.average(f)
#      #print(f'final moving average is of type: {type(f)} and has value {f}')
#      tune.report(final_moving_avg=f)
#      #tune.report(avg_episode_duration=sum(episode_durations)/len(episode_durations))
#      em.close()


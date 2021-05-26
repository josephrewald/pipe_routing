import pygame
import sys
import time

class Pipe(pygame.sprite.Group):
    def __init__(self, start, end, grid):
        pygame.sprite.Group.__init__(self)
        self.illegal_moves = 0
        self.start = start
        self.front = self.start
        self.end = end
        self.grid = grid
        self.ix = 0
        self.iy = 0
        self.add_square(self.start)
        end_square = self.grid[end]
        self.add(end_square)
        self.done = False
        #self.agent = agent
        #self.policy_net = policy_net

    def add_square(self, location):
        new_square = self.grid[location]
        if new_square.is_occupied:
            print('Square already occupied, choose another path.')
            #time.sleep(0.1)
            self.illegal_moves += 1
        else:
            self.add(new_square)
            self.front = location
            new_square.is_occupied = True
        return self.get_state()

    def get_state(self):
        state = self.grid.update_state()
        front_x = self.front[0]
        front_y = self.front[1]
        state[front_x][front_y] += 1
        return state

    # pass game_window to pipe in __init__
    # don't pass new_state to update...? 
    # don't pass agent to pipe at init, do it here. 
    def update(self, action, game_window):
        current_state = self.get_state()
        #chosen_action = self.agent.select_action(current_state, self.policy_net)
        next_square_dict = {'down'  : (self.front[0], self.front[1] + 1),
                            'up'    : (self.front[0], self.front[1] - 1),
                            'left'  : (self.front[0] - 1, self.front[1]),
                            'right' : (self.front[0] + 1, self.front[1])}
        next_square_vals = next_square_dict.values()
        next_square_obs = [self.grid[x] for x in next_square_vals]
        occupied_squares = [x.is_occupied for x in next_square_obs]
        stuck = all(occupied_squares)
        actions = { 0 : 'down', 
                    1 : 'up',
                    2 : 'left',
                    3 : 'right'}

        if stuck:
            print("you lose")
            self.illegal_moves += 100 #TODO grid size x*y
            self.done = True
            return current_state
        else:
            #print(next_squares)
            #print(int(chosen_action))
            #print(actions)
            new_state = self.add_square(next_square_dict[actions[action]])
        if self.front == self.end:
            print('you win!!')
            self.done = True
            #sys.exit()
        self.draw(game_window)
        return 1

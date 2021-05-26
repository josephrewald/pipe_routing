from game.pipe import Pipe
from game.wall import Wall
import pygame
import sys
from game.grid import Grid
import torch
import time


#pygame.init()
class MyGame():
    def __init__(self, window_width, window_height, square_side,\
            grid, fps=10):
        self.window_width = window_width
        self.window_height = window_height
        self.square_side = square_side
        self.fps = fps
        self.clock = pygame.time.Clock()
        self.game_window = pygame.display.set_mode((window_width, window_height))
        #self.pygame.display.set_caption("Automating Mechanical Engineering")

        self.grid = grid
        self.pipe = Pipe((2, 2), (2, 5), self.grid)
        self.wall = Wall(self.grid)
        self.grid.update()
        self.wall.update(self.game_window, self.grid)
        self.state = torch.zeros([self.grid.size_x, self.grid.size_y])
        
    
    def time_step(self, action):
        #while self.pipe.done == False:
        #_ = input()
        self.clock.tick(self.fps)
    
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        reward = self.pipe.update(action, self.game_window)
    
        # pipe update needs an action
        #self.state = self.pipe.update(self.game_window)
        #self.wall.update(self.game_window, self.grid)
        #self.pipe.draw(self.game_window)
        pygame.display.flip()
        #return len(self.pipe), self.pipe.illegal_moves
        return reward

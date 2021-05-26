import pygame
import time

class Wall(pygame.sprite.Group):
    def __init__(self, grid):
        pygame.sprite.Group.__init__(self)

        keys = list(grid.keys())
        x_values = [x for (x, y) in keys]
        y_values = [y for (x, y) in keys]
        x_max = max(x_values)
        y_max = max(y_values)

        for x in range(0, x_max + 1):
            self.add_square((x, 0), grid)
        for x in range(0, x_max + 1):
            self.add_square((x, y_max), grid)
        for y in range(1, y_max):
            self.add_square((0, y), grid)
        for y in range(1, y_max):
            self.add_square((x_max, y), grid)

    def add_square(self, location, grid):
        new_square = grid[location]
        if new_square.is_occupied:
            print('Square already occupied, choose another path.')
            #time.sleep(0.1)
        else:
            self.add(new_square)
            self.front = location
            new_square.is_occupied = True

    def update(self, game_window, grid):
        self.draw(game_window)

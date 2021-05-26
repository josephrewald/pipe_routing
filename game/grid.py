from game.square import Square
import torch

class Grid(dict):
    def __init__(self, window_width, window_height, square_side):
        x = 0
        y = 0
        while y * square_side < window_height:
            while x * square_side < window_width:
                new_square = Square(x, y, square_side)
                self.update({(x, y): new_square})
                x += 1
            self.size_x = x
            x = 0
            y += 1
        self.size_y = y
        self.state = torch.zeros([self.size_x, self.size_y])

    def update_state(self):
        for (x, y) in self.keys():
            self.state[x][y] = int(self[(x, y)].is_occupied)
        return self.state

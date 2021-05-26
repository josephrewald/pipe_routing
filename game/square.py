import pygame

# Settings
color_white = (255, 255, 255)


class Square(pygame.sprite.Sprite):
    def __init__(self, x, y, side):  # x, y are indices
        pygame.sprite.Sprite.__init__(self)
        self.x = x
        self.y = y
        self.image = pygame.Surface((side, side))
        self.image.fill(color_white)
        self.rect = self.image.get_rect()
        self.rect.left = x * side
        self.rect.top = y * side
        self.is_occupied = False

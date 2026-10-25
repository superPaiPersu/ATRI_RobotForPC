import os
import pygame
for i in os.listdir("res/"):
    base=pygame.image.load("base.png")
    base.blit(pygame.image.load("res/" + i), (0, 0))
    pygame.image.save(base,"output/"+i)
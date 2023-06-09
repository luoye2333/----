import pygame
import numpy as np

WIDTH, HEIGHT = 300, 300
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

def draw_grid(screen):
    for i in range(0, WIDTH, 10):
        pygame.draw.line(screen, BLACK, (i, 0), (i, HEIGHT))
        pygame.draw.line(screen, BLACK, (0, i), (WIDTH, i))

class Terrain:
    def __init__(self):
        self.grid = np.zeros((WIDTH // 10, HEIGHT // 10))
        self.is_drawing = False
        self.start_pos = None

    def draw(self, pos):
        x, y = pos[0] // 10, pos[1] // 10
        if self.is_drawing:
            if self.start_pos is None:
                self.start_pos = x, y
            else:
                x0, y0 = self.start_pos
                dx = abs(x - x0)
                dy = abs(y - y0)
                if dx > dy:
                    for i in range(min(x0, x), max(x0, x) + 1):
                        self.grid[i][y0] = 1
                else:
                    for i in range(min(y0, y), max(y0, y) + 1):
                        self.grid[x0][i] = 1
                self.start_pos = x, y

    def reset(self):
        self.grid = np.zeros((WIDTH // 10, HEIGHT // 10))

    def get_terrain(self):
        return self.grid

terrain = Terrain()

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("地形绘制程序")

done = False

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        elif event.type == pygame.MOUSEBUTTONDOWN:
            terrain.is_drawing = True
        elif event.type == pygame.MOUSEBUTTONUP:
            terrain.is_drawing = False
            terrain.start_pos = None
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                terrain.reset()

        if pygame.mouse.get_pressed()[0]:
            terrain.draw(pygame.mouse.get_pos())

    screen.fill(WHITE)
    draw_grid(screen)

    for i in range(WIDTH // 10):
        for j in range(HEIGHT // 10):
            if terrain.grid[i][j]:
                pygame.draw.rect(screen, GREEN, (i*10, j*10, 10, 10))

    pygame.display.flip()

pygame.quit()
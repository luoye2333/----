import pygame
import numpy as np
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import math

WIDTH, HEIGHT = 50, 50
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

class Terrain:
    def __init__(self, accuracy, brush_size):
        self.grid = np.zeros((WIDTH // accuracy, HEIGHT // accuracy))
        self.is_drawing = False
        self.start_pos = None
        self.accuracy = accuracy
        self.brush_size = brush_size

    def draw(self, pos,color=1):
        x, y = pos[0] // self.accuracy, pos[1] // self.accuracy
        if self.is_drawing:
            if self.start_pos is None:
                self.start_pos = x, y
            else:
                for i in range(-self.brush_size // 2, self.brush_size // 2 + 1):
                    for j in range(-self.brush_size // 2, self.brush_size // 2 + 1):
                        if math.sqrt(i**2+j**2)>=(self.brush_size/2):
                            #圆画笔
                            continue
                        if (i + x) >= 0 and (i + x) < WIDTH//self.accuracy and (j + y) >= 0 and (j + y) < HEIGHT//self.accuracy:
                            self.grid[i + x][j + y] = color
                self.start_pos = x, y

    def reset(self):
        self.grid = np.zeros((WIDTH // self.accuracy, HEIGHT // self.accuracy))

    def get_terrain(self):
        return self.grid

# def to_bitmap(terrain):
#     """
#     将最终绘制得到的障碍区域数据缩放到300 x 300 个像素，然后转换为一张对应的bitmap图像
#     terrain: Terrain对象
#     return: PIL Image对象
#     """
#     scale = 10
#     obstacle_data = np.repeat(np.repeat(terrain.get_terrain(), scale, axis=0), scale, axis=1)
#     obstacle_data = np.concatenate([obstacle_data, np.zeros((scale, obstacle_data.shape[1]))], axis=0)
#     obstacle_data = np.concatenate([obstacle_data, np.zeros((obstacle_data.shape[0], scale))], axis=1)
#     obstacle_data[obstacle_data > 0] = 255
#     obstacle_img = Image.fromarray(np.uint8(obstacle_data), 'L')
#     obstacle_img = obstacle_img.resize((WIDTH, HEIGHT), Image.NEAREST)
#     return obstacle_img

#画笔大小
brush_size=10
accuracy=1
terrain = Terrain(accuracy, brush_size)

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
            elif event.key == pygame.K_LEFT:
                terrain.brush_size = max(1, terrain.brush_size - 1)
                print(terrain.brush_size)
            elif event.key == pygame.K_RIGHT:
                # terrain.brush_size = min(10, terrain.brush_size + 1)
                terrain.brush_size = terrain.brush_size + 1
                print(terrain.brush_size)
            # elif event.key == pygame.K_UP:
            #     terrain.accuracy = min(20, terrain.accuracy + 2)
            #     terrain.grid = np.zeros((WIDTH // terrain.accuracy, HEIGHT // terrain.accuracy))
            # elif event.key == pygame.K_DOWN:
            #     terrain.accuracy = max(2, terrain.accuracy - 2)
            #     terrain.grid = np.zeros((WIDTH // terrain.accuracy, HEIGHT // terrain.accuracy))


        if pygame.mouse.get_pressed()[0]:
            #左键画图
            terrain.draw(pygame.mouse.get_pos(),1)
        if pygame.mouse.get_pressed()[2]:
            #右键擦除
            terrain.draw(pygame.mouse.get_pos(),0)
        
    screen.fill(WHITE)

    for i in range(WIDTH // terrain.accuracy):
        for j in range(HEIGHT // terrain.accuracy):
            if terrain.grid[i][j]:
                pygame.draw.rect(screen, BLACK, (i*terrain.accuracy, j*terrain.accuracy, terrain.accuracy, terrain.accuracy))

    pygame.display.flip()

pygame.quit()

# terrainImage=to_bitmap(terrain)
# plt.imshow(terrainImage)#plt image
# image_array=np.transpose(np.array(terrainImage))
# cv2.imshow("123",image_array)
# cv2.waitKey(0)
# cv2.imwrite('map1.png',image_array)

terrainImage=np.transpose(terrain.get_terrain())
terrainImage=1-terrainImage
terrainImage=terrainImage*255
# cv2.imshow("123",terrainImage)
# cv2.waitKey(0)
cv2.imwrite('map1.png',terrainImage)
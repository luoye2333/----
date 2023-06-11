import numpy as np
import math
import matplotlib.pyplot as plt
import random
import cv2
import threading
from matplotlib.animation import FuncAnimation
import time

class ant:
    def __init__(self,
                 start_x=150,
                 start_y=150,
                 start_angle=0,
                 target_x=250,
                 target_y=250,
                 map_size=[300,300],
                 obstacle_map=None,
                 ):
        self.current_pos_x=start_x
        self.current_pos_y=start_y
        self.facing_angle=start_angle
        self.arrive_ending=False
        self.path=np.array([[start_x,start_y]])
        self.target=[target_x,target_y]
        self.map_size_x=map_size[0]
        self.map_size_y=map_size[1]
        if obstacle_map is None:
            self.obstacle_map=np.zeros(map_size)
        else:
            self.obstacle_map=obstacle_map
    
    def judgeDirection(self,
                       InfoDensityMap):
        #从前方区域中找信息素浓度较高的地方前进
        x=self.current_pos_x
        y=self.current_pos_y
        current_angle=self.facing_angle
        map_size_x=InfoDensityMap.shape[0]
        map_size_y=InfoDensityMap.shape[1]

        sense_radius=15#信息素搜集半径
        #备选方向
        left_angle=30
        right_angle=-30
        random_angle=30#角度随机范围:+r~-r

        left_info_density=0
        mid_info_density=0
        right_info_density=0
        for ix in range(math.ceil(x-sense_radius),math.floor(x+sense_radius)):
            for iy in range(math.ceil(y-sense_radius),math.floor(y+sense_radius)):
                #排除超出地图边界的点
                if (ix<0)or(iy<0)or(ix>=map_size_x)or(iy>=map_size_y):continue
                #检测圆半径
                r=math.sqrt((ix-x)**2+(iy-y)**2)
                if r>sense_radius:continue
                #计算方位
                info_angle=math.atan2(iy-y,ix-x)*180/math.pi
                delta_angle=info_angle-current_angle
                if(delta_angle>left_angle/2*3)or(delta_angle<right_angle/2*3):continue
                if(delta_angle>left_angle/2):
                    left_info_density+=InfoDensityMap[ix][iy]
                elif(delta_angle>right_angle/2):
                    mid_info_density+=InfoDensityMap[ix][iy]
                else:
                    right_info_density+=InfoDensityMap[ix][iy]
        #如果差不多，则默认不转动
        tolerance=1e-5
        if(abs(mid_info_density-left_info_density)<tolerance)and\
          (abs(mid_info_density-right_info_density)<tolerance):
            angle_select=1
        else:
            angle_select=np.argmax([left_info_density,mid_info_density,right_info_density])            
        
        if(angle_select==0):
            next_facing_angle=current_angle+left_angle
        elif(angle_select==1):
            next_facing_angle=current_angle
        elif(angle_select==2):
            next_facing_angle=current_angle+right_angle
        
        next_facing_angle+=random.random()*random_angle*2-random_angle
        #整定到0~360度之间
        next_facing_angle=math.fmod(next_facing_angle,360)
        self.facing_angle=next_facing_angle

    def walk(self):
        step_length=1

        #前进一步
        x=self.current_pos_x
        y=self.current_pos_y
        angle=self.facing_angle

        next_pos_x=x+step_length*math.cos(angle/180*math.pi)
        next_pos_y=y+step_length*math.sin(angle/180*math.pi)
        self.facing_angle=angle
        #地图边界处理，碰撞检测
        if(self.obstacleDetection(x=next_pos_x,y=next_pos_y)):
            #如果碰撞，还要更改一次方向
            next_pos_x=x+step_length*math.cos(self.facing_angle/180*math.pi)
            next_pos_y=y+step_length*math.sin(self.facing_angle/180*math.pi)

        self.current_pos_x=next_pos_x
        self.current_pos_y=next_pos_y

        self.path=np.concatenate((self.path,[[next_pos_x,next_pos_y]]))

    def isEnd(self):
        x=self.current_pos_x
        y=self.current_pos_y
        tx=self.target[0]
        ty=self.target[1]

        threshold=25
        dist=math.sqrt((x-tx)**2+(y-ty)**2)
        if (dist<threshold):return True
        else:return False

    def obstacleDetection(self,
                          x,
                          y):
        #地图边界处理
        obstacle_flag=False
        if(round(x)<0)or(round(x)>=map_size_x):
            self.facing_angle=180-self.facing_angle
            obstacle_flag=True
        if(round(y)<0)or(round(y)>=map_size_y):
            self.facing_angle=-self.facing_angle
            obstacle_flag=True
        if obstacle_flag:
            self.facing_angle=math.fmod(self.facing_angle,360)
            return True

        #障碍检测
        if self.obstacle_map[round(y)][round(x)]>0:
            #图像和ndarray对应是先y坐标，再x坐标
            self.facing_angle=self.facing_angle-180
            self.facing_angle=math.fmod(self.facing_angle,360)
            return True

        return False

    def updateInfoDensity(self,
                          InfoDensityMap,
                          x,
                          y,
                          update_intensity=1):
        #根据距离插值到临近的四个点上
        x0=math.floor(x)
        x1=math.ceil(x)
        y0=math.floor(y)
        y1=math.ceil(y)  
        if(x0==x1):
            #刚好整数格
            x1=x0+1
            x=x0+1e-5
        if(y0==y1):
            y1=y0+1
            y=y0+1e-5 
        map_size_x=InfoDensityMap.shape[0]
        map_size_y=InfoDensityMap.shape[1]
        corner_points=[[x0,y0],[x0,y1],[x1,y1],[x1,y0]]
        #排除超出边界的点
        for i in range(len(corner_points)-1,0,-1):
            point=corner_points[i]
            ix=point[0]
            iy=point[1]
            if (ix<0)or(iy<0)or(ix>=map_size_x)or(iy>=map_size_y):
                del corner_points[i]
        #计算到临近四个点的距离，越近权重越高
        corner_dist=[]
        for i in range(len(corner_points)):
            point=corner_points[i]
            ix=point[0]
            iy=point[1]
            corner_dist.append(math.sqrt((x-ix)**2+(y-iy)**2))
        corner_invert_dist=1.0/np.array(corner_dist)
        corner_weight=corner_invert_dist/np.sum(corner_invert_dist)
        
        #更新信息素
        rhoTraverse=3
        for i in range(len(corner_weight)):
            point=corner_points[i]
            ix=point[0]
            iy=point[1]
            weight=corner_weight[i]
            InfoDensityMap[ix][iy]+=weight*rhoTraverse*update_intensity
    
    def updatePathInfoDensity(self,
                              InfoDensityMap):
        #信息素强度和路径长度有关
        pathLength=0
        for i in range(len(self.path)-1):
            p1=self.path[i]
            p2=self.path[i+1]
            pathLength+=math.sqrt(np.sum((p1-p2)**2))

        arrivalIntensity=1/pathLength*10000
        for point in self.path:
            self.updateInfoDensity(InfoDensityMap,
                                   point[0],
                                   point[1],
                                   arrivalIntensity)

accuracy=1.0
map_size_x=300
map_size_y=300
map_size=[map_size_x,map_size_y]
initial_infoDensity=0
InfoDensityMap=np.ones((int(map_size_x/accuracy),
                        int(map_size_y/accuracy)))*initial_infoDensity

# terrainMap=cv2.imread('map1.png',cv2.IMREAD_GRAYSCALE)
terrainMap=cv2.imread('map1.png')
terrainMap=np.flipud(terrainMap)#图像是上到下，坐标系是下到上，所以要颠倒一下

ObstacleMap=cv2.cvtColor(terrainMap, cv2.COLOR_RGB2GRAY)
# cv2.imshow('123',ObstacleMap)
ObstacleMap[ObstacleMap == 0] = 1 #黑色像素点表示障碍
ObstacleMap[ObstacleMap == 255] = 0 #白色像素点表示可通过
# fig=plt.gcf()
fig,axes = plt.subplots()
fig.set_size_inches([7,7]) # 设置图像大小
axes.set_xlim(0, map_size_x)
axes.set_ylim(0, map_size_y)
im=axes.imshow(ObstacleMap,cmap="binary",origin="upper")
plt.pause(0.01)
ant_series=[ant(obstacle_map=ObstacleMap,map_size=map_size)]



def iterate():
    global InfoDensityMap
    iteration_count=0
    while True:
        time.sleep(0.03)
        iteration_count+=1
        # print("iter:"+str(iteration_count))
        #每次增加一只蚂蚁
        # print(len(ant_series))
        if len(ant_series)<100:
            angle=random.random()*360
            ant_series.append(ant(start_angle=angle,
                                obstacle_map=ObstacleMap,
                                map_size=map_size))

        #对所有蚂蚁推进时间
        for i in range(len(ant_series)):
            m_ant=ant_series[i]
            m_ant.judgeDirection(InfoDensityMap)
            m_ant.walk()
            if m_ant.isEnd():
                #走到终点了，就重新生成一个替换
                m_ant.updatePathInfoDensity(InfoDensityMap)
                angle=random.random()*360
                ant_series[i]=ant(start_angle=angle,
                                obstacle_map=ObstacleMap,
                                map_size=map_size)
                
        #没找到终点时的信息素更新
        # for m_ant in ant_series:
        #     m_ant.updateInfoDensity(InfoDensityMap,
        #                             m_ant.current_pos_x,
        #                             m_ant.current_pos_y)

        #自然蒸发率
        rhoEvaporate=0.9
        InfoDensityMap=np.multiply(InfoDensityMap,rhoEvaporate)

def figure_update(output_count):
    start_time = time.time() 
    # print("plot:"+str(output_count))

    #复制地形
    imageArray=np.copy(terrainMap)

    #绘制蚂蚁
    for i in range(len(ant_series)):
        m_ant=ant_series[i]
        x=m_ant.current_pos_x
        y=m_ant.current_pos_y
        imageArray[round(y)][round(x)]=(255,0,0)

    #绘制信息素浓度
    map_all_x=[]
    map_all_y=[]
    max_info_density=np.max(InfoDensityMap)
    if max_info_density==0:
        # plt.pause(0.01)
        im.set_array(imageArray)
        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # print('代码执行时间为 %f 秒' % elapsed_time)
        return im

    for i in range(map_size_x):
        for j in range(map_size_y):
            if InfoDensityMap[i][j]/max_info_density<0.05:continue
            map_all_x.append(i)
            map_all_y.append(j)

    #归一化
    color_info_density=InfoDensityMap[map_all_x, map_all_y]
    max_color=np.max(color_info_density)
    min_color=np.min(color_info_density)
    color_info_density=(color_info_density-min_color)/(max_color-min_color)

    color_map = np.zeros((len(map_all_x), 4))  # 创建颜色数组
    color_map[:, 2] = 1  # 将全部的蓝色通道设为1（蓝色）
    color_map[:, 3] = color_info_density  # 将不透明度设为信息素浓度

    imageArray=cv2.cvtColor(imageArray, cv2.COLOR_RGB2RGBA)
    
    for i in range(len(map_all_x)):
        cv2.circle(imageArray,
                   (map_all_x[i],map_all_y[i]),
                   radius=1,
                   color=color_map[i],
                   thickness=-1)
    im.set_array(imageArray)
    
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print('代码执行时间为 %f 秒' % elapsed_time)
    return im

thread1 = threading.Thread(target=iterate)
# thread2 = threading.Thread(target=output)
thread1.start()
# thread2.start()

ani = FuncAnimation(fig, figure_update, frames=range(10000), interval=30, blit=False)

plt.show()


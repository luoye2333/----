import numpy as np
import math
import matplotlib.pyplot as plt
import random

class ant:
    def __init__(self,
                 start_x=150,
                 start_y=150,
                 start_angle=0):
        self.pos_x=start_x
        self.pos_y=start_y
        self.facing_angle=start_angle
    
    def walk(self,
             InfoDensityMap):
        #从前方区域中找信息素浓度较高的地方前进
        x=self.pos_x
        y=self.pos_y
        current_angle=self.facing_angle
        map_size_x=InfoDensityMap.shape[0]
        map_size_y=InfoDensityMap.shape[1]
        sense_range=10

        left_info_density=0
        mid_info_density=0
        right_info_density=0

        for ix in range(math.ceil(x-sense_range),math.floor(x+sense_range)):
            for iy in range(math.ceil(y-sense_range),math.floor(y+sense_range)):
                #排除超出地图边界的点
                if (ix<0)or(iy<0)or(ix>=map_size_x)or(iy>=map_size_y):continue
                #检测圆半径
                r=math.sqrt((ix-x)**2+(iy-y)**2)
                if r>sense_range:continue
                #计算方位
                info_angle=math.atan2(iy-y,ix-x)*180/math.pi
                delta_angle=info_angle-current_angle
                if(delta_angle>90)or(delta_angle<-90):continue
                if(delta_angle>30):
                    left_info_density+=InfoDensityMap[ix][iy]
                elif(delta_angle>-30):
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
            next_facing_angle=current_angle+60
        elif(angle_select==1):
            next_facing_angle=current_angle
        elif(angle_select==2):
            next_facing_angle=current_angle-60
        #随机一定角度
        random_angle=5
        next_facing_angle+=random.random()*random_angle*2-random_angle
        #整定到0~360度之间
        next_facing_angle=math.fmod(next_facing_angle,360)

        #前进一步
        step_length=3
        next_pos_x=x+step_length*math.cos(next_facing_angle/180*math.pi)
        next_pos_y=y+step_length*math.sin(next_facing_angle/180*math.pi)

        #地图边界处理
        obstacle_flag=False
        if(next_pos_x<0)or(next_pos_x>map_size_x):
            next_facing_angle=180-next_facing_angle
            obstacle_flag=True
        if(next_pos_y<0)or(next_pos_y>map_size_y):
            next_facing_angle=-next_facing_angle
            obstacle_flag=True

        if(obstacle_flag):
            #如果碰撞，还要更改一次方向
            next_pos_x=x+step_length*math.cos(next_facing_angle/180*math.pi)
            next_pos_y=y+step_length*math.sin(next_facing_angle/180*math.pi)

        self.pos_x=next_pos_x
        self.pos_y=next_pos_y
        self.facing_angle=next_facing_angle

    def updateInfoDensity(self,
                          InfoDensityMap):
        x=self.pos_x
        y=self.pos_y
        #根据距离插值到临近的四个点上
        x0=math.floor(x)
        x1=math.ceil(x)
        y0=math.floor(y)
        y1=math.ceil(y)  
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
            InfoDensityMap[ix][iy]+=weight*rhoTraverse
    

ant_series=[ant()]
accuracy=1.0
map_size_x=300
map_size_y=300
initial_infoDensity=0
InfoDensityMap=np.ones((int(map_size_x/accuracy),
                        int(map_size_y/accuracy)))*initial_infoDensity

fig,axes = plt.subplots(1,1)
fig.set_size_inches([7,7]) # 设置图像大小



while True:
    #每秒增加一只蚂蚁
    # print(len(ant_series))
    if len(ant_series)<100:
        ant_series.append(ant())
        #TODO 初始化随机方向
    ant_x=[]
    ant_y=[]
    #对所有蚂蚁推进时间
    for m_ant in ant_series:
        m_ant.walk(InfoDensityMap=InfoDensityMap)
        ant_x.append(m_ant.pos_x)
        ant_y.append(m_ant.pos_y)
    for m_ant in ant_series:
        m_ant.updateInfoDensity(InfoDensityMap=InfoDensityMap)
    #自然蒸发率
    rhoEvaporate=0.99
    InfoDensityMap=np.multiply(InfoDensityMap,rhoEvaporate)
    
    axes.clear()
    axes.set_xlim(0, map_size_x)
    axes.set_ylim(0, map_size_y)
    axes.scatter(ant_x, ant_y, color="black",s=1)# 绘制蚂蚁
    #绘制信息素浓度
    map_all_x=[]
    map_all_y=[]
    max_info_density=np.max(InfoDensityMap)
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

    axes.scatter(map_all_x,map_all_y, c=color_map,s=3)

    plt.pause(0.01)
    
    #TODO:增加信息素更新

    


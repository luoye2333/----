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
                 step_length=3,
                 map_size=[300,300],
                 obstacle_map=None,
                 ):
        self.current_pos_x=start_x
        self.current_pos_y=start_y
        self.facing_angle=start_angle
        self.arrive_ending=False
        self.path=np.array([[start_x,start_y]])
        self.path_length=0

        self.target=[target_x,target_y]
        self.map_size_x=map_size[0]
        self.map_size_y=map_size[1]
        if obstacle_map is None:
            self.obstacle_map=np.zeros(map_size)
        else:
            self.obstacle_map=obstacle_map

        self.step_length=step_length

    def judgeDirection(self,
                       InfoDensityMap):
        #从前方区域中找信息素浓度较高的地方前进
        x=self.current_pos_x
        y=self.current_pos_y
        current_angle=self.facing_angle
        map_size_x=InfoDensityMap.shape[0]
        map_size_y=InfoDensityMap.shape[1]

        sense_radius=10#信息素搜集半径
        #备选方向
        left_angle=30
        right_angle=-30
        random_angle_range=10#角度随机范围:+r~-r

        left_info_density=0
        mid_info_density=0
        right_info_density=0

        x_down=max(math.ceil(x-sense_radius),0)
        x_up=min(math.floor(x+sense_radius),map_size_x)
        y_down=max(math.ceil(y-sense_radius),0)
        y_up=min(math.floor(y+sense_radius),map_size_y)
        search_range_map=InfoDensityMap[x_down:x_up,y_down:y_up]
        # if (np.max(search_range_map)==0)or\
        #    (np.max(search_range_map)<np.max(InfoDensityMap)*0.01):
        if (np.max(search_range_map)==0):
            #如果周围没有信息素，就按原方向走
            max_info_density_angle=current_angle
        else:
            for ix in range(x_down,x_up):
                for iy in range(y_down,y_up):
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
            
            tolerance=1e-5
            if(abs(mid_info_density-left_info_density)<tolerance)and\
            (abs(mid_info_density-right_info_density)<tolerance):
                #如果差不多，则默认不转动
                angle_select=1
            else:
                angle_select=np.argmax([left_info_density,mid_info_density,right_info_density])            
            
            #最大信息素方向
            if(angle_select==0):
                max_info_density_angle=current_angle+left_angle
            elif(angle_select==1):
                max_info_density_angle=current_angle
            elif(angle_select==2):
                max_info_density_angle=current_angle+right_angle
        
        #随机角度
        random_angle=random.random()*random_angle_range*2-random_angle_range
        #终点的方向
        endpoint_angle=math.atan2(self.target[1]-y,self.target[0]-x)*180/math.pi

        #在最大信息素方向和终点方向终选一个
        prob=random.random()
        if prob>0.99:
            next_facing_angle=endpoint_angle
        else:
            next_facing_angle=max_info_density_angle
        next_facing_angle+=random_angle

        #整定到0~360度之间
        next_facing_angle=math.fmod(next_facing_angle,360)
        self.facing_angle=next_facing_angle

    def walk(self):
        # step_length=3
        # min_step=3
        # max_step=3
        # step_length=random.random()*(max_step-min_step)+min_step
        # self.step_length=step_length
        step_length=self.step_length
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
        self.path_length+=step_length

    def isEnd(self):
        x=self.current_pos_x
        y=self.current_pos_y
        tx=self.target[0]
        ty=self.target[1]

        threshold=5
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
        #策略1：转180度
        # if self.obstacle_map[round(y)][round(x)]>0:
        #     #图像和ndarray对应是先y坐标，再x坐标
        #     self.facing_angle=self.facing_angle-180
        #     self.facing_angle=math.fmod(self.facing_angle,360)
        #     return True

        #策略2：找障碍法向，沿墙壁法向翻转
        # if self.obstacle_map[round(y)][round(x)]>0:
        #     #先找行进方向和墙壁边线的交点
        #     current_x=self.current_pos_x
        #     current_y=self.current_pos_y
        #     next_x=x
        #     next_y=y
        #     angle=self.facing_angle

        #     surface_x=x
        #     surface_y=y
        #     for step_length in np.arange(0,self.step_length,0.5) :
        #         surface_x=round(current_x+step_length*math.cos(angle/180*math.pi))
        #         surface_y=round(current_y+step_length*math.sin(angle/180*math.pi))
        #         if self.obstacle_map[surface_y][surface_x]>0:
        #             break
            
        #     calculate_radius=3
        #     x_down=max(math.ceil(surface_x-calculate_radius),0)
        #     x_up=min(math.floor(surface_x+calculate_radius),map_size_x)
        #     y_down=max(math.ceil(surface_y-calculate_radius),0)
        #     y_up=min(math.floor(surface_y+calculate_radius),map_size_y)
        #     #计算得到交点后，截取交点附近的墙壁，作边缘检测算子
        #     calculate_map=self.obstacle_map[y_down:y_up,x_down:x_up]
        #     # cv2.imshow("456",cv2.resize(calculate_map*255,(300,300)))
        #     canny=cv2.Canny(calculate_map,0.5,0.5)
        #     # cv2.imshow("123",cv2.resize(canny,(300,300)))

        #     #二值化得到边线的点
        #     thresh_img = cv2.threshold(canny, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        #     # cv2.imshow("789",cv2.resize(thresh_img,(300,300)))
        #     points = np.column_stack(np.where(thresh_img.transpose() > 0))
        #     if(points.shape[0]==0):
        #         a=1
        #         return False
        #     vx, vy, x, y = cv2.fitLine(points, cv2.DIST_L2, 0, 0.1, 0.1)
        #     normal_angle=math.atan2(vy,vx)*180/math.pi
        #     #沿法线方向翻转
        #     #确认法线方向没有反
        #     # if(abs(normal_angle-angle)<90)
        #     self.facing_angle=2*normal_angle-self.facing_angle
        #     self.facing_angle=math.fmod(self.facing_angle,360)
        #     return True

        #策略3：往随机方向转，直到不碰撞
        if (self.obstacle_map[round(y)][round(x)]>0):
            while(True):
                self.facing_angle=random.random()*360
                x=self.current_pos_x+self.step_length*math.cos(self.facing_angle/180*math.pi)
                y=self.current_pos_y+self.step_length*math.sin(self.facing_angle/180*math.pi)
                if(round(x)<0)or(round(x)>=map_size_x):
                    continue
                if(round(y)<0)or(round(y)>=map_size_y):
                    continue
                if (self.obstacle_map[round(y)][round(x)]==0):
                    break
            return True

        return False

    def updateInfoDensity(self,
                          InfoDensityMap,
                          x,
                          y,
                          update_intensity=1):
        InfoDensityMap[round(x)][round(y)]+=update_intensity
        return

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
        rhoTraverse=1
        for i in range(len(corner_weight)):
            point=corner_points[i]
            ix=point[0]
            iy=point[1]
            weight=corner_weight[i]
            InfoDensityMap[ix][iy]+=weight*rhoTraverse*update_intensity
    
    def updatePathInfoDensity(self,
                              InfoDensityMap,
                              history_path,
                              index):
        #信息素强度和路径长度有关

        pathLength=self.path_length

        # #信息素强度应该和全局\历史最优路径有关
        # if(history_min_path>history_max_path):
        #     #未初始化
        #     history_min_path=pathLength
        #     history_max_path=pathLength
        #     arrivalIntensity=1
        # else:
        #     if pathLength>history_max_path:
        #         history_max_path=pathLength
        #     if pathLength<history_min_path:
        #         history_min_path=pathLength
        #     #用最大最小值归一化
        #     func_x=1-(pathLength-history_min_path)/(history_max_path-history_min_path)
        #     func_y=pow(func_x,100)#x^10: 0~1非线性映射到0~1
        #     arrivalIntensity=func_y
        # print("min:{0:.2f},max:{1:.2f}".format(history_min_path,history_max_path))
        


        min_x = 0
        max_x = len(history_path)
        min_y=0
        max_y=1
        if (max_x==min_x):
            func_y=max_y
        else:
            #查找当前路径长度排名
            func_x=1-(index - min_x) / (max_x - min_x)
            func_y=pow(func_x,10)*(max_y-min_y)+min_y
        # print("len:{0:.2f},y:{1:.2f},best:{2:.2f}".format(pathLength,func_y,history_path[0]))

        arrivalIntensity=func_y*100
        for i in range(len(self.path)-1):
            #不增大
            # point=self.path[i]
            # self.updateInfoDensity(InfoDensityMap,
            #                        point[0],
            #                        point[1],
            #                        arrivalIntensity)
            
            #越接近终点强度越大
            #等差式增大
            # point1=self.path[i]
            # min_scaler=0.5
            # max_scaler=1
            # scaler=i/len(self.path)*(max_scaler-min_scaler)+min_scaler
            # self.updateInfoDensity(InfoDensityMap,
            #                        point1[0],
            #                        point1[1],
            #                        arrivalIntensity*scaler)
            
            #填补步长太大导致的空隙
            point1=self.path[i]
            min_scaler=0.5
            max_scaler=1
            scaler=i/len(self.path)*(max_scaler-min_scaler)+min_scaler
            point2=self.path[i+1]
            #中间的一系列点+端点
            x=np.linspace(point1[0],point2[0],self.step_length+1)
            y=np.linspace(point1[1],point2[1],self.step_length+1)
            points=[(round(x[i]),round(y[i])) for i in range(self.step_length+1)]
            points=list(set(points))#去除重复
            for point in points:
                self.updateInfoDensity(InfoDensityMap,
                                    point[0],
                                    point[1],
                                    arrivalIntensity*scaler)

            #等比增大
            # point=self.path[i]
            # first=pow(arrivalIntensity,1/len(self.path))
            # intensity=pow(first,i)
            # self.updateInfoDensity(InfoDensityMap,
            #                        point[0],
            #                        point[1],
            #                        intensity)

        return history_path
        # return [history_min_path,history_max_path]



# terrainMap=cv2.imread('map1.png',cv2.IMREAD_GRAYSCALE)
terrainMap=cv2.imread('map3.png')
terrainMap=np.flipud(terrainMap)#图像是上到下，坐标系是下到上，所以要颠倒一下
ObstacleMap=cv2.cvtColor(terrainMap, cv2.COLOR_RGB2GRAY)
# cv2.imshow('123',ObstacleMap)
ObstacleMap[ObstacleMap == 0] = 1 #黑色像素点表示障碍
ObstacleMap[ObstacleMap == 255] = 0 #白色像素点表示可通过

accuracy=1.0
map_size_x=terrainMap.shape[0]
map_size_y=terrainMap.shape[1]
target_x=250
target_y=250
start_x=150
start_y=150

map_size=[map_size_x,map_size_y]
initial_infoDensity=0
InfoDensityMap=np.ones((int(map_size_x/accuracy),
                        int(map_size_y/accuracy)))*initial_infoDensity


# fig=plt.gcf()
fig,axes = plt.subplots()
fig.set_size_inches([7,7]) # 设置图像大小
axes.set_xlim(0, map_size_x)
axes.set_ylim(0, map_size_y)
im=axes.imshow(ObstacleMap,cmap="binary",origin="upper")
plt.pause(0.01)
ant_series=[ant(obstacle_map=ObstacleMap,
                start_x=start_x,
                start_y=start_y,
                target_x=target_x,
                target_y=target_y,
                map_size=map_size)]

# generated_min_path_length=np.inf
# generated_max_path_length=0
history_path=[]
history_best_ant=None
history_best_ant_series=[]
history_update=True

def get_max_infoDensity_angle(map):
    #创建一个0~360度的数组
    #对圆形范围内的所有像素点进行投票
    gap=10#区间长度
    bar=np.arange(0,360,gap)
    #0gap~1gap,1gap~2gap,n-1~ngap
    #共360/gap长度
    #第i对应(i~i+1)*gap范围
    map_size_x=map.shape[0]
    map_size_y=map.shape[1]
    radius=map_size_x/2
    center_x=(map_size_x-1)/2
    center_y=(map_size_y-1)/2

    if np.max(map)==0:
        return random.random()*360

    for i in range(map_size_x):
        for j in range(map_size_y):
            if map[i,j]==0:continue
            dx=i-center_x
            dy=j-center_y
            if(math.sqrt(dx**2+dy**2)>radius):continue
            angle=math.atan2(dy,dx)*180/math.pi
            index=int(angle/gap)
            bar[index]+=map[i,j]
    #得出最大区间
    max_index=np.argmax(bar)
    max_angle=(max_index+0.5)*gap
    return max_angle

#TODO RRT算法添加初始化信息，加快收敛

def figure_update(output_count):
    global InfoDensityMap
    # global generated_min_path_length
    # global generated_max_path_length
    global history_path
    global history_best_ant
    global history_best_ant_series
    global history_update

    iteration_count=output_count
    # print("iter:"+str(iteration_count))
    #每次增加一只蚂蚁
    # print(len(ant_series))
    if len(ant_series)<200:
        #找起点周围信息素最大的方向
        #统计四个象限内哪个象限信息素最多
        # start_search_radius=3
        # left_up=0;right_up=0
        # left_down=0;right_down=0
        # for i in range(-start_search_radius,start_search_radius):
        #     for j in range(-start_search_radius,start_search_radius):
        #         intensity=InfoDensityMap[start_x+i][start_y+j]
        #         if (i<0)and(j<0):
        #             left_down+=intensity
        #         if (i<0)and(j>0):
        #             left_up+=intensity
        #         if (i>0)and(j<0):
        #             right_down+=intensity
        #         if (i>0)and(j>0):
        #             right_up+=intensity
        # max_array=[right_up,left_up,left_down,right_down]
        # if max(max_array)==0:
        #     angle=random.random()*360
        # else:
        #     max_index=np.argmax(max_array)
        #     angle=45+max_index*90

        # angle=random.random()*360

        start_search_radius=3
        search_map=InfoDensityMap[start_x-start_search_radius:start_x+start_search_radius+1,
                                  start_y-start_search_radius:start_y+start_search_radius+1]
        search_map=np.flipud(search_map)
        angle=get_max_infoDensity_angle(search_map)
        #print("spawn-angle:{0:.2f}".format(angle))

        ant_series.append(ant(start_angle=angle,
                              start_x=start_x,
                              start_y=start_y,
                              target_x=target_x,
                              target_y=target_y,
                              obstacle_map=ObstacleMap,
                              map_size=map_size))

    #对所有蚂蚁推进时间
    for i in range(len(ant_series)):
        m_ant=ant_series[i]
        m_ant.judgeDirection(InfoDensityMap)
        m_ant.walk()
        if m_ant.isEnd():
            #走到终点了，就重新生成一个替换
            # return_path=m_ant.updatePathInfoDensity(InfoDensityMap,
            #                                         generated_min_path_length,
            #                                         generated_max_path_length)

            #维护队列
            # if len(history_path)>10:
                #太长了就均匀的删掉一些
                # indices=np.arange(0,len(history_path),3) 
                # history_path=list(np.array(history_path)[indices])
                # history_best_ant_series=list(np.array(history_best_ant_series)[indices])
                
                #只取最短的路径
                # indices=range(0,int(len(history_path)/3),1) 
                # history_path=list(np.array(history_path)[indices])
                # history_best_ant_series=list(np.array(history_best_ant_series)[indices])
                
                
            #二分法插入
            element=m_ant.path_length
            low, high = 0, len(history_path)
            while low < high:
                mid = (low + high) // 2
                if history_path[mid] < element:
                    low = mid + 1
                else:
                    high = mid
            
            max_length=10
            if low==max_length:
                #插入的是最后一个值，直接不用管
                pass
            else:
                history_path.insert(low, element)
                history_best_ant_series.insert(low,m_ant)
                insert_index=low

                if len(history_path)>max_length:
                    #删除最后一个元素
                    del history_path[max_length]
                    del history_best_ant_series[max_length]

                history_update=True
            # history_path=m_ant.updatePathInfoDensity(InfoDensityMap,
            #                                          history_path,
            #                                          insert_index)
            # generated_min_path_length=return_path[0]
            # generated_max_path_length=return_path[1]
            if history_best_ant is not None:
                if(m_ant.path_length<history_best_ant.path_length):
                    history_best_ant=m_ant
            else:
                history_best_ant=m_ant
            print("current:{0:.2f},best:{1:.2f}".format(m_ant.path_length,
                                                        history_best_ant_series[0].path_length))

            #历史最优路径产生信息素
            # if history_best_ant is not None:
            #     history_best_ant.updatePathInfoDensity(InfoDensityMap,history_path)
                # if (output_count%10)==0:
                #     history_best_ant.updatePathInfoDensity(InfoDensityMap,history_path)

            angle=random.random()*360
            ant_series[i]=ant(start_angle=angle,
                              start_x=start_x,
                              start_y=start_y,
                              target_x=target_x,
                              target_y=target_y,
                              obstacle_map=ObstacleMap,
                              map_size=map_size)
        else:
            if history_best_ant is not None:
                if m_ant.path_length>history_best_ant.path_length*2:
                    #路径太长的清除掉
                    angle=random.random()*360
                    ant_series[i]=ant(start_angle=angle,
                                      start_x=start_x,
                                      start_y=start_y,
                                      target_x=target_x,
                                      target_y=target_y,
                                      obstacle_map=ObstacleMap,
                                      map_size=map_size)

    if history_update:
        InfoDensityMap=np.zeros((map_size_x,map_size_y))
        for i in range(len(history_best_ant_series)):
            m_ant=history_best_ant_series[i]
            m_ant.updatePathInfoDensity(InfoDensityMap,
                                        history_path,
                                        i)
        history_update=False


    #没找到终点时的信息素更新
    # for m_ant in ant_series:
    #     m_ant.updateInfoDensity(InfoDensityMap,
    #                             m_ant.current_pos_x,
    #                             m_ant.current_pos_y)

    #自然蒸发率
    # beta=50#衰减到一半的次数
    # rhoEvaporate=math.pow(0.1,1/beta)
    # InfoDensityMap=np.multiply(InfoDensityMap,rhoEvaporate)
    # InfoDensityMap=np.where(InfoDensityMap < 0.1, 0, InfoDensityMap)
    
    # #信息素扩散到周围的格子
    # #使用高斯模糊
    # InfoDensityMap=cv2.GaussianBlur(InfoDensityMap, (3, 3), 0)
    # #有障碍物的地方置0
    # InfoDensityMap = np.where(np.transpose(ObstacleMap) ==1, 0, InfoDensityMap)



    plot=True
    if not plot:
        return im
    
    #复制地形
    imageArray=np.copy(terrainMap)

    #绘制信息素浓度
    map_all_x=[]
    map_all_y=[]
    max_info_density=np.max(InfoDensityMap)
    if max_info_density==0:
        #绘制蚂蚁
        for i in range(len(ant_series)):
            m_ant=ant_series[i]
            x=m_ant.current_pos_x
            y=m_ant.current_pos_y
            imageArray[round(y)][round(x)]=(255,0,0)
        cv2.circle(imageArray,
                    (target_x,target_y),
                    radius=1,
                    color=(241,188,202,255),
                    thickness=-1,
                    lineType=cv2.LINE_AA)
        im.set_array(imageArray)
        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # print('代码执行时间为 %f 秒' % elapsed_time)
        return im

    for i in range(map_size_x):
        for j in range(map_size_y):
            if InfoDensityMap[i][j]/max_info_density<0.01:continue
            map_all_x.append(i)
            map_all_y.append(j)

    #归一化
    color_info_density=InfoDensityMap[map_all_x, map_all_y]/1.0
    max_color=np.max(color_info_density)
    min_color=np.min(color_info_density)
    color_info_density=(color_info_density-min_color)/(max_color-min_color)

    color_map = np.zeros((len(map_all_x), 4))  # 创建颜色数组
    color_map[:, 2] = 255  # 将全部的蓝色通道设为255
    color_map[:, 3] = color_info_density*255  # 将不透明度设为信息素浓度

    imageArray=cv2.cvtColor(imageArray, cv2.COLOR_RGB2RGBA)
    
    marker_radius=1
    for i in range(len(map_all_x)):
        #单点模式
        point_x=map_all_x[i]
        point_y=map_all_y[i]
        imageArray[point_y,point_x]=tuple(color_map[i])

        #cv2函数画圆
        # cv2.circle(imageArray,
        #            (map_all_x[i],map_all_y[i]),
        #            radius=marker_radius,
        #            color=tuple(color_map[i]),
        #            thickness=-1,
        #            lineType=cv2.LINE_AA
        #            )

        #手动画圆
        # point_x=map_all_x[i]
        # point_y=map_all_y[i]
        # for ix in range(math.ceil(point_x-marker_radius),math.floor(point_x+marker_radius)):
        #     for iy in range(math.ceil(point_y-marker_radius),math.floor(point_y+marker_radius)):
        #         #排除超出地图边界的点
        #         if (ix<0)or(iy<0)or(ix>=map_size_x)or(iy>=map_size_y):continue
        #         r=math.sqrt((ix-point_x)**2+(iy-point_y)**2)
        #         if r>marker_radius:continue
        #         imageArray[round(point_y)][round(point_x)]=tuple(color_map[i])
        
    #绘制蚂蚁
    for i in range(len(ant_series)):
        m_ant=ant_series[i]
        x=m_ant.current_pos_x
        y=m_ant.current_pos_y
        imageArray[round(y)][round(x)]=(255,0,0,255)

    #绘制终点
    cv2.circle(imageArray,
                (target_x,target_y),
                radius=1,
                color=(241,188,202,255),
                thickness=-1,
                lineType=cv2.LINE_AA)
    
    im.set_array(imageArray)
    return im

ani = FuncAnimation(fig, figure_update, frames=range(10000), interval=1, blit=False)

plt.show()


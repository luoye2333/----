import numpy as np
import math
import matplotlib.pyplot as plt
import random
import cv2
import threading
from matplotlib.animation import FuncAnimation
import time
from motion_roadmap import MotionRoadmap
import motion_planning_toolbox as mpt

class ant:
    def __init__(self,
                 start_x=150,
                 start_y=150,
                 start_angle=0,
                 target_x=250,
                 target_y=250,
                 step_length=3,
                 step_length_min=3,
                 sense_radius=10,
                 random_angle_range=30,
                 end_threshold=5,
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
        self.step_length_min=step_length_min
        self.sense_radius=sense_radius
        self.random_angle_range=random_angle_range
        self.end_threshold=end_threshold

    def judgeDirection(self,
                       InfoDensityMap,
                       angleMap):
        #从前方区域中找信息素浓度较高的地方前进
        x=self.current_pos_x
        y=self.current_pos_y
        current_angle=self.facing_angle
        map_size_x=InfoDensityMap.shape[0]
        map_size_y=InfoDensityMap.shape[1]

        sense_radius=self.sense_radius#信息素搜集半径
        #备选方向
        left_angle=60
        right_angle=-60
        random_angle_range=self.random_angle_range#角度随机范围:+r~-r
        left_info_density=0
        mid_info_density=0
        right_info_density=0

        x_down=max(math.ceil(x-sense_radius),0)
        x_up=min(math.floor(x+sense_radius),map_size_x)
        y_down=max(math.ceil(y-sense_radius),0)
        y_up=min(math.floor(y+sense_radius),map_size_y)
        search_range_map=InfoDensityMap[x_down:x_up,y_down:y_up]
        if (np.max(search_range_map)==0):
            #如果周围没有信息素，就按原方向走
            max_info_density_angle=current_angle
        else:
            # max_info_density_angle=get_max_infoDensity_angle(InfoDensityMap,
            #                                                  round(x),
            #                                                  round(y),
            #                                                  radius=sense_radius)

            for ix in range(x_down,x_up):
                for iy in range(y_down,y_up):
                    #检测圆半径
                    r=math.sqrt((ix-x)**2+(iy-y)**2)
                    if r>sense_radius:continue
                    #计算方位
                    info_angle=math.atan2(iy-y,ix-x)*180/math.pi
                    if info_angle<0:info_angle+=360
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

            # #排除刚好沿着路径方向走的情况
            # reference_radius=5
            # x_down=max(math.ceil(x-reference_radius),0)
            # x_up=min(math.floor(x+reference_radius),map_size_x)
            # y_down=max(math.ceil(y-reference_radius),0)
            # y_up=min(math.floor(y+reference_radius),map_size_y)
            # reference_angle_map=angleMap[x_down:x_up,y_down:y_up]
            # mask=reference_angle_map>0
            # #求均值时排除差特别多的点
            # data=reference_angle_map[mask]
            # if not len(data)==0:
            #     # 计算数据集平均值和标准差
            #     data_mean = np.mean(data)
            #     data_std = np.std(data)
            #     # 计算每个数据点的 Z-Score
            #     z_scores = (data - data_mean) / data_std
            #     # 找到大于 3 或者小于 -3 的数据点
            #     outliers = np.where(np.abs(z_scores) > 2)
            #     # 移除离群点，重新计算平均值
            #     data_cleaned = np.delete(data, outliers)
            #     mean_cleaned = np.mean(data_cleaned)
            #     reference_angle=mean_cleaned

            #     #debug:显示当前search_map
            #     # image=cv2.resize(cv2.transpose(np.flip(reference_angle_map,1)),(280,280),cv2.INTER_BITS)
            #     # for i in range(image.shape[1]):
            #     #     for j in range(image.shape[0]):
            #     #         if (i%40==0)or(j%40==0):
            #     #             image[j, i] = np.max(image)
            #     # cv2.imshow('123',image)

            #     tolerence_angle=30
            #     delta=abs(max_info_density_angle-reference_angle)-180
            #     if abs(delta)<tolerence_angle:
            #         max_info_density_angle=max_info_density_angle-180
            #         if max_info_density_angle<0:max_info_density_angle+=360
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
        min_step=self.step_length_min
        max_step=self.step_length
        step_length=random.random()*(max_step-min_step)+min_step
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
        dist=math.sqrt((x-tx)**2+(y-ty)**2)
        if (dist<self.end_threshold):
            #把最后一个点放到path中
            self.path=np.concatenate((self.path,[[tx,ty]]))
            return True
        else:return False

    def obstacleDetection(self,
                          x,
                          y):
        #地图边界处理
        map_size_x=self.map_size_x
        map_size_y=self.map_size_y

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

        # #策略2：找障碍法向，沿墙壁法向翻转
        # if self.obstacle_map[round(x)][round(y)]>0:
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
        #         if self.obstacle_map[surface_x][surface_y]>0:
        #             break
            
        #     calculate_radius=3
        #     x_down=max(math.ceil(surface_x-calculate_radius),0)
        #     x_up=min(math.floor(surface_x+calculate_radius),map_size_x)
        #     y_down=max(math.ceil(surface_y-calculate_radius),0)
        #     y_up=min(math.floor(surface_y+calculate_radius),map_size_y)
        #     #计算得到交点后，截取交点附近的墙壁，作边缘检测算子
        #     calculate_map=self.obstacle_map[x_down:x_up,y_down:y_up]

        #     #注意用cv2的函数前x,y转置
        #     calculate_map=cv2.transpose(calculate_map)

        #     # cv2.imshow("456",cv2.resize(calculate_map*255,(300,300)))
        #     canny=cv2.Canny(calculate_map,0.5,0.5)
        #     # cv2.imshow("123",cv2.resize(canny,(300,300)))

        #     #二值化得到边线的点
        #     thresh_img = cv2.threshold(canny, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        #     # cv2.imshow("789",cv2.resize(thresh_img,(300,300)))
        #     points = np.column_stack(np.where(thresh_img.transpose() > 0))
        #     if(points.shape[0]==0):
        #         #没检测到边线的点
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
        if (self.obstacle_map[round(x)][round(y)]>0):
            while(True):
                self.facing_angle=random.random()*360
                x=self.current_pos_x+self.step_length*math.cos(self.facing_angle/180*math.pi)
                y=self.current_pos_y+self.step_length*math.sin(self.facing_angle/180*math.pi)
                if(round(x)<0)or(round(x)>=map_size_x):
                    continue
                if(round(y)<0)or(round(y)>=map_size_y):
                    continue
                if (self.obstacle_map[round(x)][round(y)]==0):
                    break
            return True

        return False
    

class continous_ant_alogrithm:
    def __init__(self,
                 terrain_image,
                 target_x,
                 target_y,
                 start_x,
                 start_y,
                 ant_limit=5000,
                 history_max_length=200,
                 ):
        #图片坐标和逻辑坐标对应关系
        #首先opencv是先y坐标后x坐标，所以转置一下
        self.terrainMap=cv2.transpose(terrain_image)
        self.map_size_x=self.terrainMap.shape[0]
        self.map_size_y=self.terrainMap.shape[1]
        self.map_size=[self.map_size_x,self.map_size_y]
        #另外图片y变小对应逻辑坐标y变大，y坐标相反
        #用flip翻转第二个坐标(y)
        self.terrainMap=np.flip(self.terrainMap,1)

        ObstacleMap=cv2.cvtColor(self.terrainMap, cv2.COLOR_RGB2GRAY)
        # cv2.imshow('123',ObstacleMap)
        ObstacleMap[ObstacleMap == 0] = 1 #黑色像素点表示障碍
        ObstacleMap[ObstacleMap == 255] = 0 #白色像素点表示可通过
        self.ObstacleMap=ObstacleMap


        self.target_x=target_x
        self.target_y=target_y
        self.start_x=start_x
        self.start_y=start_y
        self.ant_limit=ant_limit
        self.history_max_length=history_max_length

        self.history_path=[]
        self.history_best_ant_series=[]
        self.history_update=True
        initial_infoDensity=0
        self.InfoDensityMap=np.ones(self.map_size)*initial_infoDensity
        self.angleMap=np.ones(self.map_size)*(-1)#负数表示没有值
        self.angleNumberMap=np.zeros(self.map_size)

        fig,axes = plt.subplots()
        fig.set_size_inches([7,7]) # 设置图像大小
        axes.set_xlim(0, self.map_size_x)
        axes.set_ylim(0, self.map_size_y)
        self.fig=fig
        self.axes=axes
        #逻辑绘图，注意只转置就行，因为plt坐标轴是对的，但是索引还是[y][x]
        self.im=axes.imshow(np.transpose(ObstacleMap),cmap="binary",origin="upper")
        plt.pause(0.01)
        #cv2绘图，把逻辑数组恢复到图片数组
        # cv2.imshow('123',cv2.transpose(np.flip(ObstacleMap,1)*255))

        self.ant_series=[self.new_ant()]
        #直接生成所有的蚂蚁
        # for i in range(self.ant_limit-1):
        #     self.ant_series.append(self.new_ant())

        #self.InitializePathGraph()

        self.iteration_count=0

    def InitializePathGraph(self):
        #RRT
        img = self.terrainMap
        #逻辑图转回原图
        img=np.flip(img,1)
        img=cv2.transpose(img)

        #RRT算法处理的时候，根据cv2坐标
        #和逻辑坐标不同，所以把0轴[y]反一下
        img=cv2.flip(img,0)

        mr = MotionRoadmap(img)
        mr.point_strat = np.mat([self.start_x,self.start_y])
        mr.point_goal = np.mat([self.target_x,self.target_y])

        step=6
        threshold=6
        mr.rrt_planning(s=step, t=threshold, l=300000,p=True)
        # mpt.tree_plot(mr.map, mr.rrt_tree)

        path=mr.get_path()
        #生成蚂蚁放入历史记录中
        m_ant=self.new_ant()
        m_ant.step_length=step
        m_ant.path=path
        m_ant.path_length=(len(path)-1)*step

        self.history_best_ant_series.append(m_ant)
        self.history_path.append(m_ant.path_length)
        self.history_update=True
        self.updateInfoDensity()

    def iterate(self):
        self.iteration_count+=1

        if len(self.ant_series)<self.ant_limit:
            #数量没达到上限之前，每次迭代增加一只蚂蚁
            self.ant_series.append(self.new_ant())

        #对所有蚂蚁推进时间
        for i in range(len(self.ant_series)):
            m_ant=self.ant_series[i]
            m_ant.judgeDirection(self.InfoDensityMap,self.angleMap)
            m_ant.walk()
            if m_ant.isEnd():
                #走到终点了，就重新生成一个替换
                self.ant_series[i]=self.new_ant()

                #把结果插入历史队列中(二分法)
                element=m_ant.path_length
                low, high = 0, len(self.history_path)
                while low < high:
                    mid = (low + high) // 2
                    if self.history_path[mid] < element:
                        low = mid + 1
                    else:
                        high = mid
                
                #维护长度
                if low==self.history_max_length:
                    #插入的是最后一个值，丢弃不用
                    pass
                else:
                    self.history_path.insert(low,element)
                    self.history_best_ant_series.insert(low,m_ant)
                    if len(self.history_path)>self.history_max_length:
                        #插入在中间导致了超长，则删除最后一个元素
                        del self.history_path[self.history_max_length]
                        del self.history_best_ant_series[self.history_max_length]
                    self.history_update=True

                print("current:{0:.2f},best:{1:.2f},iter:{2}".format(
                    m_ant.path_length,
                    self.history_best_ant_series[0].path_length,
                    self.iteration_count))

            #路径太长，没有继续计算价值，则清除
            min_length=np.inf
            if len(self.history_best_ant_series)>0:
                min_length=self.history_best_ant_series[0].path_length
            if m_ant.path_length>min_length*2:
                self.ant_series[i]=self.new_ant()
        
        #迭代完所有蚂蚁以后，更新信息素
        self.updateInfoDensity()

    def updateInfoDensity(self):
        # #行走的过程中产生信息素
        # for m_ant in self.ant_series:
        #     path_length=m_ant.path_length
        #     x=m_ant.current_pos_x
        #     y=m_ant.current_pos_y
        #     # self.InfoDensityMap[round(x),round(y)]+=1/path_length
        #     self.InfoDensityMap[round(x),round(y)]+=1
        # return 
    
        if not self.history_update:
            return
        
        self.InfoDensityMap=np.zeros(self.map_size)
        self.angleMap=np.ones(self.map_size)*(-1)#负数表示没有值
        self.angleNumberMap=np.zeros(self.map_size)

        for history_rank in range(len(self.history_best_ant_series)):
            #对历史n个最短路径更新信息素
            m_ant=self.history_best_ant_series[history_rank]
            
            # #信息素强度和路径长度在历史路径中的排名有关
            # min_x,max_x=0,len(self.history_path)
            # min_y,max_y=0,1
            # rank=(history_rank-min_x)/(max_x-min_x)
            # if (max_x==min_x):
            #     func_y=max_y
            # else:
            #     func_x=1-rank
            #     func_y=pow(func_x,10)*(max_y-min_y)+min_y
            
            #信息素强度和路径长度倒数有关
            func_x=1/m_ant.path_length
            func_y=1*func_x

            path_max_intensity=func_y*100
            
            #使一条路径上的信息素强度不同(可以帮助识别路径的来去方向)
            #越接近终点强度越大
            for path_index in range(len(m_ant.path)-1):
                #等差式增大
                # min_scaler=0.5
                # max_scaler=1
                # scaler=path_index/len(m_ant.path)*(max_scaler-min_scaler)+min_scaler
                scaler=1
                #填补步长太大导致的空隙

                point1=m_ant.path[path_index]
                point2=m_ant.path[path_index+1]
                new_map=np.zeros(self.map_size)
                cv2.line(img=new_map,
                         pt1=np.flip(point1).astype(int),
                         pt2=np.flip(point2).astype(int),
                         color=path_max_intensity*scaler,
                         thickness=3)
                self.InfoDensityMap=self.InfoDensityMap+new_map

                #信息素半径为1
                # path_angle=math.atan2(point2[1]-point1[1],point2[0]-point1[0])*180/math.pi
                # if path_angle<0:path_angle+=360
                # x=np.linspace(point1[0],point2[0],m_ant.step_length*3)
                # y=np.linspace(point1[1],point2[1],m_ant.step_length*3)
                # points=[(round(x[i]),round(y[i])) for i in range(len(x))]
                # #去除重复
                # seen = set()
                # result = []
                # for point in points:
                #     if point not in seen:
                #         seen.add(point)
                #         result.append(list(point))
                # points=np.array(result)
                # for i in range(len(points)-1):#线段两端端点只更新一个，否则重叠后有问题
                #     point=tuple(points[i])
                #     self.InfoDensityMap[point]+=path_max_intensity*scaler
                #     if self.angleMap[point]==-1:
                #         self.angleMap[point]=0
                #     self.angleMap[point]+=path_angle
                #     self.angleNumberMap[point]+=1
        #去除信息素浓度特别小的点
        max_info_density=np.max(self.InfoDensityMap)
        mask=self.InfoDensityMap<(max_info_density*0.01)
        self.InfoDensityMap[mask]=0

        #方向地图求平均
        mask=self.angleNumberMap>0
        angleMap_updated=np.ones(self.map_size)*(-1)
        angleMap_updated[mask]=self.angleMap[mask]/self.angleNumberMap[mask]
        self.angleMap=angleMap_updated

        self.history_update=False

    def outputSolution(self):
        target_point_radius=1
        ant_color=(255,0,0,255)
        target_color=(241,188,202,255)
        info_color=(0,0,255)

        #复制原始地形
        imageArray=np.copy(self.terrainMap)
        imageArray=cv2.cvtColor(imageArray, cv2.COLOR_RGB2RGBA)

        if not (self.iteration_count%100==0):
            print("iter:{0}".format(self.iteration_count))
            return self.im

        #如果没有信息素，只绘制蚂蚁和终点
        max_info_density=np.max(self.InfoDensityMap)
        if max_info_density==0:
            for m_ant in self.ant_series:
                x=m_ant.current_pos_x
                y=m_ant.current_pos_y
                imageArray[round(x),round(y)]=ant_color

            imageArray=cv2.transpose(imageArray)
            cv2.circle(imageArray,
                       (self.target_x,self.target_y),
                       radius=target_point_radius,
                       color=target_color,
                       thickness=-1,
                       lineType=cv2.LINE_AA)
            imageArray=cv2.transpose(imageArray)
            
            imageArray=cv2.transpose(imageArray)
            self.im.set_array(imageArray)
            return self.im

        #绘制信息素
        info_update_points=[]
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                # if self.InfoDensityMap[i][j]/max_info_density<0.01:continue
                if self.InfoDensityMap[i][j]/max_info_density==0:continue
                info_update_points.append((i,j))

        #归一化
        color_info_density=[self.InfoDensityMap[point] for point in info_update_points]
        max_color=np.max(color_info_density)
        min_color=np.min(color_info_density)
        color_info_density=(color_info_density-min_color)/(max_color-min_color)
        # color_info_density=color_info_density*0.5+0.5 

        color_map = np.zeros((len(info_update_points), 4))  # 创建颜色数组
        color_map[:,0:2+1]=info_color  # 将全部的蓝色通道设为255
        color_map[:,3]=color_info_density*255  # 将不透明度设为信息素浓度
        
        for i in range(len(info_update_points)):
            point=info_update_points[i]
            imageArray[point[0],point[1]]=tuple(color_map[i])

        #绘制蚂蚁
        for m_ant in self.ant_series:
            x=m_ant.current_pos_x
            y=m_ant.current_pos_y
            imageArray[round(x),round(y)]=ant_color
        #绘制终点
        imageArray=cv2.transpose(imageArray)
        cv2.circle(imageArray,
                   (self.target_x,self.target_y),
                   radius=target_point_radius,
                   color=target_color,
                   thickness=-1,
                   lineType=cv2.LINE_AA)
        imageArray=cv2.transpose(imageArray)

        imageArray=cv2.transpose(imageArray)
        self.im.set_array(imageArray)
        return self.im
    
    def new_ant(self,
                angle=None):
        if angle is None:
            angle=get_max_infoDensity_angle(self.InfoDensityMap,
                                            self.start_x,
                                            self.start_y,
                                            radius=10)

        m_ant=ant(obstacle_map=self.ObstacleMap,
                  start_angle=angle,
                  start_x=self.start_x,
                  start_y=self.start_y,
                  target_x=self.target_x,
                  target_y=self.target_y,
                  map_size=self.map_size)
        return m_ant


def get_max_infoDensity_angle(map,x,y,radius=3):
    #找点圆形范围内信息素最大的方向
    radius=3
    search_map=map[x-radius:x+radius+1,
                   y-radius:y+radius+1]
    #创建一个0~360度的数组
    #对圆形范围内的所有像素点进行投票
    gap=30#区间长度
    bar=np.zeros(int(360/gap))
    #0gap~1gap,1gap~2gap,n-1~ngap
    #共360/gap长度
    #第i对应(i~i+1)*gap范围
    map_size_x=search_map.shape[0]
    map_size_y=search_map.shape[1]
    radius=map_size_x/2
    center_x=(map_size_x-1)/2
    center_y=(map_size_y-1)/2

    if np.max(search_map)==0:
        return random.random()*360

    #debug:显示当前search_map
    # image=cv2.resize(cv2.transpose(np.flip(search_map,1)),(300,300),cv2.INTER_BITS)
    # grid_length=int(300/radius)
    # for i in range(image.shape[1]):
    #     for j in range(image.shape[0]):
    #         if (i%grid_length==0)or(j%grid_length==0):
    #             image[j, i] = np.max(image)
    # cv2.imshow('123',image)

    for i in range(map_size_x):
        for j in range(map_size_y):
            if search_map[i,j]==0:continue
            dx=i-center_x
            dy=j-center_y
            if(math.sqrt(dx**2+dy**2)>radius):continue
            angle=math.atan2(dy,dx)*180/math.pi
            if angle<0:angle+=360#-180~0转换到180~360
            index=int(angle/gap)
            bar[index]+=search_map[i,j]
    #得出最大区间
    max_index=np.argmax(bar)
    max_angle=(max_index+0.5)*gap
    #print("spawn-angle:{0:.2f}".format(max_angle))
    return max_angle

alogrithm=None

def figure_update(iteration_count):
    global alogrithm
    alogrithm.iterate()

    plot=True
    if plot:
        alogrithm.outputSolution()

    return alogrithm.im

if __name__=='__main__':
    terrain_image=cv2.imread('map2.png')
    alogrithm=continous_ant_alogrithm(terrain_image=terrain_image,
                                        target_x=250,
                                        target_y=250,
                                        start_x=150,
                                        start_y=150)
    
    # terrain_image=cv2.imread('map7-50x50.png')
    # alogrithm=continous_ant_alogrithm(terrain_image=terrain_image,
    #                              target_x=45,
    #                              target_y=45,
    #                              start_x=25,
    #                              start_y=25)

    ani = FuncAnimation(alogrithm.fig,
                        figure_update,
                        frames=range(1000000),
                        interval=1,
                        blit=False)
    plt.show()


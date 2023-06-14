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
                 sense_radius=6,
                 random_angle_range=10,
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
        self.sense_radius=sense_radius
        self.random_angle_range=random_angle_range
        self.end_threshold=end_threshold

    def judgeDirection(self,
                       InfoDensityMap):
        #从前方区域中找信息素浓度较高的地方前进
        x=self.current_pos_x
        y=self.current_pos_y
        current_angle=self.facing_angle
        map_size_x=InfoDensityMap.shape[0]
        map_size_y=InfoDensityMap.shape[1]

        sense_radius=self.sense_radius#信息素搜集半径
        #备选方向
        left_angle=30
        right_angle=-30
        random_angle_range=self.random_angle_range#角度随机范围:+r~-r
        #TODO: 改为用最大信息素方向函数
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
        dist=math.sqrt((x-tx)**2+(y-ty)**2)
        if (dist<self.end_threshold):return True
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
        #TODO: 是否改回法线策略
        #障碍检测
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
                 ant_limit=200,
                 history_max_length=10,
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
        #TODO: 解决路径方向性问题，增加成功率

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

    def InitializePathGraph(self,
                            step_size=20,
                            threshold=20,
                            limit_try=20000,
                            plot=True):
        #TODO: 添加RRT类
        #导出路径
        123

    def iterate(self):
        # print("iter:"+str(iteration_count))

        if len(self.ant_series)<self.ant_limit:
            #数量没达到上限之前，每次迭代增加一只蚂蚁
            self.ant_series.append(self.new_ant())

        #对所有蚂蚁推进时间
        for i in range(len(self.ant_series)):
            m_ant=self.ant_series[i]
            m_ant.judgeDirection(self.InfoDensityMap)
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

                print("current:{0:.2f},best:{1:.2f}".format(m_ant.path_length,
                                                            self.history_best_ant_series[0].path_length))

            #路径太长，没有继续计算价值，则清除
            min_length=np.inf
            if len(self.history_best_ant_series)>0:
                min_length=self.history_best_ant_series[0].path_length
            if m_ant.path_length>min_length*2:
                self.ant_series[i]=self.new_ant()
        
        #迭代完所有蚂蚁以后，更新信息素
        self.updateInfoDensity()

    def updateInfoDensity(self):
        if not self.history_update:
            return
        
        self.InfoDensityMap=np.zeros(self.map_size)
        for history_rank in range(len(self.history_best_ant_series)):
            #对历史n个最短路径更新信息素
            m_ant=self.history_best_ant_series[history_rank]
            
            #信息素强度和路径长度在历史路径中的排名有关
            min_x,max_x=0,len(self.history_path)
            min_y,max_y=0,1
            rank=(history_rank-min_x)/(max_x-min_x)
            if (max_x==min_x):
                func_y=max_y
            else:
                func_x=1-rank
                func_y=pow(func_x,10)*(max_y-min_y)+min_y
            # print("len:{0:.2f},y:{1:.2f},best:{2:.2f}".format(pathLength,func_y,history_path[0]))

            path_max_intensity=func_y*100
            
            #使一条路径上的信息素强度不同(可以帮助识别路径的来去方向)
            #越接近终点强度越大
            for path_index in range(len(m_ant.path)-1):
                #TODO: 步长为1有可能漏，改成0.5
                #等差式增大
                min_scaler=0.5
                max_scaler=1
                scaler=path_index/len(m_ant.path)*(max_scaler-min_scaler)+min_scaler
                #填补步长太大导致的空隙
                point1=m_ant.path[path_index]
                point2=m_ant.path[path_index+1]
                x=np.linspace(point1[0],point2[0],m_ant.step_length+1)
                y=np.linspace(point1[1],point2[1],m_ant.step_length+1)
                points=[(round(x[i]),round(y[i])) for i in range(m_ant.step_length+1)]
                points=list(set(points))#去除重复
                for point in points:
                    self.InfoDensityMap[point]+=path_max_intensity*scaler
        self.history_update=False

    def outputSolution(self):
        target_point_radius=1
        ant_color=(255,0,0,255)
        target_color=(241,188,202,255)
        info_color=(0,0,255)

        #复制原始地形
        imageArray=np.copy(self.terrainMap)
        imageArray=cv2.cvtColor(imageArray, cv2.COLOR_RGB2RGBA)

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
                if self.InfoDensityMap[i][j]/max_info_density<0.01:continue
                info_update_points.append((i,j))

        #归一化
        color_info_density=[self.InfoDensityMap[point] for point in info_update_points]
        max_color=np.max(color_info_density)
        min_color=np.min(color_info_density)
        color_info_density=(color_info_density-min_color)/(max_color-min_color)

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

    def get_max_infoDensity_angle(self,x,y,radius=3):
        #找点圆形范围内信息素最大的方向
        radius=3
        search_map=self.InfoDensityMap[x-radius:x+radius+1,
                                       y-radius:y+radius+1]
        #创建一个0~360度的数组
        #对圆形范围内的所有像素点进行投票
        gap=10#区间长度
        bar=np.zeros((int(360/gap)-1))
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
        # image=cv2.resize(cv2.transpose(np.flip(search_map,1)),(280,280),cv2.INTER_BITS)
        # for i in range(image.shape[1]):
        #     for j in range(image.shape[0]):
        #         if (i%40==0)or(j%40==0):
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
    
    def new_ant(self,
                angle=None):
        if angle is None:
            angle=self.get_max_infoDensity_angle(self.start_x,self.start_y)

        m_ant=ant(obstacle_map=self.ObstacleMap,
                  start_angle=angle,
                  start_x=self.start_x,
                  start_y=self.start_y,
                  target_x=self.target_x,
                  target_y=self.target_y,
                  map_size=self.map_size)
        return m_ant



alogrithm=None

def figure_update(iteration_count):
    global alogrithm
    alogrithm.iterate()

    plot=True
    if plot:
        alogrithm.outputSolution()

    return alogrithm.im

if __name__=='__main__':
    terrain_image=cv2.imread('map3.png')
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
                        frames=range(10000),
                        interval=1,
                        blit=False)
    plt.show()


import numpy as np
import random
import math
import matplotlib.pyplot as plt
import cv2
import threading
from matplotlib.animation import FuncAnimation
import time
import datetime
import os

class ant:
    def __init__(self,
                 terrain_x,
                 terrain_y,
                 accuracy,
                 start_x_CoM=920.0,
                 start_y_CoM=424.3,
                 start_pitch=0,
                 target_x_CoM=2340,
                 target_y_CoM=924.3,
                 end_threshold=5,
                 ):
        
        self.path=[]
        self.path.append([start_x_CoM,start_y_CoM,start_pitch])
        self.current_x_CoM=start_x_CoM
        self.current_y_CoM=start_y_CoM
        self.current_pitch=start_pitch
        self.target_x_CoM=target_x_CoM
        self.target_y_CoM=target_y_CoM

        self.end_threshold=end_threshold

        self.terrain_x=terrain_x
        self.terrain_y=terrain_y
        self.accuracy=accuracy

        self.body_length=1040
        self.leg_length1=300
        self.leg_length2=300


        foot1=[400,0]
        foot2=[660,0]
        foot3=[1180,0]
        foot4=[1440,0]
        self.foot_path=[]
        self.foot_path.append([foot1,foot2,foot3,foot4])
        self.hip_path=[]
        self.hip_path.append(self.calculate_hip_position())

        self.path_length=0

    def calculatePathLength(self):
        #计算路径长度指标
        path_length=0
        for i in range(len(self.path)-1):
            point1=self.path[i]
            point2=self.path[i+1]
            pos1=point1[0:2]
            pos2=point2[0:2]
            path_length+=math.sqrt(np.sum(pow(pos1-pos2,2)))

        self.path_length=path_length


        #TODO:除了机身总和路径长度，还可以加其他指标
        #轨迹连续性
        #运动学约束余量

        return self.path_length
    
    def foot_GCS2BCS(self,
                     foot_pos_GCS,
                     hip_pos_GCS):

        pitch=self.path[-1][2]/180*math.pi
        dx=foot_pos_GCS[0]-hip_pos_GCS[0]
        dy=foot_pos_GCS[1]-hip_pos_GCS[1]
        
        #HCS y轴向下
        HCS_x=dx*math.cos(pitch)+dy*math.sin(pitch)
        HCS_y=dx*math.sin(pitch)-dy*math.cos(pitch)

        return [HCS_x,HCS_y]

    def inverseKinematics(self,
                          foot_pos_GCS,
                          hip_pos_GCS,
                          leg_configuration=1):
        """
        返回关节角度(rad),-1左侧,1右侧
        """

        l1=self.leg_length1
        l2=self.leg_length2
        
        x,y=self.foot_GCS2BCS(foot_pos_GCS,hip_pos_GCS)
        dist=math.sqrt(x**2+y**2)

        alpha=math.acos((dist**2+l1**2-l2**2)/(2*dist*l1))
        beta=math.acos((l1**2+l2**2-dist**2)/(2*l2*l1))

        phi=math.atan2(y,x)

        if leg_configuration==1:
            #右侧
            theta_thigh=phi-alpha
            theta_shank=-(beta-theta_thigh)
        elif leg_configuration==-1:
            #左侧
            theta_thigh=phi+alpha
            theta_shank=-(2*math.pi-beta-theta_thigh)
        else:
            Exception("关节方向未指定")

        return [theta_thigh,theta_shank]

    def calculate_knee_postion(self,leg_num=-1):
        hip_point=self.hip_path[-1][leg_num]
        foot_point=self.foot_path[-1][leg_num]
        pitch=self.path[-1][2]/180*math.pi

        if leg_num<=1:
            leg_configuration=-1
        else:
            leg_configuration=1
        [theta_thigh,theta_shank]=self.inverseKinematics(foot_point,hip_point,leg_configuration)
        kx_HCS=self.leg_length1*math.cos(theta_thigh)
        ky_HCS=self.leg_length1*math.sin(theta_thigh)

        kx_GCS=hip_point[0]+kx_HCS*math.cos(pitch)+ky_HCS*math.sin(pitch)
        ky_GCS=hip_point[1]+kx_HCS*math.sin(pitch)-ky_HCS*math.cos(pitch)

        return [kx_GCS,ky_GCS]

    def obstacleDetection(self,
                          foot_pos,
                          hip_pos,
                          leg_num=0,
                          leg_configuration=0):
        #检测有没有不符合运动学约束

        #比如落脚点超出范围
        #检测是否离髋关节超过2l就行
        foot_pos=np.array(foot_pos)
        hip_pos=np.array(hip_pos)
        hip_foot_dist=math.sqrt(np.sum(pow(foot_pos-hip_pos,2)))
        if hip_foot_dist>2*self.leg_length1:
            return True
        
        return False

    def walk(self):
        #根据信息素，生成下一步

        regenerate_flag=True

        while(regenerate_flag):

            #生成机身倾角
            next_pitch=self.current_pitch+(random.random()*5-2.5)
            
            #生成质心位置，假设只能向前走
            step_CoM=random.random()*5
            angle_down=-90
            angle_up=90
            angle_CoM=random.random()*(angle_up-angle_down)+angle_down
            next_x_CoM=self.current_x_CoM+step_CoM*math.cos(angle_CoM/180*math.pi)
            next_y_CoM=self.current_y_CoM+step_CoM*math.sin(angle_CoM/180*math.pi)

            #生成四个落脚点
            #必须在地形上取点
            #获取当前位置在地形的x索引
            foot_current_pos=np.array(self.foot_path)[-1]
            foot_x=foot_current_pos[:,0]
            foot_current_index=(foot_x/self.accuracy).astype(int)
            next_foot_path=[]
            regenerate_flag=False
            #随机前进或后退几步accuracy距离
            for leg_num in range(len(foot_x)):
                foot_step=random.randint(-5,5)
                foot_next_index=foot_current_index[leg_num]+foot_step
                foot_next=[self.terrain_x[foot_next_index],
                           self.terrain_y[foot_next_index]]
                hip=self.calculate_hip_position(leg_num=leg_num)
                try_count=0
                while self.obstacleDetection(foot_next,hip,leg_num):
                    #重新生成
                    foot_step=random.randint(-5,5)
                    foot_next_index=foot_current_index+foot_step
                    foot_next=[self.terrain_x[foot_next_index],
                            self.terrain_y[foot_next_index]]
                    hip=self.calculate_hip_position(leg_num=leg_num)

                    try_count+=1
                    if try_count>10:
                        #有可能是因机身位置不好导致怎么生成都失败
                        #尝试很多次了以后应该重新生成机身
                        regenerate_flag=True
                        break
                if regenerate_flag:break

                next_foot_path.append(foot_next)
            
            if regenerate_flag==False:
                #生成成功
                self.current_x_CoM=next_x_CoM
                self.current_y_CoM=next_y_CoM
                self.current_pitch=next_pitch
                self.path.append([next_x_CoM,next_y_CoM,next_pitch])
                self.foot_path.append(next_foot_path)
                self.hip_path.append(self.calculate_hip_position())

        return
    
    def isEnd(self):
        #如果离终点距离小于阈值，则停止
        dx=self.current_x_CoM-self.target_x_CoM
        dy=self.current_y_CoM-self.target_y_CoM
        dist=math.sqrt(dx**2+dy**2)
        if dist<self.end_threshold:
            return True
        return False

    def calculate_hip_position(self,leg_num=-1):
        #根据质心位置和俯仰角计算髋关节的坐标
        cx=self.current_x_CoM
        cy=self.current_y_CoM
        pitch=self.current_pitch/180*math.pi
        l=self.body_length

        hip1_x=cx-l/2*math.cos(pitch)
        hip1_y=cy-l/2*math.sin(pitch)
        hip1=[hip1_x,hip1_y]

        hip2_x=cx-l/4*math.cos(pitch)
        hip2_y=cy-l/4*math.sin(pitch)
        hip2=[hip2_x,hip2_y]
        
        hip3_x=cx+l/4*math.cos(pitch)
        hip3_y=cy+l/4*math.sin(pitch)
        hip3=[hip3_x,hip3_y]

        hip4_x=cx+l/2*math.cos(pitch)
        hip4_y=cy+l/2*math.sin(pitch)
        hip4=[hip4_x,hip4_y]
        
        hip=[hip1,hip2,hip3,hip4]

        if leg_num==0:
            return hip1
        elif leg_num==1:
            return hip2
        elif leg_num==2:
            return hip3
        elif leg_num==3:
            return hip4
        else:
            return hip
    
class antAlogorithm:
    def __init__(self,
                 ant_limit=500,
                 accuracy=1,
                 terrain_type=1,
                 plot=True,
                 history_max_length=10,
                 ):
        self.ant_limit=ant_limit
        self.accuracy=accuracy
        self.iteration_count=0
        self.terrain_type=terrain_type

        
        self.InitializeTerrain()

        # image_size_x=int(np.max(self.terrain_x)/accuracy)
        # image_size_y=int(np.max(self.terrain_y)/accuracy)
        image_size_x=int(np.max(self.terrain_x))
        image_size_y=int(np.max(self.terrain_y))*2

        self.plot=plot 
        if self.plot:
            fig,axes = plt.subplots()
            fig.set_size_inches([7,7]) # 设置图像大小
            axes.set_xlim(0, image_size_x)
            axes.set_ylim(0-200, image_size_y)
            self.fig=fig
            self.axes=axes

        
        terrain_image=np.zeros((image_size_x,image_size_y),dtype=np.uint8)
        terrain_image=cv2.transpose(terrain_image)

        for line in self.terrain:
            point1=line[0]
            point2=line[1]
            cv2.line(img=terrain_image,
                     pt1=point1,
                     pt2=point2,
                     color=1,
                     thickness=5,
                     lineType=cv2.LINE_8)
        #cv2和plt上下相反
        # terrain_image=cv2.flip(terrain_image,0)
        self.terrain_image=terrain_image
        #FIXME:颜色0和1反了
        self.im=axes.imshow(self.terrain_image,cmap="binary",origin="upper")
        plt.pause(0.01)

        self.ant_series=[self.new_ant()]
        self.history_best_ant_series=[]
        self.history_max_length=history_max_length

    def InitializeTerrain(self):
        #生成地形(落脚点的可行域)，由几段直线组成
        if self.terrain_type==1:
            #台阶地形
            stair_height=350
            line1=[[0,0],[1780,0]]
            line2=[[1780,stair_height],[3600,stair_height]]
            terrain=[line1,line2]

        x_accuracy=self.accuracy
        terrain_x=[]
        terrain_y=[]
        for i in range(len(terrain)):
            line_point1=terrain[i][0]
            line_point2=terrain[i][1]
            points_number=int(abs(line_point1[0]-line_point2[0])/x_accuracy)

            terrain_x_1=np.linspace(line_point1[0],line_point2[0],points_number)
            terrain_y_1=np.linspace(line_point1[1],line_point2[1],points_number)

            terrain_x=np.concatenate((terrain_x,terrain_x_1))
            terrain_y=np.concatenate((terrain_y,terrain_y_1))

        self.terrain=terrain
        self.terrain_x=terrain_x
        self.terrain_y=terrain_y

    def resetProblem(self):

        return

    def iterate(self):
        #每轮迭代需要生成所有蚂蚁的路径，然后更新信息素
        
        self.iteration_count+=1

        if len(self.ant_series)<self.ant_limit:
            #数量没达到上限之前，每次迭代增加一只蚂蚁
            self.ant_series.append(self.new_ant())
        
        #对所有蚂蚁推进时间
        for i in range(len(self.ant_series)):
            m_ant=self.ant_series[i]
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

                    print("current:{0:.2f},best:{1:.2f},bad:{2:.2f},iter:{3}".format(
                        m_ant.path_length,
                        self.history_best_ant_series[0].path_length,
                        self.history_best_ant_series[-1].path_length,
                        self.iteration_count))


            #路径太长，没有继续计算价值，则清除
            min_length=np.inf
            max_length=np.inf
            if len(self.history_best_ant_series)>0:
                min_length=self.history_best_ant_series[0].path_length
                max_length=self.history_best_ant_series[-1].path_length
            case1=m_ant.path_length>min_length*2
            case2=(not len(self.history_best_ant_series)<self.history_max_length)\
                    and (m_ant.path_length>max_length)
            if case1 or case2:
                self.ant_series[i]=self.new_ant()
        
        #迭代完所有蚂蚁以后，更新信息素
        # self.updateInfoDensity()

    def updateInfoDensity(self):
        #TODO:
        123

    def outputSolution(self,text=False):
        #输出
        body_color=(255,0,0)

        imageArray=np.copy(self.terrain_image)
        imageArray=cv2.cvtColor(imageArray, cv2.COLOR_GRAY2RGB)

        for m_ant in self.ant_series:
            body_current_pos=m_ant.path[-1]
            foot_current_pos=m_ant.foot_path[-1]
            hip=m_ant.calculate_hip_position()

            #画机身
            body_x,body_y,pitch=body_current_pos
            body1=hip[0]
            body2=hip[3]
            cv2.line(img=imageArray,
                     pt1=np.round(body1).astype(int),
                     pt2=np.round(body2).astype(int),
                     color=body_color,
                     thickness=10,
                     lineType=cv2.LINE_8
                     )
            #四个髋关节
            for hip_point in hip:
                cv2.circle(img=imageArray,
                           center=np.round(hip_point).astype(int),
                           radius=5,
                           color=(255,255,255),
                           thickness=cv2.FILLED)
            #四个脚所在的点
            for foot_point in foot_current_pos:
                cv2.circle(img=imageArray,
                           center=np.round(foot_point).astype(int),
                           radius=5,
                           color=(255,255,255),
                           thickness=cv2.FILLED)
            #大小腿连杆
            for leg_num in range(4):
                knee_point=m_ant.calculate_knee_postion(leg_num)
                cv2.circle(img=imageArray,
                           center=np.round(knee_point).astype(int),
                           radius=5,
                           color=(255,255,255),
                           thickness=cv2.FILLED)
                cv2.line(img=imageArray,
                     pt1=np.round(knee_point).astype(int),
                     pt2=np.round(hip[leg_num]).astype(int),
                     color=body_color,
                     thickness=10,
                     lineType=cv2.LINE_8
                     )
                cv2.line(img=imageArray,
                     pt1=np.round(knee_point).astype(int),
                     pt2=np.round(foot_current_pos[leg_num]).astype(int),
                     color=body_color,
                     thickness=10,
                     lineType=cv2.LINE_8
                     )
        self.im.set_array(imageArray)

        return self.im

    def new_ant(self):
        m_ant=ant(self.terrain_x,
                  self.terrain_y,
                  self.accuracy)
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

    alogrithm=antAlogorithm(ant_limit=1,
                            accuracy=10)

    ani = FuncAnimation(alogrithm.fig,
                        figure_update,
                        frames=range(1000000),
                        interval=1,
                        blit=False)
    plt.show()
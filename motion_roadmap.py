#coding=utf-8
import cv2
import numpy as np
import motion_planning_toolbox as mpt
import math
import os
import time

class MotionRoadmap(object):
    def __init__(self, map_img):
        ## 初始化实例，需要输入一张 bmp 格式的地图
        self.map = map_img
        # 读取图像尺寸
        size = self.map.shape
        # 运动规划的起点
        self.point_strat = np.mat([0, 0])
        # 运动规划的终点
        self.point_goal = np.mat([size[0] - 1, size[1] - 1])
                    
    def rrt_planning(self, **param):
        ''' 快速扩展随机树算法（RRT算法）。
        
        本函数可对 self.map 进行 RRT 规划并绘制图像。
        
        Args:
            **param: 关键字参数，用以配置规划参数    
                     s: 搜索步长，默认20。int
                     t: 判断阈值，默认20。float
                     l: 尝试次数。默认20000。int
                     p: 绘制图像指令，缺省表示绘制，’None‘表示不绘制。string
        Return:
            本函数没有返回值，但会根据计算结果赋值（或定义）以下属性变量：
                self.rrt_tree: 所生成的rrt树。numpy.mat
                    数据含义: [[横坐标, 纵坐标, 父节点索引]]，其中第一个点为根，最后一个点为终点梢

        Raises:
            暂无明显的接口错误
        Example:
            mr = MotionRoadmap(img)
            mr.rrt_planning(s=25, t=30, l=15000, p='None')
        
        '''


        print('开始 RRT 路径规划，请等待...')
        # 关键字参数处理
        step_size = 20
        threshold = 20 # 距离阈值，小于此值将被视作同一个点，不可大于 step_size
        limit_try = 20000
        if 's' in param:
            step_size = param['s']
        if 't' in param:
            threshold = param['t']
        if 'l' in param:
            limit_try = param['l']
        if not ('p' in param):
            param['p'] = True
        # 地图灰度化
        image_gray = cv2.cvtColor(self.map, cv2.COLOR_BGR2GRAY)
        # 地图二值化
        ret,img_binary = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY) 

        # 初始化 RRT 树:[横坐标，纵坐标，父节点索引]
        rrt_tree = np.hstack((self.point_strat, [[0]]))

        # 初始化尝试次数
        num_try = 0
        # 成功标识
        path_found = False
        
        while num_try <= limit_try:
            ## 向随机方向或终点生长
            if np.random.rand() < 0.5:
                    #在地图范围内随机采样一个像素
                    sample = np.mat(np.random.randint(0, img_binary.shape[0] - 1, (1, 2)))
            else:
                    #或者直接取终点
                    sample = self.point_goal
            
            ## 选择rrt树中离采样点最近的点
            # 计算各点与采样点的距离
            mat_distance = mpt.straight_distance(rrt_tree[:, 0 : 2], sample)
            # 距离最小点的索引
            index_close = np.argmin(mat_distance, 0)[0, 0] #末尾索引用以取数值，否则为矩阵
            point_close = rrt_tree[index_close, 0 : 2]

            ## 从距离最小点向采样点移动 step_size 距离，并进行碰撞检测
            theta_dir = math.atan2(sample[0, 0] - point_close[0, 0], sample[0, 1] - point_close[0, 1])
            point_new = point_close + step_size * np.mat([math.sin(theta_dir), math.cos(theta_dir)])
            # 将坐标化为整数
            point_new = np.around(point_new).astype(np.int32)
            #检测碰撞
            if not mpt.check_path(point_close, point_new, img_binary):
                num_try = num_try + 1
                continue

            ## 距离终点小于阈值距离，则规划成功
            if mpt.straight_distance(point_new, self.point_goal) < threshold*1.5:
                path_found = True
                # 加入到rrt树
                point_new = np.hstack((point_new, [[index_close]]))
                rrt_tree = np.vstack((rrt_tree, point_new))
                break
            
            ## 计算rrt树中各点与新点的距离，如果均大于 threshold 的，则添加新点到rrt树
            #保证新点和树上已有的点不重合
            mat_distance = mpt.straight_distance(rrt_tree[:, 0 : 2], point_new)
            if np.min(mat_distance, 0) < threshold:
                num_try = num_try + 1
                continue
            #不重合，则把新点加入树中
            point_new = np.hstack((point_new, [[index_close]]))
            rrt_tree = np.vstack((rrt_tree, point_new))

            self.plot(self.map, rrt_tree)

        

        if path_found == True:
            print('规划成功！')
            self.rrt_tree = rrt_tree
        else:
            print('没有找到解。')
            self.rrt_tree = rrt_tree

        
        ## 根据关键字确定是否绘图
        if not(param['p'] == 'None'):
            if (path_found == True):
                self.rrt_tree
                mpt.tree_plot(self.map, self.rrt_tree)
            else:
                print('没有找到解，无法绘图。')

    def plot(self,map_img, rrt_tree):
        ## 绘制树形图
        # 画点
        point_size = 2
        point_color = (0, 127, 0) # BGR
        thickness = 1
        # 将矩阵转化为数组并转为整型，再转化为元组，以供cv2使用   
        vertex = np.around(np.array(rrt_tree)).astype(int)
        vertex_tuple = tuple(map(tuple, vertex)) 
        # 画点画线
        for point in vertex_tuple:
            cv2.circle(map_img, point[0 : 2], point_size, point_color, thickness)
            if point[0] != 0:
                cv2.line(map_img, point[0 : 2], vertex_tuple[point[2]][0 : 2], (255,150,150), 1)

        cv2.imshow("123",cv2.resize(cv2.flip(map_img,0),(800,800),cv2.INTER_NEAREST))
        cv2.waitKey(1)

    def get_path(self):
        path_index=[]
        point_a_index = -1
        path_index.append(point_a_index)
        vertex = np.around(np.array(self.rrt_tree)).astype(int)
        while point_a_index != 0:
            point_b_index = int(self.rrt_tree[point_a_index, 2])
            # cv2.line(map_img,vertex_tuple[int(point_a_index)][0 : 2], vertex_tuple[int(point_b_index)][0 : 2],(0,0,255),4)
            point_a_index = point_b_index
            path_index.append(point_a_index)
        path_index=np.array(path_index)
        path_index=np.flip(path_index,0)

        path_nodes=vertex[path_index]
        path_points=[[point[0],point[1]] for point in path_nodes]
        end_point=[self.point_goal[0,0],self.point_goal[0,1]]
        path_points.append(end_point)
        return path_points

if __name__=="__main__": 
    ## 预处理
    # 图像路径

    image_path = os.path.dirname(__file__)+"\\sourceMap\\map_3.bmp"
    # 读取图像
    # img = cv2.imread(image_path)# np.ndarray BGR uint8

    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
    # imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    img = cv2.resize(img,(500,500))
    mr = MotionRoadmap(img)
    #mr.rrt_planning(s=20, t=20, l=15000)
    #mr.point_strat = np.mat([0,99])
    #mr.point_goal = np.mat([95,5])
    mr.rrt_planning()
    

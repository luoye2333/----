import numpy as np
import random
import math
import matplotlib.pyplot as plt

class antAlogorithm:
    def __init__(self,
                 antCount=500,
                 cityCount=50,
                 plot=False
                 ):
        self.antCount=antCount#蚂蚁数量
        self.cityCount=cityCount#目标城市数量
        self.plot=plot 
        if self.plot:
            self.fig, (self.ax1, self.ax2) = plt.subplots(2,1)
            self.fig.set_size_inches([4,7]) # 设置图像大小

        self.cityXpos=None
        self.cityYpos=None
        self.Distance=None#路径邻接矩阵，对称
        self.InfoDensity=None#路径信息素浓度矩阵，对称
        self.antSeries=np.ndarray([antCount],dtype=ant)
        self.iterationCount=0
        
        self.InitializePathGraph()

    def InitializePathGraph(self,
                            xRange=10,
                            yRange=10):
        #生成城市位置
        #随机生成X坐标，Y坐标，计算邻接矩阵

        self.Distance=np.zeros([self.cityCount,self.cityCount])
        cityXpos=np.random.random(size=(self.cityCount)) *xRange
        cityYpos=np.random.random(size=(self.cityCount)) *yRange

        # cityXpos = [50, 20, 70, 10, 60, 30, 80, 40, 90, 50]
        # cityYpos = [50, 80, 80, 20, 30, 60, 50, 90, 30, 20]

        cityXpos=np.array([ 0.        ,  5.82063766,  3.19034237,  7.6186251 ,  9.85843298,
                5.10645158,  1.63912287,  1.58919322,  6.04779042,  2.71757301,
                3.69468567,  3.84587442,  5.58184031,  8.10046059,  2.32104457,
                9.86079294,  2.25398313,  5.78197519,  1.77949771,  9.12443243,
                5.48646522,  7.24862982,  9.76648603,  5.08493055,  8.18403217,
                5.11691039,  1.06765756,  4.51224205,  9.97909674, 10.        ])
        cityYpos=np.array([ 0.        ,  0.53022814,  3.54597357,  2.42138272,  1.05124698,
                7.4271093 ,  4.01568047,  6.43997837,  5.96972804,  1.1774294 ,
                1.298142  ,  7.60908452,  8.59024184,  7.79836957,  7.80889488,
                4.37252154,  3.76985716,  1.03103831,  0.90413774,  2.31882736,
                5.04128013,  5.05152678,  9.18983026,  1.19866566,  3.74219691,
                7.97055398,  8.92362005,  1.76880202,  1.45866429, 10.        ])

        #起点和终点固定坐标
        cityXpos[0]=0
        cityYpos[0]=0
        cityXpos[-1]=xRange
        cityYpos[-1]=yRange
        self.cityXpos=cityXpos
        self.cityYpos=cityYpos

        #计算路径邻接矩阵，利用对称性
        for i in range(self.cityCount):
            for j in range(self.cityCount):
                if(i==j):continue#初始化已经置零
                if(i>j):continue
                self.Distance[i][j]=math.sqrt((cityXpos[i]-cityXpos[j])**2+\
                                              (cityYpos[i]-cityYpos[j])**2)
                self.Distance[j][i]=self.Distance[i][j]

        #贪心算法求解最短路径，获得初始信息素浓度
        visited=np.zeros([self.cityCount],dtype=bool)
        path=np.zeros([self.cityCount],dtype=int)
        path[0]=0
        visited[0]=True
        veryBig=1000
        allLength=0
        for pathIndex in range(self.cityCount):
            if pathIndex==0:continue
            pathStart=path[pathIndex-1]
            minPathLength=veryBig
            next=0
            for pathEnd in range(self.cityCount):
                if visited[pathEnd]:continue
                if self.Distance[pathStart][pathEnd]<minPathLength:
                    allLength=self.Distance[pathStart][pathEnd]
                    next=pathEnd
            path[pathIndex]=next
            visited[next]=True
            allLength+=minPathLength
        self.InfoDensityInitial=self.antCount/allLength
        #初始化信息素浓度   
        self.InfoDensity=np.ones([self.cityCount,self.cityCount])*self.InfoDensityInitial 

    def resetProblem(self):
        self.InfoDensity=np.ones([self.cityCount,self.cityCount])*self.InfoDensityInitial 
        self.iterationCount=0

    def iterate(self):
        #每轮迭代需要生成所有蚂蚁的路径，然后更新信息素
        self.antSeries=np.ndarray([self.antCount],dtype=ant)
        
        for antIndex in range(self.antCount):
            #模拟每个蚂蚁的路径
            a=ant(self.cityCount)
            a.generatePath(self.Distance,self.InfoDensity)
            a.calculatePathLength(self.Distance)
            self.antSeries[antIndex]=a
        
        self.updateInfoDensity()
        self.iterationCount+=1

    def updateInfoDensity(self):
        rhoEvaporate=0.5
        self.InfoDensity=np.multiply(self.InfoDensity,rhoEvaporate)

        #每条蚂蚁走过的整条路径 根据路径长度的倒数 增加信息素
        rhoTraverse=1#倍率
        for antIndex in range(self.antCount):
            for city in range(self.cityCount-1):
                ant=self.antSeries[antIndex]
                pathStart=ant.citySeries[city]
                pathEnd=ant.citySeries[city+1]
                self.InfoDensity[pathStart][pathEnd]+=rhoTraverse*1.0/ant.pathLength
                self.InfoDensity[pathEnd][pathStart]+=rhoTraverse*1.0/ant.pathLength#邻接矩阵正反两条路都要更新

    def outputSolution(self,text=False):
        #输出当前情况下找到的最优解

        minPath=self.antSeries[1].pathLength
        minPathIndex=1
        for i in range(self.antCount):
            d=self.antSeries[i].pathLength
            if (d<minPath):
                minPathIndex=i
                minPath = d
        print("iteration:{},length:{:2f}".format(self.iterationCount,minPath))
        
        if text:
            antMinPath=self.antSeries[minPathIndex]
            for city in range(self.cityCount-1):
                pathStart=antMinPath.citySeries[minPathIndex][city]
                pathEnd=antMinPath.citySeries[minPathIndex][city+1]
                print("{0}->{1} : {2:2f}".format(pathStart,
                                                 pathEnd,
                                                 self.Distance[pathStart][pathEnd]))

        if self.plot:
            self.ax1.clear()
            self.ax1.scatter(self.cityXpos, self.cityYpos, color="blue")# 绘制城市/点
            
            antMinPath=self.antSeries[minPathIndex]
            x=self.cityXpos[antMinPath.citySeries]
            y=self.cityYpos[antMinPath.citySeries]
            self.ax1.plot(x,y, color="red", linewidth=2)

        # for city in range(self.cityCount-1):
        #     pathStart=citySeries[minPathIndex][city]
        #     pathEnd=citySeries[minPathIndex][city+1]
        #     start_x=cityXpos[pathStart]
        #     start_y=cityYpos[pathStart]
        #     end_x=cityXpos[pathEnd]
        #     end_y=cityYpos[pathEnd]
        #     self.ax1.plot([start_x, end_x], [start_y, end_y], color="red", linewidth=2)

        #输出信息素浓度
        infoOutput=True
        if infoOutput:
            maxInfoDensity=np.max(self.InfoDensity)
            self.ax2.clear()
            self.ax2.scatter(self.cityXpos, self.cityYpos, color="blue")# 绘制城市/点
            for i in range(self.cityCount):
                for j in range(self.cityCount):
                    if i<=j :continue #只需要上三角
                    start_x=self.cityXpos[i]
                    start_y=self.cityYpos[i]
                    end_x=self.cityXpos[j]
                    end_y=self.cityYpos[j]
                    pathInfoDensity=self.InfoDensity[i][j]
                    pathColor=(1,0,0,pathInfoDensity/maxInfoDensity)
                    self.ax2.plot([start_x, end_x], [start_y, end_y], color=pathColor, linewidth=2)
        plt.pause(0.01)

class ant:
    def __init__(self,
                 cityCount):
        self.cityCount=cityCount
        self.visited=np.zeros([self.cityCount],dtype=bool)
        self.citySeries=np.zeros([self.cityCount],dtype=int)
        self.pathLength=0

    def generatePath(self,
                     Distance,
                     InfoDensity,
                     startCity=0,
                     endCity=-1,
                     alpha=1,
                     beta=1):
        #根据概率选择路径
        #概率权重=(路径长度倒数)^a * (信息素浓度)^b

        if (endCity==-1):endCity=self.cityCount-1

        for pathIndex in range(self.cityCount):
            #生成路径就是把城市排序一下，所以只要每次选择下一个城市就行
            if (pathIndex==0):
                #选择起点
                self.visited[startCity]=True
                self.citySeries[pathIndex]=startCity
            elif(pathIndex==self.cityCount-1):
                #最后一段路径必须连接到终点
                self.visited[endCity]=True
                self.citySeries[pathIndex]=endCity
            else:
                #除了起点终点，根据概率选择下一个城市

                pathStart=self.citySeries[pathIndex-1]

                #维护所有可选路径的概率和终点信息
                pathProbability=[]
                availablePath=[]

                for pathEnd in range(self.cityCount):
                    #生成概率序列
                    if self.visited[pathEnd]:
                        #排除已经去过的城市
                        continue
                    if pathEnd==endCity:
                        #中间点不能选择终点城市
                        continue
                    prob=pow(1.0/Distance[pathStart][pathEnd],alpha) * \
                           pow(InfoDensity[pathStart][pathEnd],beta)
                    availablePath.append(pathEnd)
                    pathProbability.append(prob)

                selectedCity=self.choosePath(availablePath,pathProbability)
                self.visited[selectedCity]=True
                self.citySeries[pathIndex]=selectedCity

    def choosePath(self,
                   path,
                   pathProbability):
        sumProb=np.sum(pathProbability)
        #生成随机数
        r=random.random()*sumProb
        probCount=0
        for pathIndex in range(len(path)):
            probCount=probCount+pathProbability[pathIndex]
            if r<probCount:#随机数落在该概率区间内，说明随机到了这条路径
                return path[pathIndex]

    def calculatePathLength(self,
                            Distance):
        pathLength=0
        for city in range(self.cityCount-1):
            pathStart=self.citySeries[city]
            pathEnd=self.citySeries[city+1]
            pathLength+=Distance[pathStart][pathEnd]
            self.pathLength=pathLength
        return self.pathLength


if __name__=='__main__':
    n=antAlogorithm(antCount=300,cityCount=30,plot=True)
    for i in range(100):
        n.iterate()
        n.outputSolution()

    # 防止绘图窗口闪退
    while True:
        try:
            plt.pause(0.1)
        except:
            break
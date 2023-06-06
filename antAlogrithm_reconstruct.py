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
        self.historyBestAntSeries=np.ndarray([antCount],dtype=ant)
        self.lastBestList=[]


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

        #30
        # cityXpos=np.array([ 0.        ,  5.82063766,  3.19034237,  7.6186251 ,  9.85843298,
        #         5.10645158,  1.63912287,  1.58919322,  6.04779042,  2.71757301,
        #         3.69468567,  3.84587442,  5.58184031,  8.10046059,  2.32104457,
        #         9.86079294,  2.25398313,  5.78197519,  1.77949771,  9.12443243,
        #         5.48646522,  7.24862982,  9.76648603,  5.08493055,  8.18403217,
        #         5.11691039,  1.06765756,  4.51224205,  9.97909674, 10.        ])
        # cityYpos=np.array([ 0.        ,  0.53022814,  3.54597357,  2.42138272,  1.05124698,
        #         7.4271093 ,  4.01568047,  6.43997837,  5.96972804,  1.1774294 ,
        #         1.298142  ,  7.60908452,  8.59024184,  7.79836957,  7.80889488,
        #         4.37252154,  3.76985716,  1.03103831,  0.90413774,  2.31882736,
        #         5.04128013,  5.05152678,  9.18983026,  1.19866566,  3.74219691,
        #         7.97055398,  8.92362005,  1.76880202,  1.45866429, 10.        ])

        #100
        # cityXpos=np.array([
        # 0.        ,  3.5511111 ,  9.06230271,  1.4363251 ,  0.3072557 ,
        # 0.60741698,  1.25531899,  9.2397397 ,  7.16401611,  4.19096027,
        # 6.02116748,  8.85038086,  7.34062374,  6.51145736,  6.63984811,
        # 2.25508934,  6.49591261,  9.23104083,  5.76847748,  3.67298779,
        # 2.96157793,  4.53065329,  6.910904  ,  8.53107565,  0.82110921,
        # 3.42505059,  1.87344256,  8.28739528,  4.00711577,  0.95987077,
        # 9.03381517,  7.78941248,  6.03085182,  6.34507872,  0.37732258,
        # 7.30949919,  8.32665457,  7.19181361,  4.11733612,  6.44501184,
        # 3.53581152,  3.80451034,  6.3827732 ,  1.65528398,  7.96593364,
        # 1.30264211,  9.10505169,  7.4595735 ,  8.05304307,  0.7308619 ,
        # 8.17943059,  4.65240538,  7.02849499,  8.44353851,  5.03154221,
        # 7.73331387,  3.93026162,  4.09786625,  2.23788965,  9.95552594,
        # 7.35600294,  4.51225468,  7.10814726,  5.59872169,  0.45533261,
        # 9.83806713,  9.9977801 ,  5.51634475,  6.57874279,  6.94115083,
        # 7.81443311,  7.11451714,  0.48905413,  1.20755124,  7.84561129,
        # 1.74905134,  0.60169453,  1.93118306,  3.62203554,  9.06059892,
        # 1.31172599,  1.7565024 ,  4.40044997,  5.09506255,  2.500262  ,
        # 2.86610873,  8.9970446 ,  1.15730431,  2.35609408,  3.19273281,
        # 4.19031114,  4.32001751,  1.67524795,  1.66805018,  5.15699078,
        # 9.0402412 ,  8.96291963,  8.07128558,  2.40257482, 10.        ])
        # cityYpos=np.array([
        # 0.00000000e+00, 3.50824321e+00, 4.13386929e+00, 1.30093003e+00,
        # 8.84626976e+00, 1.68321382e+00, 2.37998345e+00, 4.24832434e+00,
        # 5.66758647e+00, 6.75487359e+00, 6.49344576e+00, 8.06547827e+00,
        # 9.94942056e+00, 1.32818996e+00, 1.50543614e+00, 3.31077202e+00,
        # 1.73376424e+00, 6.17471392e+00, 4.47151370e-03, 1.33472674e+00,
        # 6.54165138e+00, 6.75071580e-03, 5.51277405e+00, 4.93156222e-01,
        # 9.87201102e+00, 7.57564975e-01, 7.71805191e+00, 6.77194433e+00,
        # 3.53233634e+00, 1.37261348e+00, 4.36782661e+00, 4.91040862e+00,
        # 6.40074255e-01, 1.93856328e+00, 1.06633635e+00, 1.61490469e+00,
        # 2.10032093e+00, 2.22619659e+00, 4.79583024e+00, 9.81576630e+00,
        # 9.36715391e+00, 5.27679115e-03, 6.90375345e+00, 6.19890505e+00,
        # 2.80617809e-01, 8.27300173e+00, 1.20564825e+00, 7.87646652e+00,
        # 5.21072233e+00, 5.23192698e+00, 3.47782476e+00, 3.36601494e+00,
        # 8.32516515e+00, 9.82647224e-01, 3.49659061e+00, 7.14521735e-01,
        # 9.39566251e+00, 3.78602421e+00, 2.87580561e+00, 7.63492476e+00,
        # 4.27381585e+00, 3.06932069e+00, 8.81160794e+00, 2.39429953e+00,
        # 8.93299443e+00, 8.91966928e+00, 6.47515543e+00, 3.00097629e+00,
        # 2.87564453e+00, 1.92935134e+00, 1.07490194e+00, 2.61816849e+00,
        # 6.64781119e+00, 9.25358619e+00, 5.50728954e+00, 1.33920678e+00,
        # 7.74121747e+00, 7.95955095e+00, 8.68222804e+00, 1.72649428e+00,
        # 4.73694527e+00, 7.68039102e+00, 6.31904236e-01, 2.35785837e+00,
        # 1.03622876e+00, 9.52300627e+00, 8.50584406e+00, 6.00615620e+00,
        # 8.71565630e+00, 1.52896427e-01, 2.88960579e+00, 3.34669705e+00,
        # 8.95686182e+00, 1.90165292e+00, 8.06277032e+00, 8.27150526e+00,
        # 7.62986816e+00, 8.22796986e+00, 9.31010265e+00, 1.00000000e+01])
        #min=75.880152

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
        allLength=0
        for pathIndex in range(self.cityCount):
            if pathIndex==0:continue
            pathStart=path[pathIndex-1]
            minPathLength=np.inf
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

        #对生成的所有蚂蚁按照路径长度排序，方便后续处理
        sorted_indices = np.argsort([ant.pathLength for ant in self.antSeries])
        sorted_antSeries = self.antSeries[sorted_indices]
        self.antSeries=sorted_antSeries

        self.updateInfoDensity()
        self.iterationCount+=1
        self.outputSolution()
        return self.isEndSituation()

    def updateInfoDensity(self):
       
        #朴素策略O(n^2)
        #每条蚂蚁走过的整条路径 根据路径长度的倒数 增加信息素
        # rhoEvaporate=0.5
        # self.InfoDensity=np.multiply(self.InfoDensity,rhoEvaporate)
        # rhoTraverse=1#倍率
        # for antIndex in range(self.antCount):
        #     for city in range(self.cityCount-1):
        #         ant=self.antSeries[antIndex]
        #         pathStart=ant.citySeries[city]
        #         pathEnd=ant.citySeries[city+1]
        #         self.InfoDensity[pathStart][pathEnd]+=rhoTraverse*1.0/ant.pathLength
        #         self.InfoDensity[pathEnd][pathStart]+=rhoTraverse*1.0/ant.pathLength#邻接矩阵正反两条路都要更新

        #策略2：根据路径长度排序，只对前10%短的路径更新信息素（计算时间减少，收敛速度加快）
        # rhoEvaporate=0.5
        # self.InfoDensity=np.multiply(self.InfoDensity,rhoEvaporate)
        # rhoTraverse=1#倍率
        # for antIndex in range(int(self.antCount*0.1)):
        #     for city in range(self.cityCount-1):
        #         ant=self.antSeries[antIndex]
        #         pathStart=ant.citySeries[city]
        #         pathEnd=ant.citySeries[city+1]
        #         self.InfoDensity[pathStart][pathEnd]+=rhoTraverse*1.0/ant.pathLength
        #         self.InfoDensity[pathEnd][pathStart]+=rhoTraverse*1.0/ant.pathLength#邻接矩阵正反两条路都要更新

        #策略3：对路径使用softmax归一化映射，放大路径差距（不容易收敛）
        # rhoEvaporate=0.5
        # self.InfoDensity=np.multiply(self.InfoDensity,rhoEvaporate)
        # sorted_path=[ant.pathLength for ant in self.antSeries]
        # sorted_path-=np.max(sorted_path)#防止溢出
        # softmax_sum=np.sum(np.exp(1.0/sorted_path[0:int(self.antCount*0.1)]))
        # rhoTraverse=1#倍率
        # for antIndex in range(int(self.antCount*0.1)):
        #     for city in range(self.cityCount-1):
        #         ant=self.antSeries[antIndex]
        #         pathStart=ant.citySeries[city]
        #         pathEnd=ant.citySeries[city+1]
        #         #邻接矩阵正反两条路都要更新
        #         self.InfoDensity[pathStart][pathEnd]+=rhoTraverse*np.exp(1.0/ant.pathLength)/softmax_sum
        #         self.InfoDensity[pathEnd][pathStart]+=rhoTraverse*np.exp(1.0/ant.pathLength)/softmax_sum

        #策略4，记忆历史最短路径，RAS(rank ant system)
        # rhoEvaporate=0.5
        # self.InfoDensity=np.multiply(self.InfoDensity,rhoEvaporate)
        # if self.iterationCount>1:
        #     #和历史序列合并
        #     self.antSeries=np.concatenate((self.antSeries,self.historyBestAntSeries))

        # sorted_path=[ant.pathLength for ant in self.antSeries]
        # sorted_path-=np.max(sorted_path)#防止溢出
        # softmax_sum=np.sum(np.exp(1.0/sorted_path[0:int(self.antCount*0.1)]))
        # rhoTraverse=5#倍率
        # for antIndex in range(int(self.antCount*0.1)):
        #     for city in range(self.cityCount-1):
        #         ant=self.antSeries[antIndex]
        #         pathStart=ant.citySeries[city]
        #         pathEnd=ant.citySeries[city+1]
        #         #邻接矩阵正反两条路都要更新
        #         self.InfoDensity[pathStart][pathEnd]+=rhoTraverse*np.exp(1.0/ant.pathLength)/softmax_sum
        #         self.InfoDensity[pathEnd][pathStart]+=rhoTraverse*np.exp(1.0/ant.pathLength)/softmax_sum
        
        # #保存历史前3%短路径
        # self.historyBestAntSeries=self.antSeries[0:int(self.antCount*0.03)]

        #策略5，只保留一条最优路径，EAS(elite ant system)
        rhoEvaporate=0.1
        self.InfoDensity=np.multiply(self.InfoDensity,rhoEvaporate)
        if self.iterationCount>=1:
            #和历史序列合并
            self.antSeries=np.concatenate((self.antSeries,self.historyBestAntSeries))
        #TODO 逻辑可能有问题，不应该把历史最优放进当前批次产生的蚂蚁中

        sorted_path=[ant.pathLength for ant in self.antSeries]
        sorted_path-=np.max(sorted_path)#防止溢出
        softmax_sum=np.sum(np.exp(1.0/sorted_path[0:int(self.antCount*0.1)]))
        rhoTraverse=3#倍率
        for antIndex in range(int(self.antCount*0.1)):
            for city in range(self.cityCount-1):
                ant=self.antSeries[antIndex]
                pathStart=ant.citySeries[city]
                pathEnd=ant.citySeries[city+1]
                #邻接矩阵正反两条路都要更新
                self.InfoDensity[pathStart][pathEnd]+=rhoTraverse*np.exp(1.0/ant.pathLength)/softmax_sum
                self.InfoDensity[pathEnd][pathStart]+=rhoTraverse*np.exp(1.0/ant.pathLength)/softmax_sum
        #保存历史最优
        self.historyBestAntSeries=self.antSeries[0:1]

    def outputSolution(self,text=False):
        #输出当前情况下找到的最优解

        minPath=np.inf
        minPathIndex=-1
        for i in range(len(self.historyBestAntSeries)):
            d=self.historyBestAntSeries[i].pathLength
            if (d<minPath):
                minPathIndex=i
                minPath = d
        print("iteration:{},length:{:2f}".format(self.iterationCount,minPath))
        
        if text:
            antMinPath=self.historyBestAntSeries[minPathIndex]
            for city in range(self.cityCount-1):
                pathStart=antMinPath.citySeries[minPathIndex][city]
                pathEnd=antMinPath.citySeries[minPathIndex][city+1]
                print("{0}->{1} : {2:2f}".format(pathStart,
                                                 pathEnd,
                                                 self.Distance[pathStart][pathEnd]))

        if self.plot:
            self.ax1.clear()
            self.ax1.scatter(self.cityXpos, self.cityYpos, color="blue")# 绘制城市/点
            
            antMinPath=self.historyBestAntSeries[minPathIndex]
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

    def isEndSituation(self):
        #策略1：如果最优路径很久没有变化则结束
        currentBest=self.historyBestAntSeries[0].pathLength
        tolerance=1e-5
        longTimeCount=10
        if len(self.lastBestList)==0:
            self.lastBestList.append(currentBest)
        else:
            lastBest=self.lastBestList[-1]
            if (currentBest-lastBest<tolerance):
                self.lastBestList.append(currentBest)
                if(len(self.lastBestList)>=longTimeCount):
                    return True
            else:#最优解有变化，则重新记录
                self.lastBestList=[currentBest]
            
        #策略2：如果找到的路径和最优路径一样，那么结束
        sorted_path=[ant.pathLength for ant in self.antSeries]
        longTimeCount=10
        for i in range(longTimeCount):
            123#TODO

        #策略3：当满足一次结束条件以后，重置信息素，但保留最优路径，重新计算
        #TODO

        return False
    
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
                    #alpha=0 只和信息素浓度有关，收敛太慢
                    #beta=0 只和路径距离有关，有概率的贪心算法，不收敛
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
    n=antAlogorithm(antCount=100,cityCount=20,plot=True)
    # n=antAlogorithm(antCount=300,cityCount=50,plot=True)
    # n=antAlogorithm(antCount=500,cityCount=100,plot=True)

    for i in range(100):
        exitStatus=n.iterate()
        if exitStatus:break

    # 防止绘图窗口闪退
    while True:
        try:
            plt.pause(0.1)
        except:
            break
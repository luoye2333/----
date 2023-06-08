import numpy as np
import random
import math
import matplotlib.pyplot as plt

class antAlogorithm:
    def __init__(self,
                 antCount=500,
                 accuracy=1,
                 terrain=1,
                 plot=False,
                 ):
        self.antCount=antCount
        self.accuracy=accuracy
        self.iterationCount=0
        self.terrain=terrain

        self.plot=plot 
        if self.plot:
            self.fig, self.axes = plt.subplots(1,1)
            self.fig.set_size_inches([4,7]) # 设置图像大小

        self.InitializeTerrain()
        

    def InitializeTerrain(self):
        #生成地形(落脚点的可行域)，由几段直线组成
        if self.terrain==1:
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

        self.terrain_x=terrain_x
        self.terrain_y=terrain_y



    def resetProblem(self):
        123

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
        self.outputSolution()
        return self.isEndSituation()

    def updateInfoDensity(self):
        123

    def outputSolution(self,text=False):
        #输出当前情况下找到的最优解
        123

    def isEndSituation(self):
     
        return False


    
class ant:
    def __init__(self,
                 ):
        self.COM_x=np.array([920.0])
        self.COM_y=np.array([424.3])
        self.pitch=np.array([0.0])
        self.leg=np.array([[400,660,1180,1440]])

    def generatePath(self):
        123

    def calculatePathLength(self):

        return self.pathLength


if __name__=='__main__':
    isplot=False
    # n=antAlogorithm(antCount=100,cityCount=20,plot=isplot)
    # n=antAlogorithm(antCount=200,cityCount=35,plot=isplot)
    # n=antAlogorithm(antCount=300,cityCount=50,plot=isplot)
    n=antAlogorithm(antCount=500,accuracy=1,plot=isplot,terrain=1)

    while True:
        exitStatus=n.iterate()
        if exitStatus:break

    # 防止绘图窗口闪退
    while True:
        try:
            plt.pause(0.1)
        except:
            break
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import time

AntCount=500#蚂蚁数量
InfoDensityInitial=10#初始信息素浓度
cityCount=50#目标城市数量

# maxDistance=10
# Distance=np.random.random(size=(cityCount, cityCount)) *maxDistance

#初始化路径长度
Distance=np.zeros([cityCount,cityCount])
xRange=10;yRange=10
cityXpos=np.random.random(size=(cityCount)) *xRange
cityYpos=np.random.random(size=(cityCount)) *yRange
cityXpos[0]=0;cityYpos[0]=0
cityXpos[cityCount-1]=xRange;cityYpos[cityCount-1]=yRange

# cityXpos = [50, 20, 70, 10, 60, 30, 80, 40, 90, 50]
# cityYpos = [50, 80, 80, 20, 30, 60, 50, 90, 30, 20]

# cityXpos=[ 0.        ,  5.82063766,  3.19034237,  7.6186251 ,  9.85843298,
#         5.10645158,  1.63912287,  1.58919322,  6.04779042,  2.71757301,
#         3.69468567,  3.84587442,  5.58184031,  8.10046059,  2.32104457,
#         9.86079294,  2.25398313,  5.78197519,  1.77949771,  9.12443243,
#         5.48646522,  7.24862982,  9.76648603,  5.08493055,  8.18403217,
#         5.11691039,  1.06765756,  4.51224205,  9.97909674, 10.        ]
# cityYpos=[ 0.        ,  0.53022814,  3.54597357,  2.42138272,  1.05124698,
#         7.4271093 ,  4.01568047,  6.43997837,  5.96972804,  1.1774294 ,
#         1.298142  ,  7.60908452,  8.59024184,  7.79836957,  7.80889488,
#         4.37252154,  3.76985716,  1.03103831,  0.90413774,  2.31882736,
#         5.04128013,  5.05152678,  9.18983026,  1.19866566,  3.74219691,
#         7.97055398,  8.92362005,  1.76880202,  1.45866429, 10.        ]


# fig1 = plt.figure(1)
# fig2 = plt.figure(2)
# ax1 = fig1.add_subplot(1, 1, 1)
# ax2 = fig2.add_subplot(1, 1, 1)

for i in range(cityCount):
    for j in range(cityCount):
        Distance[i][j]=math.sqrt((cityXpos[i]-cityXpos[j])**2+\
                                 (cityYpos[i]-cityYpos[j])**2)

InfoDensity=np.ones([cityCount,cityCount])*InfoDensityInitial #初始化信息素浓度

#维护每个蚂蚁选择的路径
#但是路径需要邻接表/两个变量存储
#所以转换成生成城市序列
citySeries=np.zeros([AntCount,cityCount],dtype=int)
#维护是否去过某个城市
visited=np.zeros([AntCount,cityCount],dtype=bool)

startCity=0
endCity=cityCount-1

iterationCount=100#迭代次数
for iteration in range(iterationCount):
    #每轮迭代需要生成所有蚂蚁的路径，然后更新信息素
    antPathLength=np.zeros([AntCount],dtype=np.float32)
    for ant in range(AntCount):
        #每个蚂蚁根据概率选择路径
        #概率权重=路径长度倒数 * 信息素浓度
        
        for pathIndex in range(cityCount):
            #生成路径就是把城市排序一下，所以只要每次选择下一个城市就行
            if (pathIndex==0):
                #选择起点
                visited[ant][pathIndex]=True
                citySeries[ant][pathIndex]=startCity
            elif(pathIndex==cityCount-1):
                #最后一段路径必须连接到终点
                visited[ant][pathIndex]=True
                citySeries[ant][pathIndex]=endCity
            else:
                #除了起点终点，根据概率选择下一个城市

                lastCity=citySeries[ant][pathIndex-1]
                sumProb=0
                allPath=[]#维护一个概率序列
                alpha=1;beta=1

                #注意中间路径不能选择终点城市
                for nextCity in range(cityCount-1):
                    #生成概率序列
                    if visited[ant][nextCity]:
                        #排除已经去过的城市
                        continue
                    prob=pow(1.0/Distance[lastCity][nextCity],alpha) * \
                           pow(InfoDensity[lastCity][nextCity],beta)
                    sumProb=sumProb+prob
                    allPath.append([prob,nextCity])#保存每条路径的概率和终点信息
                
                #生成随机数
                r=random.random()*sumProb
                probCount=0
                for path in allPath:
                    probCount=probCount+path[0]
                    if r<probCount:#随机数落在该概率区间内，说明随机到了这条路径
                        visited[ant][path[1]]=True
                        citySeries[ant][pathIndex]=path[1]
                        break#选好后直接退出，不然后面的路径也满足判定条件

        #单个蚂蚁生成完路径了，统计信息素更新依照的指标（路径长度）
        pathLength=0
        for city in range(cityCount-1):
            pathStart=citySeries[ant][city]
            pathEnd=citySeries[ant][city+1]
            pathLength+=Distance[pathStart][pathEnd]
        antPathLength[ant]=pathLength

    #更新信息素
    #自然蒸发率0~1
    rhoEvaporate=0.5
    InfoDensity=np.multiply(InfoDensity,rhoEvaporate)

    #每条蚂蚁走过的整条路径 根据路径长度的倒数 增加信息素
    rhoTraverse=1#倍率
    for ant in range(AntCount):
        for city in range(cityCount-1):
            pathStart=citySeries[ant][city]
            pathEnd=citySeries[ant][city+1]
            InfoDensity[pathStart][pathEnd]+=rhoTraverse*1.0/antPathLength[ant]
            InfoDensity[pathEnd][pathStart]+=rhoTraverse*1.0/antPathLength[ant]#邻接矩阵正反两条路都要更新

    #输出当前情况下找到的最优解
    minPathIndex=np.argmin(antPathLength)
    minPath = antPathLength[minPathIndex]
    
    # fig1.clf()#清空上一次的图形
    # ax1.clear()
    # ax1.scatter(cityXpos, cityYpos, color="blue")# 绘制城市/点
    #输出具体的路径
    print("iteration:{},length:{:2f}".format(iteration,minPath))
    for city in range(cityCount-1):
        pathStart=citySeries[minPathIndex][city]
        pathEnd=citySeries[minPathIndex][city+1]
        start_x=cityXpos[pathStart]
        start_y=cityYpos[pathStart]
        end_x=cityXpos[pathEnd]
        end_y=cityYpos[pathEnd]
        # ax1.plot([start_x, end_x], [start_y, end_y], color="red", linewidth=2)
        # print("{0}->{1} : {2:2f}".format(pathStart,pathEnd,Distance[pathStart][pathEnd]))

    #输出信息素浓度
    maxInfoDensity=np.max(InfoDensity)
    # ax2.clear()
    # ax2.scatter(cityXpos, cityYpos, color="blue")# 绘制城市/点
    for i in range(cityCount):
        for j in range(cityCount):
            if i<=j :continue #只需要上三角
            start_x=cityXpos[i]
            start_y=cityYpos[i]
            end_x=cityXpos[j]
            end_y=cityYpos[j]
            pathInfoDensity=InfoDensity[i][j]
            pathColor=(1,0,0,pathInfoDensity/maxInfoDensity)
            # ax2.plot([start_x, end_x], [start_y, end_y], color=pathColor, linewidth=2)
    # plt.pause(0.01)

    #清空变量
    citySeries=np.zeros([AntCount,cityCount],dtype=int)
    visited=np.zeros([AntCount,cityCount],dtype=bool)
    
                


# 防止绘图窗口闪退
# while True:
#     try:
#         plt.pause(0.1)
#     except:
#         break


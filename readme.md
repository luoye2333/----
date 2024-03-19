## 1.用蚁群算法解决旅行商问题(寻找n个城市中间的最短遍历路径)
antAlogrithm.py\
antAlogrithm_reconstruct.py(重构：包装成类)

运行结果：video\旅行商

## 2.用蚁群算法解决连续空间的寻路(寻找从起点到终点的最短路径)
continous_alog.py\
continous_alog_multithread.py(多线程求解实现)\
continous_alogrithm_reconstruct_.py(重构：包装成类)\
map_creator.py(生成地图)

运行结果: solutionMap\
把图片合成视频: video\旅行商

其中一种算法是初始轨迹用RRT生成,这部分使用了知乎某个作者的源码\
https://github.com/chenjm1109/robotics-zhihu/tree/main/src/MotionPlanning/motion_planning

## 3.步行机器人机身位置规划和足端落脚点规划
robot_antAlogrithm.py(未完成)\
求解效率很低,约等于弱化版的RL算法,所以没继续做

运行结果：video\足式机器人
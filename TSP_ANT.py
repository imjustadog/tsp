#!/usr/bin/env python
#coding:utf-8
import numpy as np
from Tkinter import *
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import axes3d, Axes3D
import random
import threading
import copy
import time
import sys
import math

# 参数
(ALPHA, BETA, RHO, Q) = (1.0,2.0,0.5,100.0)
# 城市数，蚁群
(dest_num, ant_num) = (33,33)
# 城市坐标

# 迭代次数
iteration = 300

distance_graph = [[0.0 for col in xrange(dest_num)] for raw in xrange(dest_num)]
pheromone_graph = [[1.0 for col in xrange(dest_num)] for raw in xrange(dest_num)]

distance_x = np.random.randint(0, 100, size=dest_num)
distance_y = np.random.randint(0, 100, size=dest_num)
distance_z = np.random.randint(0, 100, size=dest_num)

#----------- 蚂蚁 -----------
class Ant(object):

    # 初始化
    def __init__(self,ID):

        self.ID = ID                 # ID
        self.__clean_data()          # 随机初始化出生点

    # 初始数据
    def __clean_data(self):

        self.path = []               # 当前蚂蚁的路径
        self.total_distance = 0.0    # 当前路径的总距离
        self.move_count = 0          # 移动次数
        self.current_city = -1       # 当前停留的城市
        self.open_table_city = [True for i in xrange(dest_num)] # 探索城市的状态

        city_index = random.randint(0,dest_num-1) # 随机初始出生点
        self.current_city = city_index
        self.path.append(city_index)
        self.open_table_city[city_index] = False
        self.move_count = 1

    # 选择下一个城市
    def __choice_next_city(self):

        next_city = -1
        select_citys_prob = [0.0 for i in xrange(dest_num)]
        total_prob = 0.0

        # 获取去下一个城市的概率
        for i in xrange(dest_num):
            if self.open_table_city[i]:
                try :
                    # 计算概率：与信息素浓度成正比，与距离成反比
                    select_citys_prob[i] = pow(pheromone_graph[self.current_city][i], ALPHA) * pow((1.0/distance_graph[self.current_city][i]), BETA)
                    total_prob += select_citys_prob[i]
                except ZeroDivisionError, e:
                    print 'Ant ID: {ID}, current city: {current}, target city: {target}'.format(ID = self.ID, current = self.current_city, target = i)
                    sys.exit(1)

        # 轮盘选择城市
        if total_prob > 0.0:
            # 产生一个随机概率
            temp_prob = random.uniform(0.0, total_prob)
            for i in xrange(dest_num):
                if self.open_table_city[i]:
                    # 轮次相减
                    temp_prob -= select_citys_prob[i]
                    if temp_prob < 0.0:
                        next_city = i
                        break

        # 未从概率产生，顺序选择一个未访问城市
        if next_city == -1:
            for i in xrange(dest_num):
                if self.open_table_city[i]:
                    next_city = i
                    break

        # 返回下一个城市序号
        return next_city

    # 计算路径总距离
    def __cal_total_distance(self):

        temp_distance = 0.0

        for i in xrange(1, dest_num):
            start, end = self.path[i], self.path[i-1]
            temp_distance += distance_graph[start][end]

        # 回路
        end = self.path[0]
        temp_distance += distance_graph[start][end]
        self.total_distance = temp_distance


    # 移动操作
    def __move(self, next_city):

        self.path.append(next_city)
        self.open_table_city[next_city] = False
        self.total_distance += distance_graph[self.current_city][next_city]
        self.current_city = next_city
        self.move_count += 1

    # 搜索路径
    def search_path(self):

        # 初始化数据
        self.__clean_data()

        # 搜素路径，遍历完所有城市为止
        while self.move_count < dest_num:
            # 移动到下一个城市
            next_city =  self.__choice_next_city()
            self.__move(next_city)

        # 计算路径总长度
        self.__cal_total_distance()

class tsp():
    def __init__(self, root):
        self.root = root
        self.f = plt.figure()
        plt.ion()
        self.canvas = FigureCanvasTkAgg(self.f, master=root)
        self.canvas.show()
        self.canvas.get_tk_widget().grid(row=0, columnspan=5)

    def drawPic(self):
        #清空图像，以使得前后两次绘制的图像不会重叠
        self.f.clf()
        self.a = Axes3D(self.f)

        self.__lock = threading.RLock()  # 线程锁

        self.__lock.acquire()
        self.__running = False
        self.__lock.release()

        self.nodes = []  # 节点坐标
        self.lines = []
        for i in range(len(distance_x)):
            # 在画布上随机初始坐标
            x = distance_x[i]
            y = distance_y[i]
            z = distance_z[i]
            self.nodes.append((x, y, z))

        self.line(range(dest_num))

        # 初始城市之间的距离和信息素
        for i in xrange(dest_num):
            for j in xrange(dest_num):
                pheromone_graph[i][j] = 1.0

        self.ants = [Ant(ID) for ID in xrange(ant_num)]  # 初始蚁群
        self.best_ant = Ant(-1)                          # 初始最优解
        self.best_ant.total_distance = 1 << 31           # 初始最大距离
        self.iter = 1                                    # 初始化迭代次数

        for i in range(dest_num):
            for j in range(dest_num):
                temp_distance = pow((distance_x[i] - distance_x[j]), 2) + pow((distance_y[i] - distance_y[j]), 2) + pow((distance_z[i] - distance_z[j]), 2)
                temp_distance = pow(temp_distance, 0.5)
                distance_graph[i][j] = float(int(temp_distance + 0.5))

        #绘制这些随机点的散点图，颜色随机选取
        self.a.scatter(distance_x, distance_y, distance_z)
        self.a.set_title('Demo: Draw N Random Dot')
        self.canvas.show()

    def stop(self):
        self.__lock.acquire()
        self.__running = False
        self.__lock.release()
        self.root.destroy()
        sys.exit()

    # 将节点按order顺序连线
    def line(self, order):
        if self.lines != []:
            for l in self.lines:
                l.pop(0).remove()
        self.lines = []
        def line2(i1, i2):
            p1, p2 = self.nodes[i1], self.nodes[i2]
            x = [p1[0],p2[0]]
            y = [p1[1],p2[1]]
            z = [p1[2],p2[2]]
            self.lines.append(self.a.plot(x, y, z))
            return i2

        # order[-1]为初始值
        reduce(line2, order, order[-1])


    def search_path(self, evt = None):

        # 开启线程
        self.__lock.acquire()
        self.__running = True
        self.__lock.release()

        while self.__running:
            # 遍历每一只蚂蚁
            for ant in self.ants:
                # 搜索一条路径
                ant.search_path()
                # 与当前最优蚂蚁比较
                if ant.total_distance < self.best_ant.total_distance:
                    # 更新最优解
                    self.best_ant = copy.deepcopy(ant)
            # 更新信息素
            self.__update_pheromone_gragh()
            print u"迭代次数：",self.iter,u"最佳路径总距离：",int(self.best_ant.total_distance)
            # 连线
            self.line(self.best_ant.path)
            # 更新画布
            self.canvas.show()
            self.iter += 1
            if self.iter > iteration:
                break


    # 更新信息素
    def __update_pheromone_gragh(self):

        # 获取每只蚂蚁在其路径上留下的信息素
        temp_pheromone = [[0.0 for col in xrange(dest_num)] for raw in xrange(dest_num)]
        for ant in self.ants:
            for i in xrange(1,dest_num):
                start, end = ant.path[i-1], ant.path[i]
                # 在路径上的每两个相邻城市间留下信息素，与路径总距离反比
                temp_pheromone[start][end] += Q / ant.total_distance
                temp_pheromone[end][start] = temp_pheromone[start][end]

        # 更新所有城市之间的信息素，旧信息素衰减加上新迭代信息素
        for i in xrange(dest_num):
            for j in xrange(dest_num):
                pheromone_graph[i][j] = pheromone_graph[i][j] * RHO + temp_pheromone[i][j]

if __name__ == '__main__':

    matplotlib.use('TkAgg')
    root = Tk()
    tsp_class = tsp(root)

    #放置标签、文本框和按钮等部件，并设置文本框的默认值和按钮的事件函数
    Label(root,text='请输入样本数量：').grid(row=1,column=0)
    inputEntry=Entry(root)
    inputEntry.grid(row=1,column=1)
    inputEntry.insert(0,'50')
    Button(root,text='画图',command=tsp_class.drawPic).grid(row=1,column=2,columnspan=5)
    Button(root, text='开始', command=tsp_class.search_path).grid(row=1, column=3, columnspan=5)
    root.protocol("WM_DELETE_WINDOW", tsp_class.stop)
    #启动事件循环
    root.mainloop()
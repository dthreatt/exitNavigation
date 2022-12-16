#!/usr/bin/python3.8
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid
import std_msgs
from std_msgs.msg import Int16
import message_filters


def plotSeg(OccupancyGridMsg):
    shaped_map = np.reshape(np.asarray(OccupancyGridMsg.data),(128,128))+128
    #stamp = msg.header.stamp
    #time = stamp.secs + stamp.nsecs * 1e-9
    unique,counts = np.unique(shaped_map,return_counts=True)
    print(np.asarray((unique,counts)).T)
    plt.figure(1)
    plt.imshow(shaped_map,cmap="gray")
    plt.colorbar()
    plt.draw()
    plt.pause(0.0000000000000000000001)

def plotHeat(OccupancyGridMsg):#,goalx,goaly):
    shaped_map = np.reshape(np.asarray(OccupancyGridMsg.data),(128,128))/127.0
    #stamp = msg.header.stamp
    #time = stamp.secs + stamp.nsecs * 1e-9
    plt.figure(2)
    plt.imshow(shaped_map)
    plt.colorbar()
    plt.draw()
    plt.pause(0.0000000000000000000001)
    


def plotter():
    
    rospy.init_node('plotter', anonymous=True)
    #would like to add ability to wait for topics
    rospy.Subscriber('segmented_map',OccupancyGrid,plotSeg)
    rospy.Subscriber('heat_map',OccupancyGrid,plotHeat)
    #heat = message_filters.Subscriber('heat_map', OccupancyGrid)
    #goalx = message_filters.Subscriber('goal_point_x',Int16)
    #goaly = message_filters.Subscriber('goal_point_y',Int16)
    #ts = message_filters.TimeSynchronizer([heat, goalx, goaly], 10)
    #ts.registerCallback(plotHeat)
    plt.ion()
    plt.show()
    rospy.spin()

if __name__ == '__main__':
    try:
        plotter()
    except rospy.ROSInterruptException:
        pass
#!/usr/bin/python3.8

import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal, MoveBaseFeedback, MoveBaseResult
from exit_navigation.srv import *
from nav_msgs.msg import OccupancyGrid
import std_msgs
from std_msgs.msg import Int16

def termination():
    rospy.init_node('termination_check', anonymous=True)

    #set up move base client
    client = actionlib.SimpleActionClient('/move_base', MoveBaseAction)#might need (self.name+) which was in rrt
    client.wait_for_server()
    goalpoint = MoveBaseGoal()
    probability = 0
    
    #set up publisher for segmented map and heatmap
    segmented_pub = rospy.Publisher('segmented_map',OccupancyGrid,queue_size=2,latch = True)
    heat_pub = rospy.Publisher('heat_map',OccupancyGrid,queue_size=2, latch = True)
    goalx_pub = rospy.Publisher('goal_point_x',Int16,queue_size=10,latch=True)
    goaly_pub = rospy.Publisher('goal_point_y',Int16,queue_size=10,latch=True)

    #set up goal point service and request first goal point
    rospy.wait_for_service('exit_prediction')
    try:
        exit_prediction = rospy.ServiceProxy('exit_prediction', ExitRequest)
        exit_response = exit_prediction() #currently not supplying the service with info
        goalpoint.target_pose.pose.position.x = exit_response.x
        goalpoint.target_pose.pose.position.y = exit_response.y
        probability = exit_response.probability
        segmented_map = exit_response.segmented
        heat_map = exit_response.heat
        goalGridx = exit_response.goalGridx
        goalGridy = exit_response.goalGridy

        goalpoint.target_pose.header.stamp = rospy.Time.now()
        goalpoint.target_pose.header.frame_id = 'map' #might want to change the frame.
        goalpoint.target_pose.pose.orientation.w = 1.0
        print(probability,"Exit probability")
        print(goalpoint.target_pose.pose.position.x,",",goalpoint.target_pose.pose.position.y,"x,y Goal Point")
    except rospy.ServiceException as e:
        print ("Service call failed: %s"%e)
    
    #publish segmented map and heat map for visualization
    segmented_pub.publish(segmented_map)
    heat_pub.publish(heat_map)
    goalx_pub.publish(goalGridx)
    goaly_pub.publish(goalGridy)

    #send first goal and check if satisfactory for termination
    client.send_goal(goalpoint)
    threshold = 0.5
    if(probability > threshold):
        rospy.signal_shutdown("Exit threshold reached. Terminating")

    #repeat until satisfactory point is found    
    while not rospy.is_shutdown():
        client.wait_for_result()
        #if(text == "Goal reached." or status == 3) #necessary? Wait for result should wait until it is finished (ie stopped moving)
        exit_response = exit_prediction() #currently not supplying the service with info
        goalpoint.target_pose.pose.position.x = exit_response.x
        goalpoint.target_pose.pose.position.y = exit_response.y
        probability = exit_response.probability
        segmented_map = exit_response.segmented
        heat_map = exit_response.heat
        goalGridx = exit_response.goalGridx
        goalGridy = exit_response.goalGridy

        #publish segmented map and heat map for visualization
        segmented_pub.publish(segmented_map)
        heat_pub.publish(heat_map)
        goalx_pub.publish(goalGridx)
        goaly_pub.publish(goalGridy)

        print(probability,"Exit probability")
        print(goalpoint.target_pose.pose.position.x,",",goalpoint.target_pose.pose.position.y,"x,y Goal Point")
        goalpoint.target_pose.header.stamp = rospy.Time.now()
        client.send_goal(goalpoint)
        if(probability > threshold):
            rospy.signal_shutdown("Exit threshold reached. Terminating")


if __name__ == '__main__':
    try:
        termination()
    except rospy.ROSInterruptException:
        pass
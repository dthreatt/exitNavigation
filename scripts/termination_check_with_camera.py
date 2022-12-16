#!/usr/bin/python3

import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal, MoveBaseFeedback, MoveBaseResult
from exit_navigation.srv import *
from nav_msgs.msg import OccupancyGrid, Odometry
import std_msgs
from std_msgs.msg import Int16
from apriltag_ros.msg import AprilTagDetectionArray
from geometry_msgs.msg import Twist
from tf.transformations import euler_from_quaternion
import math
import numpy as np
#from detec.py import PID, Jackal

def setTagDetections(msg):
    global tag_detections
    tag_detections = msg

def getTagDetection():
    temp = tag_detections    
    return temp

def setOdometry(msg):
    global Odom 
    Odom = msg.pose

def getOdometry():
    tempO = Odom
    return tempO

def findTag(timerMSG):
    #create velocity command message and set up publisher
    velocityCMD = Twist()
    velocityCMD.angular.z = 1 #radians
    velocity_pub = rospy.Publisher('cmd_vel',Twist,queue_size=1)
    velocity_pub.publish(velocityCMD)
    detectedTag = getTagDetection()
    if detectedTag.detections:
        TagDistx = detectedTag.detections[0].pose.pose.pose.position.x #distance left or right of the camera
        TagDisty = detectedTag.detections[0].pose.pose.pose.position.y
        TagDistz = detectedTag.detections[0].pose.pose.pose.position.z #distance in front of the camera
        detected_Distance = np.linalg.norm(np.asarray([TagDistx,TagDisty,TagDistz]))
        if(detected_Distance < 2):
            RobotO = getOdometry()
            explicit_quat = [RobotO.pose.orientation.x,RobotO.pose.orientation.y,RobotO.pose.orientation.z,RobotO.pose.orientation.w]
            (roll,pitch,yaw) = euler_from_quaternion(explicit_quat)
            pose = (RobotO.pose.position.x,RobotO.pose.position.y,yaw)
            clientExit = actionlib.SimpleActionClient('/move_base', MoveBaseAction)#might need (self.name+) which was in rrt
            clientExit.wait_for_server()
            goalpoint = MoveBaseGoal()
            goalpoint.target_pose.header.stamp = rospy.Time.now()
            goalpoint.target_pose.header.frame_id = 'map' #might want to change the frame.
            goalpoint.target_pose.pose.orientation = RobotO.pose.orientation
            goalpoint.target_pose.pose.position.x = (math.cos(yaw)*TagDistz - math.sin(yaw)*TagDistx) + pose[0]
            goalpoint.target_pose.pose.position.y = (math.sin(yaw)*TagDistz + math.cos(yaw)*TagDistx) + pose[1]
            clientExit.send_goal(goalpoint)
            print("Exit found")
            rospy.signal_shutdown("Exit found. Terminating") #terminate

def termination():
    rospy.init_node('termination_check', anonymous=True)
    
    #subscribe to apriltag_detections
    rospy.Subscriber('tag_detections', AprilTagDetectionArray, setTagDetections)

    #subscribe to the odometry
    rospy.Subscriber('odometry/filtered', Odometry, setOdometry)
    #might need a pause to fill data buffe but it might be covered by the move base wait for server

    #set up move base client
    client = actionlib.SimpleActionClient('/move_base', MoveBaseAction)#might need (self.name+) which was in rrt
    client.wait_for_server()
    goalpoint = MoveBaseGoal()
    probability = 0

    #initial check to see if the program was started right next to the door
    threshold = 0.0
    if(threshold == 0):
        accumulate_flag = True
        Caller = rospy.Timer(rospy.Duration(0.2),findTag)
        rospy.sleep(6.5)#sleep?? want to let the timer run and the bot spin, but stop both after 7 seconds
        Caller.shutdown()
    else:
        accumulate_flag = False
    
    #set up publisher for segmented map and heatmap
    segmented_pub = rospy.Publisher('segmented_map',OccupancyGrid,queue_size=2,latch = True)
    heat_pub = rospy.Publisher('heat_map',OccupancyGrid,queue_size=2, latch = True)
    goalx_pub = rospy.Publisher('goal_point_x',Int16,queue_size=10,latch=True)
    goaly_pub = rospy.Publisher('goal_point_y',Int16,queue_size=10,latch=True)

    #set up goal point service and request first goal point
    rospy.wait_for_service('exit_prediction')
    try:
        exit_prediction = rospy.ServiceProxy('exit_prediction', ExitRequest)
        exit_response = exit_prediction(accumulate_flag)
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
        rospy.signal_shutdown("Terminating") #terminate
    
    #publish segmented map and heat map for visualization
    segmented_pub.publish(segmented_map)
    heat_pub.publish(heat_map)
    goalx_pub.publish(goalGridx)
    goaly_pub.publish(goalGridy)

    #send first goal and check if satisfactory for termination
    client.send_goal(goalpoint)
    client.wait_for_result()
    if(probability > threshold):
        accumulate_flag = True
        Caller = rospy.Timer(rospy.Duration(0.2),findTag)
        rospy.sleep(7)#sleep?? want to let the timer run and the bot spin, but stop both after 7 seconds
        Caller.shutdown()
    else:
        accumulate_flag = False

    #repeat until satisfactory point is found    
    while not rospy.is_shutdown():
        #if(text == "Goal reached." or status == 3) #necessary? Wait for result should wait until it is finished (ie stopped moving)
        exit_response = exit_prediction(accumulate_flag)
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

        client.wait_for_result()        
        if(probability > threshold):
            accumulate_flag = True
            Caller = rospy.Timer(rospy.Duration(0.2),findTag)
            rospy.sleep(7)#sleep?? want to let the timer run and the bot spin, but stop both after 7 seconds
            Caller.shutdown()
        else:
            accumulate_flag = False    


if __name__ == '__main__':
    try:
        termination()
    except rospy.ROSInterruptException:
        pass
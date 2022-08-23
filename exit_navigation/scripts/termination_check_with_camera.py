#!/usr/bin/python3.8

import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal, MoveBaseFeedback, MoveBaseResult
from exit_navigation.srv import *
from nav_msgs.msg import OccupancyGrid
import std_msgs
from std_msgs.msg import Int16
from apriltag_ros.msg import AprilTagDetectionArray
from geometry_msgs.msg import Twist

def setTagDetections(msg):
    global tag_detections
    tag_detections = msg

def getTagDetection():
    temp = tag_detections    
    return temp

def findTag(timerMSG):
    detectedTag = getTagDetection()
    if(detectedTag.detections != []):
        detected_Distance = np.linalg.norm(np.asarray([detectedTag.detections.pose.pose.pose.position.x,detectedTag.detections.pose.pose.pose.position.y,detectedTag.detections.pose.pose.pose.position.z]))
        if(detected_Distance < 2):
            #Use velocity commands to drive it to the tag
            rospy.signal_shutdown("Exit found. Terminating") #terminate

def termination():
    rospy.init_node('termination_check', anonymous=True)
    
    #subsribe to apriltag_detections
    rospy.Subscriber('tag_detections', AprilTagDetectionArray, setTagDetections)
    #might need a pause, might be covered by the move base wait for server

    #create velocity command message and set up publisher
    velocityCMD = Twist()
    velocityCMD.angular = 1 #radians
    velocity_pub = rospy.Publisher('cnd_vel',Twist,queue_size=1)

    #set up move base client
    client = actionlib.SimpleActionClient('/move_base', MoveBaseAction)#might need (self.name+) which was in rrt
    client.wait_for_server()
    goalpoint = MoveBaseGoal()
    probability = 0

    #initial check to see if the program was started right next to the door
    threshold = 0.0
    if(threshold == 0):
        accumulate_flag = True
        velocity_pub.publish(velocityCMD)
        rospy.Timer(rospy.Duration(0.5),findTag)
        rospy.sleep(7)#sleep?? want to let the timer run and the bot spin, but stop both after 7 seconds
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
        detectedTag = getTagDetection()
        if(detectedTag.detections != []):
            detected_Distance = np.linalg.norm(np.asarray([detectedTag.detections.pose.pose.pose.position.x,detectedTag.detections.pose.pose.pose.position.y,detectedTag.detections.pose.pose.pose.position.z]))
            if(detected_Distance < 2):
                #get current pose
                #move base goal = current position + detected relative pose
                #client.send_goal(goalpoint)
                rospy.signal_shutdown("Exit found. Terminating") #terminate
        else:
            accumulate_flag = True
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
            detectedTag = getTagDetection()
            if(detectedTag.detections != []):
                detected_Distance = np.linalg.norm(np.asarray([detectedTag.detections.pose.pose.pose.position.x,detectedTag.detections.pose.pose.pose.position.y,detectedTag.detections.pose.pose.pose.position.z]))
                if(detected_Distance < 2):
                    #get current pose
                    #move base goal = current position + detected relative pose
                    #client.send_goal(goalpoint)
                    rospy.signal_shutdown("Exit found. Terminating") #terminate
            else:
                accumulate_flag = True
        else:
            accumulate_flag = False    


if __name__ == '__main__':
    try:
        termination()
    except rospy.ROSInterruptException:
        pass
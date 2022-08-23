#!/usr/bin/python3.8

from exit_navigation.srv import ExitRequest,ExitRequestResponse
import rospy
import tf
import tensorflow #might not be necessary
import numpy as np
from nav_msgs.msg import OccupancyGrid
from nav_msgs.srv import GetMap
from os import path
import os
import sys
sys.path.append('home/robot/py3_ws/src/exit_navigation/scripts')
from UNet_new_conditional import Conditional_New_VAE
from A_Star_Planner import AStarPlanner
from scipy import ndimage
from Functions import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

map_color= {'uncertain':127, 'free':0, 'obstacle':255}
exit_map_color= {'uncertain':127, 'non-exit':0, 'exit':255}
#round_cnt = 0

def within_bound(p,shape,r=0):
    """ check if point p [y;x] or [y;x;theta] with radius r is inside world of shape (h,w)
    return bool if p is single point | return bool matrix (vector) if p: [y;x] where y & x are matrix (vector) """
    return (p[0] >= r) & (p[0] < shape[0]-r) & (p[1] >= r) & (p[1] < shape[1]-r)

def accumulate_non_exits(nonExitMap,local_world,pose,exit_threshold=12):
    window_halfsize = local_world.shape[1]//2
    print(local_world.shape,"local world shape")
    print(window_halfsize,"window halfsize")
    world = np.ones((local_world.shape[1]*2,local_world.shape[2]*2))*map_color['uncertain']
    world[window_halfsize:window_halfsize+local_world.shape[1],
          window_halfsize:window_halfsize+local_world.shape[2]] = local_world

    pose = [window_halfsize + pose[0],window_halfsize + pose[1],0]

    laser_fov = 60
    laser_resol = 0.125
    angles_vect = np.arange(-laser_fov*0.5, laser_fov*0.5,step=laser_resol)
    angles_vect = angles_vect.reshape(angles_vect.shape[0], 1) # generate angles vector from -laser_angle/2 to laser_angle
    """ find the coord matrix that the laser cover """
    angles= pose[2] + angles_vect
    radius_vect= np.arange(exit_threshold+1)
    radius_vect= radius_vect.reshape(1, radius_vect.shape[0]) # generate radius vector of [0,1,2,...,laser_range]
    y_rangeCoordMat= pose[0] - np.matmul(np.sin(angles), radius_vect)
    x_rangeCoordMat= pose[1] + np.matmul(np.cos(angles), radius_vect)

    # Round y and x coord into int
    y_rangeCoordMat = (np.round(y_rangeCoordMat)).astype(int)
    x_rangeCoordMat = (np.round(x_rangeCoordMat)).astype(int)

    """ Check for index of y_mat and x_mat that are within the world """
    inBound_ind= within_bound(np.array([y_rangeCoordMat, x_rangeCoordMat]), world.shape)

    """ delete coordinate that are not within the bound """
    outside_ind = np.argmax(~inBound_ind, axis=1)
    ok_ind = np.where(outside_ind == 0)[0]
    need_amend_ind = np.where(outside_ind != 0)[0]
    outside_ind = np.delete(outside_ind, ok_ind)
    
    inside_ind = np.copy(outside_ind)
    inside_ind[inside_ind != 0] -= 1
    bound_ele_x = x_rangeCoordMat[need_amend_ind, inside_ind]
    bound_ele_y = y_rangeCoordMat[need_amend_ind, inside_ind]

    count = 0
    for i in need_amend_ind:
        x_rangeCoordMat[i, ~inBound_ind[i,:]] = bound_ele_x[count]
        y_rangeCoordMat[i, ~inBound_ind[i,:]] = bound_ele_y[count]
        count += 1
    """ find obstacle along the laser range """
    obstacle_ind = np.argmax(world[y_rangeCoordMat, x_rangeCoordMat] == map_color['obstacle'], axis=1)
    obstacle_ind[obstacle_ind == 0] = x_rangeCoordMat.shape[1]
    """ generate a matrix of [[1,2,3,...],[1,2,3...],[1,2,3,...],...] for comparing with the obstacle coord """
    bx = np.arange(x_rangeCoordMat.shape[1]).reshape(1, x_rangeCoordMat.shape[1])
    by = np.ones((x_rangeCoordMat.shape[0], 1))
    b = np.matmul(by, bx)

    """ get the coord that the robot can percieve (ignore pixel beyond obstacle) """
    b = b <= obstacle_ind.reshape(obstacle_ind.shape[0], 1)
    y_coord = y_rangeCoordMat[b]
    x_coord = x_rangeCoordMat[b]
    
    filled_tmp = np.ones(world.shape)*exit_map_color['uncertain']
    filled_tmp[y_coord,x_coord] = world[y_coord,x_coord]
    filled_tmp[filled_tmp==map_color['obstacle']] = exit_map_color['non-exit']
    filled_tmp[filled_tmp==map_color['free']] = exit_map_color['non-exit']
    nonExitMap[y_coord,x_coord] = filled_tmp[y_coord,x_coord]
    return nonExitMap

def reset_cameraMap():
    return np.ones((352,352))*exit_map_color['uncertain']

def load_model(vae_model,model_name,latent_dim,exit_Flag):
    unet = vae_model(latent_dim=latent_dim,exit_Flag=exit_Flag)
    env_mode = 'RPLAN'
    # model_name = 'multi_step_dataset/rplan_raw_multiTask_exit_u-net_latent16_0208'
    filepath = path.join(path.dirname(__file__), "..","learning","model",model_name,'base_model_weights')
    unet.load_weights(filepath)
    return unet

def compute_centroid(mask):
    sum_h = 0
    sum_w = 0 
    count = 0
    shape_array = mask.shape
    for h in range(shape_array[0]):  
        for w in range(shape_array[1]):
            if mask[h, w] != 0:
                sum_h += h
                sum_w += w
                count += 1
    return np.array([sum_h//count, sum_w//count])  

def exit_loc_to_map(map_input,exit_loc,filter_size = 7):
    filter_half_size = filter_size//2
    exit_loc_map = np.zeros(map_input.shape)
    exit_h_min = max(0,exit_loc[1]-filter_half_size)
    exit_h_max = min(map_input.shape[0],exit_loc[1]+filter_half_size)
    exit_w_min = max(0,exit_loc[0]-filter_half_size)
    exit_w_max = min(map_input.shape[1],exit_loc[0]+filter_half_size)
    exit_loc_map[exit_h_min:exit_h_max+1,exit_w_min:exit_w_max+1] = 1
    return exit_loc_map

def find_exit_from_heatmap(exit_build,camera_map,robot_pose,filter_size=9,mode='nearest'):   
    # print('filter_size: ',filter_size,camera_map.shape,exit_build.shape)
    # print(np.unique(camera_map==exit_map_color['non-exit']),(camera_map==exit_map_color['non-exit']).shape)
    exit_build = exit_build.numpy()
    exit_build[camera_map==exit_map_color['non-exit']] = 0.0
    filter_half_size = int(np.floor(filter_size/2))
    kernel = np.ones((filter_size,filter_size))
    result = ndimage.convolve(exit_build, kernel, mode=mode, cval=0.0)
    max_idx = np.unravel_index(np.argmax(result, axis=None), result.shape)
    exit_loc = np.array([max_idx[0],max_idx[1]])
#     print(exit_build[exit_loc[0]-filter_half_size:exit_loc[0]+filter_half_size+1,exit_loc[1]-filter_half_size:exit_loc[1]+filter_half_size+1].shape)
    min_h = exit_loc[0]-filter_half_size if (exit_loc[0]-filter_half_size>=0) else 0
    max_h = exit_loc[0]+filter_half_size+1 if (exit_loc[0]+filter_half_size+1<=128) else 128
    min_w = exit_loc[1]-filter_half_size if (exit_loc[1]-filter_half_size>=0) else 0
    max_w = exit_loc[1]+filter_half_size+1 if (exit_loc[1]+filter_half_size+1<=128) else 128
    prob = exit_build[exit_loc[0],exit_loc[1]]
    region_max_prob = np.max(exit_build[min_h:max_h,min_w:max_w])
    region_avr_prob = result[exit_loc[0],exit_loc[1]]/(filter_size**2)
    return np.array([exit_loc[1],exit_loc[0]]),prob,region_max_prob,region_avr_prob

def find_nearest_reachable_point_from_exit(map_input,exit_loc,camera_map,filter_size=9):
    filter_half_size = filter_size//2
    map_input = map_input[0,:,:,0]*map_color['free']+map_input[0,:,:,1]*map_color['uncertain']+map_input[0,:,:,2]*map_color['obstacle']
    free_space_map = (map_input==map_color['free'])
    unknown_space_map = (map_input==map_color['uncertain'])
    exit_loc_map = exit_loc_to_map(map_input,exit_loc,filter_size=filter_size)
    exit_loc_map = np.logical_and(exit_loc_map,(camera_map[0]!=exit_map_color['non-exit']))
    # exit_loc_map = np.logical_and(exit_loc_map,(map_input!=map_color['obstacle']))
    
    freeSpace_exit_map = np.logical_and(free_space_map,exit_loc_map)
    unknownSpace_exit_map = np.logical_and(unknown_space_map,exit_loc_map)
    # print(np.where(freeSpace_exit_map)[0].shape,np.where(unknownSpace_exit_map)[0].shape,np.where(obstacleSpace_exit_map)[0].shape,np.where(exit_loc_map)[0].shape)
    if freeSpace_exit_map.any():
        location = compute_centroid(freeSpace_exit_map)
        return (location[1],location[0])
    elif unknownSpace_exit_map.any():
        location = compute_centroid(unknownSpace_exit_map)
        return (location[1],location[0])
    # print(exit_loc_map[0][64,64],camera_map[0][64,64])
    # if np.logical_and(free_space_map,exit_loc_map).any():
    #     freeSpace = np.where(free_space_map)
    #     freeSpace = np.concatenate((freeSpace[0][np.newaxis,...],freeSpace[1][np.newaxis,...]),axis=0)
    #     dist_to_exit = np.linalg.norm(freeSpace-np.array([[exit_loc[1]],[exit_loc[0]]]),axis=0)
    #     min_dist_idx = np.argmin(dist_to_exit)
    #     return (freeSpace[1,min_dist_idx],freeSpace[0,min_dist_idx])
    else:
        unknownSpace = np.where(unknown_space_map)
        unknownSpace = np.concatenate((unknownSpace[0][np.newaxis,...],unknownSpace[1][np.newaxis,...]),axis=0)
        dist_to_exit = np.linalg.norm(unknownSpace-np.array([[exit_loc[1]],[exit_loc[0]]]),axis=0)
        min_dist_idx = np.argmin(dist_to_exit)
        return (unknownSpace[1,min_dist_idx],unknownSpace[0,min_dist_idx])

def plan_path_to_exit(local_map_input,start_point=[0,0],goal_point=[0,0],grid_size=1,robot_radius=1,show_animation=False,plot_dir=[],fig_idx=[],exit_round_cnt=0):
    #fig = plt.figure(fig_idx)
    #axs = []
    #axs.append(fig.add_subplot(1,1,1))
    #axs[0].cla()
    #ax=axs[0]
    # start and goal position
    #ax.invert_yaxis()
    #ax.axis('off')
    #sx = start_point[0] # [m]
    #sy = start_point[1]  # [m]
    #gx = goal_point[0]  # [m]
    #gy = goal_point[1]  # [m]

    mapFlag = True
    obstacle_map = (local_map_input==map_color["obstacle"]) #| (local_map_input==map_color["uncertain"])
    if mapFlag:
        if show_animation:  # pragma: no cover
            plt.imshow(obstacle_map!=1,cmap='gray')
            plt.plot(sx, sy, "og")
            plt.plot(gx, gy, "xr")
            plt.grid(True)
            plt.axis("equal")
        a_star = AStarPlanner(grid_size, robot_radius,show_animation,{'mapFlag':mapFlag,'obstacle_map':obstacle_map.transpose()})
    else:
        oy,ox = np.where(obstacle_map)
        if show_animation:  # pragma: no cover
            ax.plot(ox, oy, ".k", markersize=1)
            ax.plot(sx, sy, "og")
            ax.plot(gx, gy, "xb")
            ax.grid(True)
            ax.axis("equal")
            ax.axis('off')
        a_star = AStarPlanner(grid_size, robot_radius,show_animation,{'mapFlag':mapFlag,'ox':ox,'oy':oy})
    rx, ry, open_set_Flag = a_star.planning(start_point[0],start_point[1],goal_point[0],goal_point[1])
    if show_animation:  # pragma: no cover
        ax.plot(rx, ry, "-r")
        plt.savefig(plot_dir + 'a_star_step_' +str(exit_round_cnt)+ '.png')
        # ax.pause(0.001)
        # plt.show()
    return rx,ry, open_set_Flag

def find_nearest_freeSpace_point_from_exit(map_input,exit_loc,camera_map,filter_size=9):
    filter_half_size = filter_size//2
    map_input = map_input[0,:,:,0]*map_color['free']+map_input[0,:,:,1]*map_color['uncertain']+map_input[0,:,:,2]*map_color['obstacle']
    # exit_loc_map = np.logical_and(exit_loc_map,(map_input!=map_color['obstacle']))
    
    freeSpace_exit_map = np.logical_and((map_input==map_color['free']),(camera_map[0]!=exit_map_color['non-exit']))
    # print(np.where(freeSpace_exit_map)[0].shape,np.where(unknownSpace_exit_map)[0].shape,np.where(obstacleSpace_exit_map)[0].shape,np.where(exit_loc_map)[0].shape)

    freeSpace = np.where(freeSpace_exit_map)
    # print('check:',freeSpace)
    freeSpace = np.concatenate((freeSpace[0][np.newaxis,...],freeSpace[1][np.newaxis,...]),axis=0)
    dist_to_exit = np.linalg.norm(freeSpace-np.array([[exit_loc[1]],[exit_loc[0]]]),axis=0)
    min_dist_idx = np.argmin(dist_to_exit)
    # print('check:',np.unique(freeSpace[0]),freeSpace[:,min_dist_idx],map_input[freeSpace[0,min_dist_idx],freeSpace[1,min_dist_idx]])
    return (freeSpace[1,min_dist_idx],freeSpace[0,min_dist_idx])

def exit_driven_path_plan(unet,laser_input_global,segmented_map,laser_input,camera_input,camera_channel,centroid_size=7,grid_size=1,robot_radius=0.1,robot_pose=[0,0],show_animation=False,plot_dir=[],fig_idx=[],exit_round_cnt=0):
    # print('camera_input:',camera_input.shape)
    Astar_rplan_Flag = False
    if camera_channel==3:
        x_input = np.zeros((laser_input.shape[0],laser_input.shape[1],laser_input.shape[2],6))
        x_input[laser_input==map_color["free"],0] = 1
        x_input[laser_input==map_color["uncertain"],1] = 1
        x_input[laser_input==map_color["obstacle"],2] = 1
        x_input[camera_input==exit_map_color["non-exit"],3] = 1
        x_input[camera_input==exit_map_color["uncertain"],4] = 1
        x_input[camera_input==exit_map_color["exit"],5] = 1
    else:
        x_input = np.zeros((laser_input.shape[0],laser_input.shape[1],laser_input.shape[2],4))
        x_input[laser_input==map_color["free"],0] = 1
        x_input[laser_input==map_color["uncertain"],1] = 1
        x_input[laser_input==map_color["obstacle"],2] = 1
        x_input[camera_input!=exit_map_color["non-exit"],3] = 1
    # x_laser = tf.one_hot(laser.astype(np.float32), 3, axis=-1,dtype=np.float32).numpy()
    x_input = x_input.astype(np.float32)#labels = , (semantic_output[np.newaxis,:,:,np.newaxis].astype(np.float32),
                                                    #centroid_output[np.newaxis,:,:,np.newaxis].astype(np.float32),
                                                   # exit_output[np.newaxis,:,:,np.newaxis].astype(np.float32))
    #print(x_input.shape)
    z_mean, z_log_var, z,semantics_build,centroid_build,exit_build = unet.encoder_decoder(x_input,method=1)
    #print(camera_input.shape)
    exit_loc,prob,region_max_prob,region_avr_prob = find_exit_from_heatmap(exit_build[0,:,:,0],camera_input[0],robot_pose,filter_size=centroid_size)# from exit_build map
    exit_loc_nearest = find_nearest_reachable_point_from_exit(x_input,exit_loc,camera_input,filter_size=centroid_size)# closest reachable point to exit_loc
    # print((robot_pose[1],robot_pose[0]),(robot_pose[1]+exit_loc_nearest[0]-64, robot_pose[0]+exit_loc_nearest[1]-64))
    rx, ry, Astar_openSet_Flag = plan_path_to_exit(local_map_input=laser_input_global[64:64+128,64:64+128],start_point=(robot_pose[1],robot_pose[0]),\
        goal_point=(robot_pose[1]+exit_loc_nearest[0]-64, robot_pose[0]+exit_loc_nearest[1]-64),grid_size=grid_size,robot_radius=robot_radius,show_animation=show_animation,plot_dir=plot_dir,fig_idx=fig_idx,exit_round_cnt=exit_round_cnt)
    
    if not Astar_openSet_Flag:
        exit_loc_nearest = find_nearest_freeSpace_point_from_exit(x_input,exit_loc,camera_input,filter_size=centroid_size)# closest reachable point to exit_loc
        #print(map_input[0][exit_loc_nearest[1],exit_loc_nearest[0]]) Drew: Can't figure out where this map input was defined in your code, and since it is just printing, I commented it
        rx, ry, Astar_openSet_Flag = plan_path_to_exit(local_map_input=laser_input_global[64:64+128,64:64+128],start_point=(robot_pose[1],robot_pose[0]),\
        goal_point=(robot_pose[1]+exit_loc_nearest[0]-64, robot_pose[0]+exit_loc_nearest[1]-64),grid_size=grid_size,robot_radius=robot_radius,show_animation=show_animation,plot_dir=plot_dir,fig_idx=fig_idx,exit_round_cnt=exit_round_cnt)
    
    Astar_traj = np.array([rx,ry])
    Astar_traj = Astar_traj[:,::-1]
    Astar_traj_remapped = Astar_traj - (np.array([[robot_pose[1]],[robot_pose[0]]])-np.array([[64],[64]])) # from global to local
    
    Astar_traj_tmp = Astar_traj
    Astar_traj = Astar_traj_remapped
    Astar_traj_remapped = Astar_traj_tmp
    for i in range(Astar_traj.shape[1]):
        if laser_input[0][Astar_traj[1,i],Astar_traj[0,i]] == map_color['uncertain']:
            Astar_traj = Astar_traj[:,0:i]
            Astar_rplan_Flag = True
            break
    Goalpoint = Astar_traj[:,-1]
    # print(Astar_traj[1,-1],Astar_traj[0,-1])
    point_loc = np.array([Astar_traj[1,-1],Astar_traj[0,-1]])
#     print(exit_build[exit_loc[0]-filter_half_size:exit_loc[0]+filter_half_size+1,exit_loc[1]-filter_half_size:exit_loc[1]+filter_half_size+1].shape)
    filter_half_size = int(np.floor(centroid_size/2))
    min_h = point_loc[0]-filter_half_size if (point_loc[0]-filter_half_size>=0) else 0
    max_h = point_loc[0]+filter_half_size+1 if (point_loc[0]+filter_half_size+1<=128) else 128
    min_w = point_loc[1]-filter_half_size if (point_loc[1]-filter_half_size>=0) else 0
    max_w = point_loc[1]+filter_half_size+1 if (point_loc[1]+filter_half_size+1<=128) else 128
    prob = exit_build[0,point_loc[0],point_loc[1],0].numpy()
    region_max_prob = np.max(exit_build[0,min_h:max_h,min_w:max_w,0])
    exit_build_tmp=exit_build[0,min_h:max_h,min_w:max_w,0]
    region_avr_prob = np.multiply(exit_build_tmp,np.ones(exit_build_tmp.shape))
    region_avr_prob = np.sum(region_avr_prob)/(exit_build_tmp[0]*exit_build_tmp[1])
    return laser_input,x_input,semantics_build,centroid_build,exit_build,exit_loc,prob,region_max_prob,region_avr_prob,exit_loc_nearest,Astar_traj,Astar_traj_remapped,Goalpoint,Astar_rplan_Flag, Astar_openSet_Flag

cameraMap = reset_cameraMap()

def handle_exit_request(req):
    #request map
    #set up map request service
    rospy.wait_for_service('dynamic_map')
    try:
        map_request = rospy.ServiceProxy('dynamic_map', GetMap) #service type 2nd arg? nav_msgs/GetMap? can check with rosservice cmd line tool
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)
    mapResponse = map_request() #guess it doesn't need parameter? 
    shaped_map = np.reshape(np.asarray(mapResponse.map.data),(mapResponse.map.info.width,mapResponse.map.info.height)).T
    print(mapResponse.map.info.height,"Map height")
    print(mapResponse.map.info.width,"Map width")
    print(mapResponse.map.info.origin.position,"The position of map origin (top left corner)")
    print(np.shape(shaped_map),"Map shape")
    #orient map

    #locate robot, set up transform listener
    listener = tf.TransformListener()
    listener.waitForTransform('/base_link', '/map',rospy.Time(),rospy.Duration(3))
    (trans,rot) = listener.lookupTransform('/base_link', '/map', rospy.Time())
    print(trans,"Robot position (meters)")
    trans[0] = -1*trans[0]
    trans[1] = -1*trans[1]
    print(trans,"Robot position corrected (meters)")
    #roboGrid is the robot's position on the map; found using the map's top left corner defined in meters from the origin
    #compared to the robots position in meters from the origin,
    #resulting in the position known in a number of grid cells from the top left corner of the map
    #roboMeters = [(mapResponse.map.info.origin.position.x-trans[0]),(mapResponse.map.info.origin.position.y-trans[1])]
    roboGrid = [abs(round((mapResponse.map.info.origin.position.x-trans[0])*(128/18))),abs(round((mapResponse.map.info.origin.position.y-trans[1])*(128/18)))]
    #transGrid = [round(trans[0]*(128/18)), round(trans[1]*(128/18))]#change translation to grid relation, it was in meters
    #transOrigin = [round(mapResponse.map.info.origin.point[0]*(128/18)), round(mapResponse.map.info.origin.point[1]*(128/18))]
    print(roboGrid,"Robot position (grid cells from top left)")
    
    #segment map
    x_lower_lim = roboGrid[0]-64
    x_upper_lim = roboGrid[0]+64
    y_lower_lim = roboGrid[1]-64
    y_upper_lim = roboGrid[1]+64
    segmented_map = np.full((128,128),-1)
    
    #early code that might be necessary in the case the local segmented map attempts to grab area that is outside the map
    #currently being avoided by just making the initial map really big
    #x_undershoot = roboGrid[0]-63
    #x_overshoot = mapResponse.map.info.width - roboGrid[0]+64
    #y_undershoot = roboGrid[1]-63
    #y_overshoot = mapResponse.map.info.height - roboGrid[1]+64
    #m=0
    #n=0
    #if(x_lower_lim < 0):
    #    m = start at how much it undershoots
    #    x_lower_lim = 0
    #if(x_upper_lim > mapResponse.map.info.width-1):
    #    m = go to next row earlier
    #    x_upper_lim = mapResponse.map.info.width-1
    #if(y_lower_lim < 0):
    #    n = start at how much it undershoots
    #    y_lower_lim = 0
    #if(y_upper_lim > mapResponse.map.info.height-1):
    #    n = go to next column earlier
    #    y_upper_lim = mapResponse.map.info.height-1
    
    for i in range(x_lower_lim,x_upper_lim): 
        for j in range(y_lower_lim,y_upper_lim): 
            segmented_map[i-x_lower_lim][j-y_lower_lim] = shaped_map[i][j]
            #n = n + 1
        #m = m + 1
    #segmented_map = shaped_map[roboGrid[0]-63:roboGrid[0]+64][roboGrid[1]-63:roboGrid[1]+64]
    print(np.shape(segmented_map),"Segmented map shape, should be 128x128")
        
    #threshold map
    lowThresh = 25
    highThresh = 75
    threshmap = np.zeros((1,128,128))#removed 4th dimension with 4 channels (,4) or 3 channels, causing an error with x input @line 130
    dim = np.shape(segmented_map)
    for i in range(dim[0]): 
        for j in range(dim[1]): 
            if(segmented_map[i][j] == -1):# or (segmented_map[i][j] <= highThresh and segmented_map[i][j] >= lowThresh)):
                threshmap[0][i][j] = 127 #unknown [0] 
            elif(segmented_map[i][j] > highThresh):
                threshmap[0][i][j] = 255 #obstacle [2]
            else:
                threshmap[0][i][j] = 0 #free [1]
            #laser_input[i][j][3] = 1 #camera/exit map not used

    #if we have taken a picture (flag) update map
    print(req)
    if(req):
        global cameraMap
        cameraMap = accumulate_non_exits(cameraMap,threshmap,roboGrid)
        #changed reset camera map to create the map to be the same size as the gmapping map that way roboGrid can be used for the pose
        #got the gmapping map size by running the program and checking the printed output which was 352,352 which is slightly different from the expected 355,355

    #segment cameraMap, can use the same limits since the map should be the same shape
    camera_input = np.full((1,128,128),127)
    for i in range(x_lower_lim,x_upper_lim): 
        for j in range(y_lower_lim,y_upper_lim): 
            camera_input[0][i-x_lower_lim][j-y_lower_lim] = cameraMap[i][j]

    #deep learning exit prediction
    vae_model_name = Conditional_New_VAE
    vae_model_path ='camera/rplan_new_cameraTrue_multiTask_exit_u-net_latent64_0315'
    latent_dim = 64
    unet = load_model(vae_model_name,vae_model_path,latent_dim=latent_dim,exit_Flag=True)
    camera_channel = 1
    robotPose = [roboGrid[0], roboGrid[1]]#maybe need to be grid
    #Might need to format map
    laser_input,x_input,semantics_build,centroid_build,exit_build,exit_loc,prob,region_max_prob,region_avr_prob,exit_loc_nearest,Astar_traj,Astar_traj_remapped,Goalpoint,Astar_rplan_Flag, Astar_openSet_Flag = exit_driven_path_plan(unet,shaped_map,segmented_map,threshmap,camera_input,camera_channel,robot_pose=robotPose)
    
    #make an occupancy grid version of segmented&thresholded map for publishing/visualization
    segmented_OccGrid = OccupancyGrid()
    segmented_OccGrid.info.width = 128
    segmented_OccGrid.info.height = 128
    unique,counts = np.unique(segmented_map,return_counts=True)
    print(np.asarray((unique,counts)).T)
    unique,counts = np.unique(laser_input,return_counts=True)
    print(np.asarray((unique,counts)).T)
    print(laser_input[0,:,:].shape,np.unique(laser_input),np.unique(segmented_map))

    print(Goalpoint,"the predicted goal on the map")
    print(exit_loc, "the predicted exit location")
    print(exit_loc_nearest,"the nearest exit")

    fig_astar = plt.figure(1)
    axs_astar = []
    axs_astar.append(fig_astar.add_subplot(1,2,1))
    axs_astar.append(fig_astar.add_subplot(1,2,2))
    axs_astar[0].cla()
    axs_astar[0].title.set_text('local_map')
    axs_astar[0].imshow(laser_input[0,:,:],cmap="gray")
    # axs_astar[0].invert_yaxis()
    axs_astar[0].plot(exit_loc[0], exit_loc[1], "xr",markersize=6,label='exit est.')
    axs_astar[0].plot(64, 64, "og",markersize=6,label='init_pose')
    axs_astar[0].plot(exit_loc_nearest[0], exit_loc_nearest[1],"Dr",markersize=5,label='goal')
    axs_astar[0].plot(Goalpoint[0], Goalpoint[1],"or",markersize=5,label='goalUsed')
    axs_astar[0].axis('off')
    axs_astar[1].cla()
    axs_astar[1].title.set_text('exit_pred')
    axs_astar[1].imshow(exit_build[0,:,:,0],vmax=1.0,vmin=0,cmap='inferno')
    # axs_astar[1].invert_yaxis()
    axs_astar[1].plot(exit_loc[0], exit_loc[1], "xr",markersize=6,label='exit est.')
    axs_astar[1].plot(exit_loc_nearest[0], exit_loc_nearest[1],"Dr",markersize=5,label='goal')
    axs_astar[1].plot(Goalpoint[0], Goalpoint[1],"or",markersize=5,label='goalUsed')
    axs_astar[1].plot(64,64,'og',markersize=5)
    axs_astar[1].axis('off')
    axs_astar[0].legend(framealpha=1, frameon=True,ncol=1, prop={'size': 6},loc='lower right')
    axs_astar[1].legend(framealpha=1, frameon=True,ncol=1, prop={'size': 6},loc='lower right')
    plt.draw()
    plt.pause(0.00000001)
    
    # plt.show()


    #plt.figure(2)
    #plt.imshow(laser_input[0,:,:],cmap="gray")
    #plt.colorbar()
    #plt.draw()
    #plt.pause(0.0000000000000000000001)
    plt.show()

    #saveDir='home/maps/'
    #if not os.path.exists(saveDir):
    #    os.makedirs(saveDir)
    #global round_cnt
    #plt.savefig(saveDir + 'no_camera_local_prediction_step_' + str(round_cnt) + '.png')

    segmented_OccGrid.data = tuple(map(int, laser_input[0,:,:].reshape(128*128,1)-128))
    #make an occupancy grid version of the heat map for publishing/visualization
    heat_OccGrid = OccupancyGrid()
    heat_OccGrid.info.width = 128
    heat_OccGrid.info.height = 128
    heatMap = exit_build[0,:,:,0].numpy()*127.0
    heat_OccGrid.data = tuple(map(int,heatMap.reshape(128*128,1)))
    
    #plt.figure(2)
    #plt.imshow(heatMap)
    #plt.show

    #convert back to robot frame (unsegment) and set up for service reply
    #exitpoint = goalpoint #the goal from deep learning is supposed to be on global map
    #might have to change the point to a positional meters value for move base
    exitpoint = [((Goalpoint[1]-64)*(18/128))+trans[0],((Goalpoint[0]-64)*(18/128))+trans[1]]#the goal is returned on the local map and needs to be turned into meters for move base
    #grid cells into meters relative to robot. Added to robots position relative to origin, to get goal relative to origin
    print(exitpoint,"the predicted goal in meters")
    #round_cnt += 1
    return ExitRequestResponse(exitpoint[0],exitpoint[1],prob,segmented_OccGrid,heat_OccGrid,Goalpoint[1],Goalpoint[0])

def exit_prediction_server():
    rospy.init_node('exit_prediction_server')
    s = rospy.Service('exit_prediction', ExitRequest, handle_exit_request)
    rospy.spin()

if __name__ == "__main__":
    exit_prediction_server()



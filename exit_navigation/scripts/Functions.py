

def accumulate_non_exits(nonExitMap,local_world,pose,exit_threshold=12):
    window_halfsize = local_exit_output.shape[0]//2
    world = np.ones((local_world.shape[0]*2,local_world.shape[1]*2))*map_color['uncertain']
    world[window_halfsize:window_halfsize+local_world.shape[0],
          window_halfsize:window_halfsize+local_world.shape[1]] = local_world

    pose = [window_halfsize + pose[0],window_halfsize + pose[1],0]

    laser_fov = 45
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
    inBound_ind= util.within_bound(np.array([y_rangeCoordMat, x_rangeCoordMat]), world.shape)

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
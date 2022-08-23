# exitNavigation
ROS implementation of a deep learning exit navigation system


apriltag_1 and apriltag_3 are boxes with an apriltag on the side made by Ola Ghattas. I usually place them manually, which requires that you edit Gazebo's path so that it can find where they are. Install apriltag and apriltag_ros using sudo apt-get and edit tags.yaml file so that the standalone tags has:
{id: 1, size: 0.8},
{id: 3, size: 0.8}

the configs folder and the accessories.urdf.xacro are from the Jackal description package by Clearpath Robotics. I have created additional configs that combine using the cameras and the laser simultaneously. I also edited the accessories.urdf.xacro file so that the flea camera is not tilted downwards.

Exit navigation is the primary package in the workspace. Make sure your workspace is running with python 3. You will need to add a folder called learning with the deep learning model as it was too big to upload to github.

Launch and params are folders within the Jackal Navigation package by Clearpath Robotics that I have edited to create custom settings for.

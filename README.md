# exitNavigation
ROS implementation of a deep learning exit navigation system


apriltag_1 and apriltag_3 are boxes with an apriltag on the side made by Ola Ghattas. I usually place them manually, which requires that you edit Gazebo's path so that it can find where they are. Install apriltag and apriltag_ros using sudo apt-get and edit tags.yaml file so that the standalone tags has:
{id: 1, size: 0.8},
{id: 3, size: 0.8}

The configs folder and the accessories.urdf.xacro are from the Jackal description package by Clearpath Robotics. I have created additional configs that combine using the cameras and the laser simultaneously. I also edited the accessories.urdf.xacro file so that the flea camera is not tilted downwards.

Exit navigation is the primary package in the workspace. Make sure your workspace is running with python 3. You will need to add a folder called learning with the deep learning model as it was too big to upload to github. To get a python 3 workspace working on Ubuntu 18.04, you may need to perform some additional steps found here https://answers.ros.org/question/326226/importerror-dynamic-module-does-not-define-module-export-function-pyinit__tf2/. This includes installing empy, and setting up a separate version of TF. Commands copied below in case the link is lost in the future.

Launch and params are folders within the Jackal Navigation package by Clearpath Robotics that I have edited to create custom settings for.


Commands copied from the ros.org post for getting the python3 workspace working:

Install some prerequisites to use Python3 with ROS.

sudo apt update
sudo apt install python3-catkin-pkg-modules python3-rospkg-modules python3-empy


Prepare catkin workspace

mkdir -p ~/catkin_ws/src; cd ~/catkin_ws
catkin_make
source devel/setup.bash
wstool init
wstool set -y src/geometry2 --git https://github.com/ros/geometry2 -v 0.6.5
wstool up
rosdep install --from-paths src --ignore-src -y -r


Finally compile for Python 3

catkin_make --cmake-args \
            -DCMAKE_BUILD_TYPE=Release \
            -DPYTHON_EXECUTABLE=/usr/bin/python3 \
            -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m \
            -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so

FROM ros:noetic-ros-core-focal

# install bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    python3-rosdep \
    python3-rosinstall \
    python3-vcstools \
    && rm -rf /var/lib/apt/lists/*

# bootstrap rosdep
RUN rosdep init && \
  rosdep update --rosdistro $ROS_DISTRO

# install ros packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-ros-base=1.5.0-1* \
    && rm -rf /var/lib/apt/lists/*


# Install ROS 2 Foxy
RUN apt-get update && apt-get install --no-install-recommends -y curl gnupg2 lsb-release && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    sh -c 'echo "deb [arch=amd64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu focal main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null' && \
    apt-get update && \ 
    apt-get upgrade -y --no-install-recommends && \
    apt-get install --no-install-recommends -y \
    ros-foxy-desktop \
    python3-argcomplete \
    python3-pip \
    ros-dev-tools

# Source ROS
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
RUN echo "source /opt/ros/foxy/setup.bash" >> ~/.bashrc
RUN echo "source /ros1_bridge/install/local_setup.bash" >> ~/.bashrc

RUN git clone -b foxy https://github.com/ros2/ros1_bridge.git ros1_bridge && \
    /bin/bash -c "source /opt/ros/foxy/setup.bash && \
                  source /opt/ros/noetic/setup.bash && \
                  cd ros1_bridge && \
                  colcon build --symlink-install --packages-select ros1_bridge --cmake-force-configure && \
                  source install/local_setup.bash"

RUN pip install opencv-python Pillow transforms3d "scipy<1.6.0"

RUN apt-get install --no-install-recommends -y \ 
    ros-foxy-sensor-msgs-py \
    ros-foxy-plotjuggler-ros \
    ros-foxy-cv-bridge \
    ros-foxy-tf-transformations \
    ros-foxy-vision-opencv


RUN apt-get install --no-install-recommends -y \
    python3.9 python3.9-distutils && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9
RUN python3.9 -m pip install google-genai flask



# Set up entrypoint to source both ROS 1 and ROS 2
# ENTRYPOINT ["/bin/bash", "-c", "source /opt/ros/noetic/setup.bash && source /opt/ros/foxy/setup.bash && exec \"$@\""]

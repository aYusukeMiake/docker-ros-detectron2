# base image
FROM nvidia/cudagl:11.3.0-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

# Setup basic packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    vim \
    pkg-config \
    wget \
    zip \
    unzip \
    tzdata \
    lsb-release \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoclean

# Installl ROS noetic
# http://wiki.ros.org/noetic/Installation/Ubuntu
# Add the package keys.
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
# Update the package list.
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
# Install 'ros-noetic-desktop-full' packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-desktop-full \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoclean

# Setup basic ROS packages
RUN apt-get update && apt-get install --no-install-recommends -y \
    python3-rosdep \
    python3-rosinstall-generator \
    python3-vcstool \
    build-essential \
    python3-mock \
    && rm -rf /var/lib/apt/lists/*
RUN rosdep init
RUN rosdep update

# Install catkin build
RUN apt-get update && apt-get install --no-install-recommends -y \
    python3-osrf-pycommon \
    python3-catkin-tools \
    && rm -rf /var/lib/apt/lists/*

# build-catkin_ws
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc
RUN echo "cd /root/ros/catkin_ws; catkin build" >> /root/.bashrc
RUN echo "source /root/ros/catkin_ws/devel/setup.bash" >> /root/.bashrc

WORKDIR /root

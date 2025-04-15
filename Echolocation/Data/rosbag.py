#!/usr/bin/env python

import subprocess
import time
import os
import signal

def get_next_bag_filename(directory, base_name="my_robot_data"):
    index = 1
    while os.path.exists(os.path.join(directory, "{}_{}.bag".format(base_name, index))):
        index += 1
    return os.path.join(directory, "{}_{}.bag".format(base_name, index))

def record_rosbag(bag_file, topics, duration):
    # Construct the rosbag record command
    command = ['rosbag', 'record', '-O', bag_file] + topics

    print("Recording the following topics: {}".format(", ".join(topics)))
    print("Saving to bag file: {}".format(bag_file))
    print("Recording for {} seconds...".format(duration))

    # Start the rosbag record command
    process = subprocess.Popen(command)
    
    # Wait for the specified duration (in seconds)
    time.sleep(duration)
    
    # Stop the rosbag recording after the duration
    process.send_signal(signal.SIGINT) 
    print("Recording stopped after {} seconds.".format(duration))
if __name__ == '__main__':
    # Directory to save the bag files
    save_directory = "/home/robot/group_665/rosbags"

    # Ensure the directory exists
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # topics
    topics = ["/robot/front_laser/scan_filtered", "/robot/front_rgbd_camera/rgb/image_raw", "/robot/front_rgbd_camera/depth_registered/points"]

    # Duration (in seconds)
    duration = 10

    # Get the next available filename
    bag_file = get_next_bag_filename(save_directory)

    # Function to start and stop recording
    record_rosbag(bag_file, topics, duration)

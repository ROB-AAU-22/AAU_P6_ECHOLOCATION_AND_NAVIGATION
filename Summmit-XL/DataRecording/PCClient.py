#!/usr/bin/env python
import rospy
from std_msgs.msg import Float32MultiArray, MultiArrayLayout, MultiArrayDimension
import os
import time
import json
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from scipy.io.wavfile import write
from LidarNode import LaserScanListener
from ROSbag import start_recording, stop_recording


# Constants
NUM_SWEEPS = 1
FILE_DIR = '/media/volle/USB/DataRecording/'
SAMPLE_RATE = 96000

# Initialize LaserScanListener
laser_scan_listener = LaserScanListener(FILE_DIR)

class SoundListen():
    def __init__(self):
        self.sound_data = None

    def callback_sound(self, msg):
        rospy.loginfo("Received sound file data")

        # Convert the received data back to a numpy array
        sound_file_data = np.array(msg.data, dtype=np.float32)
        sound_file_data = sound_file_data.reshape((msg.layout.dim[0].size, msg.layout.dim[1].size))
        
        print(sound_file_data)
        self.sound_data = sound_file_data
        print(sound_file_data.shape)
        rospy.signal_shutdown("We got the soundfile")


    def sound_file_subscriber(self):
        
        rospy.Subscriber('/sound_file_raw', Float32MultiArray, self.callback_sound)

        rospy.loginfo("Sound file subscriber is ready")
        rospy.spin()  # Keeps the subscriber running

    

if __name__ == '__main__':
    WavListener = SoundListen()
    try:
        WavListener.sound_file_subscriber()
    except rospy.ROSInterruptException:
        pass
    laser_distance = laser_scan_listener.receive_laser_scan()
    print(laser_distance)
    print(WavListener.sound_data)
    write("/home/volle/Desktop/wav",SAMPLE_RATE,WavListener.sound_data)
    laser_angle = laser_scan_listener.angle_measurement(laser_distance)
    config_distance = {
        "LiDAR_distance": laser_distance,  # LiDAR distance data
    }
    config_angle = {
        "LiDAR_angle": laser_angle,  # LiDAR angle data
    }
    distance_file_path = "/home/volle/Desktop/dist.json"
    with open(distance_file_path, 'w') as config_file:
        json.dump(config_distance["LiDAR_distance"],config_file, indent=4)

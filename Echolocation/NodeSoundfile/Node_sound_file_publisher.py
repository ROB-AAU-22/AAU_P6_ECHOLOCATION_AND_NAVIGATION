#!/usr/bin/env python

import rospy
from std_msgs.msg import ByteMultiArray
import os

sound_folder = "path/"  # Make sure to replace this with the correct path

def publish_raw_sound_files():
    rospy.init_node('sound_file_raw_publisher')
    pub = rospy.Publisher('/sound_file_raw', ByteMultiArray, queue_size=10)
    rate = rospy.Rate(10)

    # Folder that contains sound files
    if not os.path.exists(sound_folder):
        rospy.logerr("Folder not found: %s" % sound_folder)
        return
    
    # Flag to check if any .wav files were found
    wav_files_found = False

    for filename in os.listdir(sound_folder):
        if filename.endswith('.wav'):
            wav_files_found = True
            full_path = os.path.join(sound_folder, filename)
            with open(full_path, 'rb') as f:
                data = f.read()
                msg = ByteMultiArray()
                msg.data = list(data)
                rospy.loginfo("Publishing raw data from: %s" % filename)
                pub.publish(msg)
                rate.sleep()
        else:
            rospy.logwarn("Skipping non-wav file: %s" % filename)

    # Check if no .wav files were found and log a warning
    if not wav_files_found:
        rospy.logwarn("No .wav files found in the folder: %s" % sound_folder)

if __name__ == '__main__':
    try:
        publish_raw_sound_files()
    except rospy.ROSInterruptException:
        pass

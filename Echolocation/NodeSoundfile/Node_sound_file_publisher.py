#!/usr/bin/env python

import rospy
from std_msgs.msg import ByteMultiArray
import os

sound_folder="path/"

def publish_raw_sound_files():
    rospy.init_node('sound_file_raw_publisher')
    pub = rospy.Publisher('/sound_file_raw', ByteMultiArray, queue_size=10)
    rate = rospy.Rate(10)

    # Folder that contains sound files
    if not os.path.exists(sound_folder):
        rospy.logerr(f"Folder not found: {sound_folder}")
        return
    

    for filename in os.listdir(sound_folder):
        if filename.endswith('.wav'):
            full_path = os.path.join(sound_folder, filename)
            with open(full_path, 'rb') as f:
                data = f.read()
                msg = ByteMultiArray()
                msg.data = list(data)
                rospy.loginfo(f"Publishing raw data from: {filename}")
                pub.publish(msg)
                rate.sleep()
        else:
            rospy.logwarn(f"Skipping non-wav file: {filename}")

if __name__ == '__main__':
    try:
        publish_raw_sound_files()
    except rospy.ROSInterruptException:
        pass
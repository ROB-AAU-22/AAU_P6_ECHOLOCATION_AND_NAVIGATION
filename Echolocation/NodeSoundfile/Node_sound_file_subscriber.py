#!/usr/bin/env python

import rospy
from std_msgs.msg import ByteMultiArray
import os
from datetime import datetime

def callback(msg):
    # Create an output folder if it doesn't exist
    output_folder = "path"  # Make sure to update with the correct folder path
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Use timestamp to generate unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = os.path.join(output_folder, "received_%s.wav" % timestamp)

    with open(output_path, 'wb') as f:
        f.write(bytearray(msg.data))

    rospy.loginfo("Received and saved WAV file to: %s" % output_path)

def listener():
    rospy.init_node('sound_file_subscriber')
    rospy.Subscriber('/sound_file_raw', ByteMultiArray, callback)
    rospy.loginfo("Subscriber node started. Waiting for sound data...")
    rospy.spin()

if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass

#!/usr/bin/env python

import rospy
from std_msgs.msg import ByteMultiArray
import os
from datetime import datetime

def callback(msg):
    # Log the number of bytes received
    rospy.loginfo("Received %d bytes of data" % len(msg.data))  # Corrected for Python 2.7

    # Expand the ~ to the full path for the home directory
    output_folder = os.path.expanduser("~/Desktop")  # Changed output folder to Desktop

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Use timestamp to generate unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = os.path.join(output_folder, "received_%s.wav" % timestamp)

    # Save the received data to the file
    with open(output_path, 'wb') as f:
        f.write(bytearray(msg.data))

    # Log that the file has been saved
    rospy.loginfo("Received and saved WAV file to: %s" % output_path)  # Corrected for Python 2.7

def listener():
    # Initialize the ROS node
    rospy.init_node('sound_file_subscriber')

    # Subscribe to the topic where the data is being published
    rospy.Subscriber('/sound_file_raw', ByteMultiArray, callback)

    # Log that the subscriber node is running
    rospy.loginfo("Subscriber node started. Waiting for sound data...")

    # Keep the node running to process messages
    rospy.spin()

if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass

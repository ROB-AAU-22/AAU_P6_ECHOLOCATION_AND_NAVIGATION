#!/usr/bin/env python

import rospy
from std_msgs.msg import ByteMultiArray
import os
from datetime import datetime

# Buffer to store chunks
buffer = bytearray()

def callback(msg):
    global buffer
    # Append the received chunk to the buffer
    buffer.extend(msg.data)

    # Log the number of bytes received
    rospy.loginfo("Received %d bytes of data" % len(msg.data))

    # Check if we've received all the chunks (if you know the total file size, you could use it)
    # For now, we'll just save it periodically or once the message is received
    # For large files, you might need a different method to determine when to save the complete file

    # Use timestamp to generate unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_folder = os.path.expanduser("~/Desktop")
    output_path = os.path.join(output_folder, "received_%s.wav" % timestamp)

    # Check if the full file is received (for example, you could check buffer size against a known file size)
    if len(buffer) > 0:
        with open(output_path, 'wb') as f:
            f.write(buffer)  # Write all buffered data to the file
        rospy.loginfo("Received and saved WAV file to: %s" % output_path)
        buffer = bytearray()  # Clear buffer for next file

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

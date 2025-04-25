#!/usr/bin/env python
import rospy
import numpy as np
from std_msgs.msg import Float32MultiArray

def callback(msg):
    rospy.loginfo("Received sound file data")

    # Convert the received data back to a numpy array
    sound_file_data = np.array(msg.data, dtype=np.float32)
    sound_file_data = sound_file_data.reshape((msg.layout.dim[0].size, msg.layout.dim[1].size))

    print(sound_file_data.shape)


def sound_file_subscriber():
    rospy.init_node('sound_file_subscriber')
    rospy.Subscriber('/sound_file_raw', Float32MultiArray, callback)

    rospy.loginfo("Sound file subscriber is ready")
    rospy.spin()  # Keeps the subscriber running

if __name__ == '__main__':
    try:
        sound_file_subscriber()
    except rospy.ROSInterruptException:
        pass
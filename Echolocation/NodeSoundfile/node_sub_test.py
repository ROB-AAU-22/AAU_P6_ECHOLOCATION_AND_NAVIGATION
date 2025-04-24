# Test Subscriber
#!/usr/bin/env python
import rospy
from std_msgs.msg import ByteMultiArray

def callback(msg):
    rospy.loginfo("Received data: %s", msg.data)

def test_listener():
    rospy.init_node('test_listener')
    rospy.Subscriber('/sound_file_raw', ByteMultiArray, callback)
    rospy.spin()  # Keeps the subscriber running

if __name__ == '__main__':
    try:
        test_listener()
    except rospy.ROSInterruptException:
        pass

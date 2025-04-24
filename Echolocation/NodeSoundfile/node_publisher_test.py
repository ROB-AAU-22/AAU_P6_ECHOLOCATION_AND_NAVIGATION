# Test Publisher
#!/usr/bin/env python
import rospy
from std_msgs.msg import ByteMultiArray

def test_publisher():
    rospy.init_node('test_publisher')
    pub = rospy.Publisher('/sound_file_raw', ByteMultiArray, queue_size=10)
    rate = rospy.Rate(1)  # Publish at 1 Hz

    while not rospy.is_shutdown():
        msg = ByteMultiArray()
        msg.data = [1, 2, 3, 4, 5]  # Simple test data
        rospy.loginfo("Publishing simple data")
        pub.publish(msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        test_publisher()
    except rospy.ROSInterruptException:
        pass

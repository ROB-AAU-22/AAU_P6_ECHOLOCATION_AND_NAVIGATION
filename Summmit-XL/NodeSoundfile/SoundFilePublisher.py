#!/usr/bin/env python
import rospy
import soundfile as sf
from std_msgs.msg import Float32MultiArray, MultiArrayLayout, MultiArrayDimension

def publisher():
    rospy.init_node('sound_file_publisher')
    
    pub = rospy.Publisher('/sound_file_raw', Float32MultiArray, queue_size=10)
    rate = rospy.Rate(1)  # 1 Hz
    
    sound_file_path = "/home/robot/group_665/x.wav"  # Replace with your sound file path
    sound_file, sr = sf.read(sound_file_path)
    flattened_sound_file = sound_file.flatten()
    
    while not rospy.is_shutdown():
        array_layout = MultiArrayLayout()
        array_layout.dim.append(MultiArrayDimension(label='rows', size=sound_file.shape[0], stride=sound_file.shape[0]*2))
        array_layout.dim.append(MultiArrayDimension(label='cols', size=sound_file.shape[1], stride=sound_file.shape[1]))
        array_layout.data_offset = 0
        
        msg = Float32MultiArray()
        msg.layout = array_layout
        msg.data = flattened_sound_file.tolist()
        
        rospy.loginfo("Publishing sound file data")
        
        pub.publish(msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        publisher()
    except rospy.ROSInterruptException:
        pass
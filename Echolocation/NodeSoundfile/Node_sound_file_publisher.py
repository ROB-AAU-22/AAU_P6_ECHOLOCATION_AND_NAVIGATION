#!/usr/bin/env python
import rospy
from std_msgs.msg import ByteMultiArray
import os

# Expand ~ to the user's home directory
sound_folder = os.path.expanduser("~/group_665/old/sound/dataset/")

def publish_raw_sound_files():
    rospy.init_node('sound_file_raw_publisher')
    pub = rospy.Publisher('/sound_file_raw', ByteMultiArray, queue_size=10)
    rate = rospy.Rate(10)  # 10 Hz, adjust based on your needs

    # Check if folder exists
    if not os.path.exists(sound_folder):
        rospy.logerr("Folder not found: %s" % sound_folder)
        return
    
    wav_files_found = False

    # Iterate over files in the folder
    for filename in os.listdir(sound_folder):
        if filename.endswith('.wav'):
            wav_files_found = True
            full_path = os.path.join(sound_folder, filename)
            try:
                with open(full_path, 'rb') as f:
                    data = f.read()
                    msg = ByteMultiArray()
                    msg.data = list(data)  # Convert byte data to list
                    
                    # Log the number of bytes being published
                    rospy.loginfo("Publishing %d bytes of data from: %s" % (len(msg.data), filename))
                    
                    pub.publish(msg)
                    rate.sleep()  # Control the publishing rate
            except Exception as e:
                rospy.logerr("Error reading file %s: %s" % (filename, str(e)))
        else:
            rospy.logwarn("Skipping non-wav file: %s" % filename)

    if not wav_files_found:
        rospy.logwarn("No .wav files found in the folder: %s" % sound_folder)

if __name__ == '__main__':
    try:
        publish_raw_sound_files()
    except rospy.ROSInterruptException:
        pass

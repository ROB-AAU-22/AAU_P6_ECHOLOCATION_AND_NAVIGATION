#!/usr/bin/env python
import rospy
from std_msgs.msg import ByteMultiArray
import os

# Expand ~ to the user's home directory
sound_folder = os.path.expanduser("~/group_665/old/sound/dataset/")

# Chunk size for large files (in bytes)
CHUNK_SIZE = 1024 * 1024  # 1MB chunks

def publish_raw_sound_files():
    rospy.init_node('sound_file_raw_publisher')
    pub = rospy.Publisher('/sound_file_raw', ByteMultiArray, queue_size=10)
    rate = rospy.Rate(0.1)  # 0.1 Hz = 1 file every 10 seconds

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
            rospy.loginfo("Found file: %s" % filename)

            try:
                with open(full_path, 'rb') as f:
                    data = f.read()  # Read the entire file into memory
                    # Check if file is too large
                    if len(data) > CHUNK_SIZE:
                        rospy.logwarn("File is too large, splitting into chunks: %s" % filename)
                        # Split the file into chunks
                        chunks = [data[i:i + CHUNK_SIZE] for i in range(0, len(data), CHUNK_SIZE)]
                        for chunk in chunks:
                            msg = ByteMultiArray()
                            msg.data = list(chunk)  # Convert byte data to list
                            # Log the first 10 bytes of the message
                            rospy.loginfo("Publishing first 10 bytes from chunk of %s: %s" % (filename, msg.data[:10]))
                            pub.publish(msg)
                            rate.sleep()  # Control the publishing rate
                    else:
                        # If file is not too large, publish it as a whole
                        msg = ByteMultiArray()
                        msg.data = list(data)  # Convert byte data to list
                        # Log the first 10 bytes of the message
                        rospy.loginfo("Publishing first 10 bytes from: %s: %s" % (filename, msg.data[:10]))
                        pub.publish(msg)
                        rate.sleep()  # Control the publishing rate

            except Exception as e:
                rospy.logerr("Error reading or publishing file %s: %s" % (filename, str(e)))
        else:
            rospy.logwarn("Skipping non-wav file: %s" % filename)

    if not wav_files_found:
        rospy.logwarn("No .wav files found in the folder: %s" % sound_folder)

if __name__ == '__main__':
    try:
        publish_raw_sound_files()
    except rospy.ROSInterruptException:
        rospy.logerr("ROS Interrupt exception caught.")
    except Exception as e:
        rospy.logerr("An unexpected error occurred: %s" % str(e))


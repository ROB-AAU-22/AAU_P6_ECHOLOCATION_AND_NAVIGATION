#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" ROS1 (Melodic) WAV file publisher using ByteMultiArray """
import rospy
from std_msgs.msg import ByteMultiArray

# Configuration: adjust topic name, file path, and chunk size if needed
TOPIC_NAME = "wav_bytes"
FILE_PATH  = rospy.get_param("~file_path", "input.wav")    # WAV file to send (path can be overridden via ROS param)
CHUNK_SIZE = rospy.get_param("~chunk_size", 1024*1024)     # safe chunk size threshold in bytes (default 1 MB)

def main():
    rospy.init_node("wav_file_publisher", anonymous=False)
    pub = rospy.Publisher(TOPIC_NAME, ByteMultiArray, queue_size=10)
    rospy.loginfo("Starting WAV file publisher...")

    # Read the full WAV file as binary data (including header and audio samples)
    try:
        with open(FILE_PATH, "rb") as f:
            data = f.read()
    except Exception as e:
        rospy.logerr("Failed to read file '%s': %s", FILE_PATH, str(e))
        return

    total_bytes = len(data)
    if total_bytes == 0:
        rospy.logwarn("File '%s' is empty, nothing to send.", FILE_PATH)
        return

    # Determine if we need to split into chunks
    if total_bytes > CHUNK_SIZE:
        total_chunks = (total_bytes + CHUNK_SIZE - 1) // CHUNK_SIZE  # round up division
        rospy.loginfo("File size %d bytes exceeds chunk threshold. Splitting into %d chunks of up to %d bytes.",
                      total_bytes, total_chunks, CHUNK_SIZE)
    else:
        total_chunks = 1
        rospy.loginfo("File size %d bytes is within threshold. Sending in a single chunk.", total_bytes)

    bytes_sent = 0
    # Publish the file data, chunk by chunk if necessary
    for i in range(0, total_bytes, CHUNK_SIZE):
        chunk = data[i:i+CHUNK_SIZE]
        # Convert chunk (bytes) to list of uint8 ints for ByteMultiArray
        chunk_list = [ord(c) for c in chunk]   # `ord` gets the integer value of each byte (0–255)
        msg = ByteMultiArray()
        msg.data = chunk_list
        pub.publish(msg)
        bytes_sent += len(chunk_list)
        chunk_idx = i // CHUNK_SIZE + 1
        rospy.loginfo("Published chunk %d/%d (%d bytes)", chunk_idx, total_chunks, len(chunk_list))
        rospy.sleep(0.01)  # small delay to avoid flooding the network/queue

    # Publish an empty message as an end-of-file marker
    end_msg = ByteMultiArray()
    end_msg.data = []  # no data signifies termination
    pub.publish(end_msg)
    rospy.loginfo("Published end-of-file marker. Total bytes sent: %d", bytes_sent)

    # Give time for messages to propagate before shutting down
    rospy.sleep(0.5)
    rospy.loginfo("WAV file transmission complete.")

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass

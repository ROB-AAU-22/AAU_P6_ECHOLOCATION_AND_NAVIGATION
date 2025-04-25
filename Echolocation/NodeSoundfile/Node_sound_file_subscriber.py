#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" ROS1 (Melodic) WAV file subscriber using ByteMultiArray """
import rospy
from std_msgs.msg import ByteMultiArray

# Configuration: adjust topic name and output file path if needed
TOPIC_NAME  = "wav_bytes"
OUTPUT_FILE = rospy.get_param("~output_file", "received.wav")  # file path to save the received WAV

# Globals to keep track of file writing state
file_handle    = None
bytes_received = 0

def callback(msg):
    global file_handle, bytes_received
    # Check for end-of-file marker (empty data array)
    if len(msg.data) == 0:
        rospy.loginfo("Received end-of-file marker. Total bytes received: %d", bytes_received)
        if file_handle:
            file_handle.close()
            rospy.loginfo("WAV file successfully written to '%s' (%d bytes).", OUTPUT_FILE, bytes_received)
        else:
            rospy.logwarn("End-of-file marker received but no file was open.")
        # Terminate the subscriber node after file transfer is complete
        rospy.signal_shutdown("File transfer complete")
        return

    # If this is the first chunk, open the output file
    if file_handle is None:
        try:
            file_handle = open(OUTPUT_FILE, "wb")
        except Exception as e:
            rospy.logerr("Failed to open output file '%s': %s", OUTPUT_FILE, str(e))
            rospy.signal_shutdown("File open error")
            return

    # Write the incoming bytes to the file
    # msg.data is a sequence of uint8 values; convert to bytes for writing
    chunk_bytes = bytearray(msg.data)   # bytearray interprets the list of ints as bytes
    file_handle.write(chunk_bytes)
    bytes_received += len(msg.data)
    rospy.loginfo("Received chunk of %d bytes (total %d bytes so far)", len(msg.data), bytes_received)

def main():
    rospy.init_node("wav_file_subscriber", anonymous=False)
    rospy.Subscriber(TOPIC_NAME, ByteMultiArray, callback, queue_size=10)
    rospy.loginfo("WAV file subscriber started, waiting for data...")
    rospy.spin()  # Keep the node running until shut down in the callback

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass

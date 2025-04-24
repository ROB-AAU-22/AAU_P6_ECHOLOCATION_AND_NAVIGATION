#!/usr/bin/env python
import rospy
import numpy as np
import soundfile as sf
from audio_common_msgs.msg import AudioData
import os

# If known in advance
SAMPLE_RATE = 96000
NUM_CHANNELS = 2   # Set to 1 for mono, 2 for stereo, etc.
OUTPUT_FILE = "/home/robot/group_665"
OUTPUT_FILE= os.path.expanduser("~/group_665/old/sound/Recordings/sample1.wav")
def callback(msg):
    rospy.loginfo("Received audio data")

    # Convert bytes back to float32 array
    audio = np.frombuffer(msg.data, dtype=np.float32)

    # If it's multi-channel, reshape it appropriately
    if NUM_CHANNELS > 1:
        audio = audio.reshape(-1, NUM_CHANNELS)

    # Save the audio file
    sf.write(OUTPUT_FILE, audio, SAMPLE_RATE, subtype='FLOAT')
    rospy.loginfo(f"Saved audio to {OUTPUT_FILE}")

def audio_subscriber():
    rospy.init_node('audio_subscriber', anonymous=True)
    rospy.Subscriber('audio_data', AudioData, callback)
    rospy.spin()

if __name__ == '__main__':
    try:
        audio_subscriber()
    except rospy.ROSInterruptException:
        pass

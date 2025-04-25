#!/usr/bin/env python
import rospy
import numpy as np
import soundfile as sf
import os
from audio_common_msgs.msg import AudioData

# Known values
SAMPLE_RATE = 96000
NUM_CHANNELS = 2  # Set to 1 for mono, 2 for stereo

OUTPUT_FILE = os.path.expanduser("~/group_665/old/sound/recordings/received_sample.wav")

def callback(msg):
    rospy.loginfo("Received audio data.")

    # Convert bytes back to float32 array
    audio = np.frombuffer(msg.data, dtype=np.float32)

    # If multi-channel, reshape
    if NUM_CHANNELS > 1:
        audio = audio.reshape(-1, NUM_CHANNELS)

    # Ensure directory exists
    output_dir = os.path.dirname(OUTPUT_FILE)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save audio file
    sf.write(OUTPUT_FILE, audio, SAMPLE_RATE, subtype='FLOAT')
    rospy.loginfo("Saved audio to %s" % OUTPUT_FILE)

def audio_subscriber():
    rospy.init_node('audio_subscriber', anonymous=True)
    rospy.Subscriber('audio_data', AudioData, callback, queue_size=10)
    rospy.spin()

if __name__ == '__main__':
    try:
        audio_subscriber()
    except rospy.ROSInterruptException:
        pass

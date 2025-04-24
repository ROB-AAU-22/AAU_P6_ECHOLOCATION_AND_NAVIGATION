#!/usr/bin/env python
import rospy
import soundfile as sf
from audio_common_msgs.msg import AudioData
import os

def audio_publisher():
    rospy.init_node('audio_publisher', anonymous=True)
    pub = rospy.Publisher('audio_data', AudioData, queue_size=10)
    #data, samplerate = os.path.expanduser("~/group_665/old/sound/Recordings/sample1.wav")
    # Load the .wav file as float32 audio
    data, samplerate = sf.read("/home/robot/group_665/old/sound/recordings/sample1.wav", dtype='float32')  # e.g. shape = (samples, channels) or (samples,)

    # Convert float32 data to bytes
    msg = AudioData()
    msg.data = data.astype('float32').tobytes()

    rospy.loginfo("Publishing audio data with sample rate: %d Hz, shape: %s" % (samplerate, data.shape))
    pub.publish(msg)

    rospy.sleep(1)  # allow time for subscriber to receive

if __name__ == '__main__':
    try:
        audio_publisher()
    except rospy.ROSInterruptException:
        pass

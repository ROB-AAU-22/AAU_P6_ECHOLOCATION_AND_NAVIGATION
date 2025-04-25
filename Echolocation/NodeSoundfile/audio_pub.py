#!/usr/bin/env python
import rospy
import soundfile as sf
from audio_common_msgs.msg import AudioData

def audio_publisher():
    rospy.init_node('audio_publisher', anonymous=True)
    pub_audio = rospy.Publisher('audio_data', AudioData, queue_size=10)

    file_path = "/home/robot/group_665/old/sound/recordings/sample1.wav"
    data, samplerate = sf.read(file_path, dtype='float32')

    # Determine number of channels and samples
    if len(data.shape) == 1:
        num_channels = 1
        num_samples = data.shape[0]
    else:
        num_channels = data.shape[1]
        num_samples = data.shape[0]

    # Publish audio data
    msg = AudioData()
    msg.data = bytearray(data.astype('float32').tobytes())
    rospy.loginfo("Published audio: sample_rate=%d, channels=%d, samples=%d" %
                  (samplerate, num_channels, num_samples))
    pub_audio.publish(msg)

    rospy.sleep(1.0)  # sleep to allow transmission

if __name__ == '__main__':
    try:
        audio_publisher()
    except rospy.ROSInterruptException:
        pass

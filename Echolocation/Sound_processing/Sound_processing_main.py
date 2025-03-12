import wave

def process_sound(sound_file_path):
    print("Processing sound file: ")
    sound = wave.open(sound_file_path,'rb')
    print("Amount of channles", sound.getnchannels())
    print("sample widht", sound.getsampwidth())
    print("framerate", sound.getframerate())
    print("number of frames", sound.getnframes())
    print("all prams", sound.getparams())
    print("length of sound", sound.getnframes()/sound.getframerate())
    return 0





import soundfile as sf

filename = "C:\github\AAU_P6_ECHOLOCATION_AND_NAVIGATION\Summmit-XL\DataRecording\chirps\ChirpsHigh\chirp_10kHz-20kHz_1ms.wav"
info = sf.info(filename)

print(f"File      : {filename}")
print(f"Samplerate: {info.samplerate} Hz")
print(f"Channels  : {info.channels}")
print(f"Subtype   : {info.subtype}")
print(f"Format    : {info.format}")
print(f"Frames    : {info.frames}")
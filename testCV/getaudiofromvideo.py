#encoding=utf-8
from pydub import AudioSegment

song = AudioSegment.from_mp3("/home/zzy/Videos/other/ladinggirlWAV.wav")

song10 = AudioSegment.silent(duration=2000)

song1 = song[0:1000]+song[0:1000] + song

song2 = song[0:1000]+song10+song[0:3000]+song
#song = song[:21*1000]
song1.export("/home/zzy/Videos/other/ladinggirlWAV10.wav",format="wav")
song2.export("/home/zzy/Videos/other/ladinggirlWAV20.wav",format="wav")

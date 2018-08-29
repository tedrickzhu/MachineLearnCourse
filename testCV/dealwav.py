#encoding=utf-8

from pydub import AudioSegment
import os

filepath = '/home/zzy/Downloads/221974/'

file_list = os.listdir(filepath)

first_wav = AudioSegment.from_wav(filepath+'221974.wav')

silence = first_wav[2000:2040]

# test1 = silence+first_wav+silence
# test1.export(filepath+'221974_tmp.wav',format='wav')

for file_name in file_list:
    name = file_name.split('.')[0]
    print name
    wav_file = AudioSegment.from_wav(filepath+file_name)
    wav_file = silence+wav_file+silence
    wav_file.export(filepath+name+'_new.wav',format='wav')
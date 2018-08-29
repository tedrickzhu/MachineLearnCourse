#encoding=utf-8

from pydub import AudioSegment
#import os, re


# song1 = AudioSegment.from_mp3("/home/zzy/Music/happy78s.mp3")
# song1 = song1[:65.6*1000]
# song1.export("/home/zzy/Music/happy65s.mp3",format="mp3")
song1 = AudioSegment.from_mp3("/home/zzy/Music/happy65s.mp3")

half_second_silence = AudioSegment.silent(duration=500)
two_second_silence = AudioSegment.silent(duration=1000)

# song2 = AudioSegment.from_mp3("/home/zzy/Music/lookhim2_15s.mp3")
# song2 = song2[3.5*1000:10*1000]
# song2.export("/home/zzy/Music/lookhimfore2.mp3",format="mp3")
# song2 = AudioSegment.from_mp3("/home/zzy/Music/lookhim2_40s.mp3")
# song2 = song2[17.5*1000:30*1000]
# song2.export("/home/zzy/Music/lookhim2_ol.mp3",format="mp3")

song20 = AudioSegment.from_mp3("/home/zzy/Music/lookhimfore2.mp3")

song2 = AudioSegment.from_mp3("/home/zzy/Music/djlookhim21.mp3")

# song3 = AudioSegment.from_mp3("/home/zzy/Music/ilu47end.mp3")
# song3= song3[0.5*1000:40.5*1000]
# song3.export("/home/zzy/Music/iov40.mp3",format="mp3")
song3 = AudioSegment.from_mp3("/home/zzy/Music/iov40.mp3")

# song20=song2[6*1000:12.5*1000]
song21=song2[6*1000:12.7*1000]+song2[6*1000:12.7*1000]+song2[12.5*1000:19*1000]+song2[6*1000:12.7*1000]+song2[6*1000:12.7*1000]
song22=song2[12.5*1000:19*1000]

song = song1+half_second_silence+song21+song22+two_second_silence+song3
#song = song[:21*1000]
song.export("/home/zzy/Music/result6.mp3",format="mp3")

#song.export("/home/zzy/Music/djlookhim21.mp3",format="mp3")
# 循环目录下所有文件
# for each in os.listdir('.'):
#     filename = re.findall(r"(.*?)\.mp3", each) # 取出.mp3后缀的文件名
#     if filename:
#         filename[0] += '.mp3'
#         mp3 = AudioSegment.from_mp3(filename[0]) # 打开mp3文件
#         mp3[17*1000+500:].export(filename[0], format="mp3") # 切割前17.5秒并覆盖保存

# enPath = "%s%s/%s" % (enDir, file, enfile)  # 英文文件的路径
# cnPath = "%s%s/%s" % (cnDir, file, enfile.replace("en_w", "cn_w"))  # 中文文件的路径
# targetPath = "%s%s/%s" % (toDir, file, enfile.replace("en_w", "all"))  # 合并文件的路径
# # 加载MP3文件
# song1 = AudioSegment.from_mp3(enPath)
# song2 = AudioSegment.from_mp3(cnPath)
#
# # 取得两个MP3文件的声音分贝
# db1 = song1.dBFS
# db2 = song2.dBFS
#
# song1 = song1[300:]  # 从300ms开始截取英文MP3
#
# # 调整两个MP3的声音大小，防止出现一个声音大一个声音小的情况
# dbplus = db1 - db2
# if dbplus < 0:  # song1的声音更小
#     song1 += abs(dbplus)
# elif dbplus > 0:  # song2的声音更小
#     song2 += abs(dbplus)
#
# # 拼接两个音频文件
# song = song1 + song2
#
# # 导出音频文件
# song.export(targetPath, format="mp3")  # 导出为MP3格式

#encoding=utf-8


f = open("unicode_to_hanyu_pinyin.txt", "r")

py = {}
for l in f:
    l = l.strip()  #行末回车
    key, val = l.split(' ')
    sd = val[1:-1].replace('u:', 'v').split(',')  #去掉左右括号，把u:转为v（驴 lv），然后按逗号分隔
    arr_sd = []
    for i in sd:
        arr_sd.append({'py': i[0:-1], 'tone': i[-1]})  #把每个读音的声调分离出来
    py[int(key, 16)] = arr_sd  #把unicode编码转成10进制作为key

f.close()



def convert(str, encoding = 'utf-8'):
    ret = ''
    for i in str.decode(encoding):
        w = ord(i)
        if py.has_key(w):
            ret += "%s-%s " % (py[w][0]['py'] , py[w][0]['tone'])
        else:
            ret += i
    return ret
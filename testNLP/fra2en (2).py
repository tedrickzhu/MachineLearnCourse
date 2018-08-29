#/usr/bin/env python
#coding=utf8
 
import httplib
import md5
import urllib
import random

import csv





list = ['1','2','3','4']
out = open('output.csv','w')
csv_writer = csv.writer(out,dialect='excel')
csv_writer.writerow(list)


#一是逐行读取：用到列表
#二是根据列名获取对应单元格的值：用到字典
#
#import csv
#
#bid_info = csv.DictReader(open('category_names_sample.csv','r'))
#dict_data = []
#for lines in bid_info:
#        if bid_info.line_num == 1:
#            continue
#        else:
#            dict_data.append(lines)
#row_num = len(dict_data)
#print('this is all the data---' + str(dict))
#
#循环读取每一行
#i = 0
#while(i < row_num):
#    print('this is'+str(i)+'row----'+ str(dict_data[i]))
#    print(dict_data[i])
#    i += 1


#csv_reader = csv.reader(open('category_names_sample.csv'))
#test=0
#for row in csv_reader:
#   if(test>10):
#       break
#   print(row)
#   test = test+1

appid = '20171012000087839'
secretKey = 'jZ5en1Lh3PnRESBsGoTB'

httpClient = None
myurl = '/api/trans/vip/translate'

fromLang = 'fra'
toLang = 'en'
salt = random.randint(32768, 65536)
m1 = md5.new()
csv_reader = csv.reader(open('category_names_sample.csv'))
test=0
q = ''
sign=''
data = []
for line in csv_reader:
   q = line[1]+','+line[2]+','+line[3]
   sign = appid+q+str(salt)+secretKey
   m1.update(sign)
   sign = m1.hexdigest()
   myurl = myurl+'?appid='+appid+'&q='+urllib.quote(q)+'&from='+fromLang+'&to='+toLang+'&salt='+str(salt)+'&sign='+sign
   try:
       httpClient = httplib.HTTPConnection('api.fanyi.baidu.com')
       httpClient.request('GET', myurl)
    
       #response是HTTPResponse对象
       response = httpClient.getresponse()
       print response.read()
   except Exception, e:
       print e
   finally:
       if httpClient:
           httpClient.close()
   
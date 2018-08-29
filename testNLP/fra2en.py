#/usr/bin/env python
#coding=utf8
 
import httplib
import md5
import urllib
import urllib2
import random

import csv

#list = ['1','2','3','4']
#out = open('output.csv','w')
#csv_writer = csv.writer(out,dialect='excel')
#csv_writer.writerow(list)


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

appKey = '1f4000484c1a4276'
secretKey = 'ATWUWzGEQ9dana1AKtmziSxP7b1UEuF9'

httpClient = None
myurl = '/api'

f = open('category_names_sample.csv','r')
#csv_reader = csv.reader(open('category_names_sample.csv'))

q = f.read()
fromLang = 'fr'
toLang = 'EN'
salt = random.randint(1, 65536)
sign = appKey+q+str(salt)+secretKey
m1 = md5.new()
m1.update(sign)
sign = m1.hexdigest()
#myurl = myurl+'?appid='+appid+'&q='+urllib.quote(q)+'&from='+fromLang+'&to='+toLang+'&salt='+str(salt)+'&sign='+sign
myurl = myurl+'?appKey='+appKey+'&q='+urllib.quote(q)+'&from='+fromLang+'&to='+toLang+'&salt='+str(salt)+'&sign='+sign

try:
   httpClient = httplib.HTTPConnection('openapi.youdao.com')
   httpClient.request('GET', myurl)
   
   #response是HTTPResponse对象
   response = httpClient.getresponse()

   #cc = response.read()
   #ccc = cc.decode("unicode_escape")
   #ccc = ccc.encode("utf-8")
   #file_object = open('thefileyoudao.txt','w')
   #file_object.write(ccc)
   #file_object.close()
   print response.read()
except Exception, e:
    print e
finally:
    if httpClient:
        httpClient.close()
f.close()
   
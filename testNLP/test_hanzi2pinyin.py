#!/usr/bin/env python
# -*- coding:utf-8 -*-

__version__ = '0.0'
__all__ = ["PinYin"]

special_char_list = [u'①',u'②',u'③',u'④',u'⑤',u'⑥',u'⑦',u'⑧',u'⑨',u'⑩']

import os.path
import xlwt,re


class PinYin(object):
    def __init__(self, dict_file='word.data'):
        self.word_dict = {}
        self.dict_file = dict_file

    def load_word(self):
        if not os.path.exists(self.dict_file):
            raise IOError("NotFoundFile")

        with file(self.dict_file) as f_obj:
            for f_line in f_obj.readlines():
                try:
                    line = f_line.split('    ')
                    self.word_dict[line[0]] = line[1]
                except:
                    line = f_line.split('   ')
                    self.word_dict[line[0]] = line[1]

    def hanzi2pinyin(self, string=""):
        result = []
        print 'this is the string 000: ',string
        if not isinstance(string, unicode):
            string = string.decode("utf-8")
            print 'change to unicode'
        print 'this is the string : ',string
        for char in string:
            key = '%X' % ord(char)
            print 'this is char :' ,char
            print 'this is key :',key
            print 'this is ord function:',ord(char)
            print str(self.word_dict.get(key, char).lower().split())
            result.append(self.word_dict.get(key, char).split()[0][:-1].lower())

        return result

    def hanzi2pinyin_split(self, string="", split=""):
        result = self.hanzi2pinyin(string=string)
        if split == "":
            return result
        else:
            return split.join(result)

def change_chinese_file(pydict,chinese_file_path):
    rownums = 0
    workbook = xlwt.Workbook(encoding='utf-8')
    sheet1 = workbook.add_sheet('single')
    sheet2 = workbook.add_sheet('multiple')

    with file(chinese_file_path) as f_obj:
        for f_line in f_obj.readlines():
            f_line = f_line.strip()
            if f_line is not '':
                if not isinstance(f_line, unicode):
                    f_line = f_line.decode("utf-8")
                    # print 'change to unicode'
                f_line = f_line.split(' ')
                keyword = f_line[1]
                #从第三个位置，第一个发音的拼音开始取
                keyword_pinyin = None
                # print 'this is line data: ',type(f_line),f_line
                # print 'this is squeen data: ',type(special_char_list),special_char_list
                for i in range(2,len(f_line)):
                    words0 = f_line[i]
                    # print 'each single words0 : ',type(words0),words0
                    #根据拼音前面的编号判断是否是拼音
                    if re.search(r'\d',words0):
                        keyword_pinyin = words0[1:]
                        print 'this is pinyin: ',words0,'  ',keyword_pinyin
                        continue
                    elif keyword_pinyin is not None:
                        words = words0
                        # words = words.split()
                        # print 'each single words0 : ', type(words0), words0
                        # print 'each single words : ', type(words), words
                        words_pinyin = None
                        ismultiwords = False
                        print 'this is type of words:',type(words),words
                        for j in range(len(words)):
                            single_word = words[j]
                            single_word_pinyin = None
                            if single_word == keyword:
                                single_word_pinyin = keyword_pinyin
                            else:

                                print 'this is single word: ',single_word
                                key = '%X' % ord(single_word)
                                single_pinyins = pydict.get(key, single_word).lower().split()
                                if len(single_pinyins)>0 and len(single_pinyins)<2:
                                    single_word_pinyin = single_pinyins[0]
                                else:
                                    ismultiwords = True
                                    single_word_pinyin = single_pinyins[0]
                                    print 'two word is multi pinyin!'
                            if words_pinyin is None:
                                words_pinyin = single_word_pinyin
                            else:
                                words_pinyin = words_pinyin+','+single_word_pinyin
                        # if not ismultiwords:
                        sheet1.write(rownums, 0, keyword)
                        sheet1.write(rownums, 1, words0)
                        sheet1.write(rownums, 2, words_pinyin)
                        rownums+=1
                        # else:
                        #     sheet2.write(rownums, 0, keyword)
                        #     sheet2.write(rownums, 1, words0)
                        #     sheet2.write(rownums, 2, words_pinyin)
                        #     rownums += 1
                    # workbook.save('multipinyinwords.xls')
            else:
                print 'blank line:',f_line

    workbook.save('unusalpinyin.xls')


def write_xls():
    workbook = xlwt.Workbook(encoding='utf-8')
    worksheet = workbook.add_sheet('My Worksheet')
    worksheet.write(0, 0, 'EnglishName')  # 其中的'0-行, 0-列'指定表中的单元，'EnglishName'是向该单元写入的内容
    worksheet.write(1, 0, 'Marcovaldo')
    workbook.save('Excel_Workbook.xls')
    txt1 = '中文名字'
    worksheet.write(0, 1, txt1.decode('utf-8'))  # 此处需要将中文字符串解码成unicode码，否则会报错
    txt2 = '马可瓦多'
    worksheet.write(1, 1, txt2.decode('utf-8'))
    workbook.save('Excel_Workbook.xls')

if __name__ == "__main__":
    # write_xls()
    pinyin = PinYin()
    pinyin.load_word()
    # string = "钓鱼岛是中国的"
    # print "in: %s" % string
    # print "out: %s" % str(pinyindict.hanzi2pinyin(string=string))
    # print "out: %s" % pinyindict.hanzi2pinyin_split(string=string, split="-")
    chinese_file = './unusalword.txt'
    drt_file = './'
    pydict = pinyin.word_dict
    change_chinese_file(pydict,chinese_file)
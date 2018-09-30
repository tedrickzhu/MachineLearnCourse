# coding=utf-8

import configparser as confpr

def get_conf():
    cf = confpr.ConfigParser()
    cf.read("config.ini")
    return cf

print('test')
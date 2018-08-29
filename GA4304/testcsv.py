#encoding=utf-8

import csv
import os
import numpy as np

datapath = './data/'

filenamelist = os.listdir(datapath)
#print(filenamelist)

datamat = np.float64(np.zeros((12,335)))

for i in range(len(filenamelist)):
	filename = filenamelist[i]
	fileformat = filename.split('.')[1]
	fileindex = 0
	if fileformat == 'csv':
		print(filename)
		with open(datapath+filename) as f:
			reader = csv.DictReader(f)
			
			count = 0
			for row in reader:
				datamat[fileindex,count]=np.float64(row['Close'])
				count+=1
				#print(row['Close'])
			#print(datamat[i])
			#print(filename+' has lines :',str(count))
		fileindex+=1

print(datamat.shape)



#filepath = './data/1JD.csv'
#with open(filepath) as f:
#	reader = csv.DictReader(f)
	#for row in reader:
	#	print(row['Close'])
	    #print(type(reader),reader)



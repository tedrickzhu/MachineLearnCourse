#encoding=utf-8

import cv2

def getfacefromvideo(videopath):

	cap = cv2.VideoCapture(videopath)
	index = 0
	while(1):
		# get a frame
		ret, frame = cap.read()
		# show a frame
		cv2.imshow("capture", frame)
	
		cv2.imwrite('/home/zzy/work/zzyface/'+str(index)+'.jpg',frame)
		index +=1
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows() 

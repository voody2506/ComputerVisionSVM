import numpy as np
import cv2
import os

#SVM = cv2.ml.SVM_create()
#SVM.setKernel(cv2.ml.SVM_LINEAR)
#SVM.setP(0.2)
#SVM.setType(cv2.ml.SVM_EPS_SVR)
#SVM.setC(1.0)

'''
    This code has some errors, please, see logistic regression realization
'''

def getTrainingData():

	address = "/Users/voody/Desktop/Vision H:W/HA/pos_new"
	labels = []
	trainingData = []
	for items in os.listdir(address):
	    ## extracts labels
	    name = address + "/" + items
	    img = cv2.imread(name, cv2.IMREAD_COLOR)
	    img = cv2.resize(img, (64,64))
	    d = np.array(img, dtype = np.float32)
	    q = d.flatten()
	    trainingData.append(q)
	    labels.append(items)
	        ######DEBUG######

	        #cv.namedWindow(path,cv.WINDOW_NORMAL)
	        #cv.imshow(path,img)

	return trainingData, labels



svm_params = dict( kernel_type = cv2.ml.SVM_LINEAR,
                    svm_type = cv2.ml.SVM_C_SVC,
                    C=2.67, gamma=3 )

training, labels = getTrainingData()
train = np.asarray(training)
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 100, 1.e-06))
svm.train(train, cv2.ml.ROW_SAMPLE, labels)

svm.save('svm_data.dat')






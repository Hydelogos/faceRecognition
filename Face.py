import os
import cv2
import dlib
import numpy as np
from math import atan2, degrees
from scipy import ndimage
import PIL
from PIL import Image

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
cascade_path='haarcascade_frontalface_default.xml'
cascade = cv2.CascadeClassifier(cascade_path)
detector = dlib.get_frontal_face_detector()

def get_number_face(im):
	rects = detector(im, 1)
	return len(rects)

# #This is using the Dlib Face Detector . Better result more time taking
def get_landmarks(im, i = 0):
	rects = detector(im, 1)
	print(i)
	print(len(rects))
	if len(rects) < 2:
		i = 0
	rect=rects[i]
	fwd=int(rect.width())
	if len(rects) == 0:
		return None,None

	return np.matrix([[p.x, p.y] for p in predictor(im, rects[i]).parts()]),fwd

def annotate_landmarks(im, landmarks, filename, i = 0):
	angle = 0.1
	im = im.copy()
	while angle != 0.0:

		#Start Rotate
		leftEye = np.array(landmarks[0][36:42])
		rightEye = np.array(landmarks[0][42:48])
		x = [p[0] for p in leftEye]
		y = [p[1] for p in leftEye]
		leftCenter = (sum(x) / len(leftEye), sum(y) / len(leftEye))
		x = [p[0] for p in rightEye]
		y = [p[1] for p in rightEye]
		rightCenter = (sum(x) / len(rightEye), sum(y) / len(rightEye))
		angle = degrees(atan2(rightCenter[1]-leftCenter[1], rightCenter[0]-leftCenter[0]))
		im = ndimage.rotate(im, angle)
		angle = 0.0
		landmarks = get_landmarks(im, i)
		#End Rotate

	#Start Resize
	left = np.array(landmarks[0][16])
	right = np.array(landmarks[0][0])
	percent = 250 / (left[0][0] - right[0][0])
	im = Image.fromarray(im)
	height = int((float(im.size[1]) * float(percent)))
	width = int((float(im.size[0]) * float(percent)))
	im = im.resize((width, height), PIL.Image.ANTIALIAS)
	im = np.array(im)
	landmarks = get_landmarks(im, i)

	#Start Crop
	im = Image.fromarray(im)

	center = np.array(landmarks[0][0:17])
	x = [p[0] for p in center]
	y = [p[1] for p in center]
	center = (sum(x) / len(center), sum(y) / len(center))
	im = im.crop((center[0] - 200, center[1] - 250, center[0] + 200, center[1] + 150))

	im = np.array(im)
	landmarks = get_landmarks(im, i)
	#End Crop

	#save file
	cv2.imwrite('img/Ellen/converted/' + filename, im)

	left = np.array(landmarks[0][16])
	right = np.array(landmarks[0][0])
	#End Resize

	for idx, point in enumerate(landmarks[0]):
		point = np.array(point)
		pos = (point[0, 0], point[0, 1])
		cv2.putText(im, str(idx), pos,
				fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
				fontScale=0.4,
				color=(0, 0, 255))
		cv2.circle(im, pos, 3, color=(0, 255, 255))
	return im


directory = "img/Ellen"

for filename in os.listdir(directory):
	if filename.endswith(".jpg"): 
		# print(os.path.join(directory, filename))
		im=cv2.imread(directory + "/" + filename)
		for i in range(get_number_face(im)):
			cv2.imshow('Result',annotate_landmarks(im,get_landmarks(im, i), filename, i))
			im = cv2.imread(filename)

			cv2.waitKey(0)
			cv2.destroyAllWindows()
		continue
	else:
		continue
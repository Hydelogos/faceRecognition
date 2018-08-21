import os
import cv2
import dlib
import numpy as np
import math
from math import atan2, degrees, radians
from scipy import ndimage
import PIL
from PIL import Image
import tensorflow as tf
import facenet



PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
cascade_path='haarcascade_frontalface_default.xml'
cascade = cv2.CascadeClassifier(cascade_path)
detector = dlib.get_frontal_face_detector()
facenet.load_model("20180408-102900.pb")
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]


input_image_size = 160
sess = tf.Session()


def get_number_face(suppliedImg):
	
	img = cv2.imdecode(suppliedImg, flags=cv2.IMREAD_COLOR)
	im = Image.fromarray(img)
	percent = 0.5
	if float(im.size[1]) < float(1000) or float(im.size[0]) < float(1000):
		percent = 0.8
	if float(im.size[1]) > float(1600) or float(im.size[0]) > float(1600):
		percent = 0.3
	height = int((float(im.size[1]) * float(percent)))
	width = int((float(im.size[0]) * float(percent)))
	im = im.resize((width, height), PIL.Image.ANTIALIAS)
	im = np.array(im)
	rects = detector(im, 1)
	return len(rects)

# #This is using the Dlib Face Detector . Better result more time taking
def get_landmarks(im, i = 0):

	im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	rects = detector(im, 1)
	if len(rects) < 2:
		i = 0
	
	if len(rects) == 0:
		return None,None


	rect=rects[i]
	fwd=int(rect.width())

	return np.matrix([[p.x, p.y] for p in predictor(im, rects[i]).parts()]),fwd

def rotate(image, im_rot, xy, angle):
	org_center = (np.array(image.shape[:2][::-1])-1)/2.
	rot_center = (np.array(im_rot.shape[:2][::-1])-1)/2.
	org = xy-org_center
	a = np.deg2rad(angle)
	new = np.array([org[0]*np.cos(a) + org[1]*np.sin(a),-org[0]*np.sin(a) + org[1]*np.cos(a) ])
	return new+rot_center
	#return round(qx), round(qy)

def annotate_landmarks(im, landmarks, filename, i = 0):
	if not landmarks:
		return None
	angle = 0.1
	im = im.copy()

	
	while angle != 0.0:

		#Start Rotate
		im = np.array(im)
		leftEye = np.array(landmarks[0][36:42])
		rightEye = np.array(landmarks[0][42:48])
		x = [p[0] for p in leftEye]
		y = [p[1] for p in leftEye]
		leftCenter = (sum(x) / len(leftEye), sum(y) / len(leftEye))
		x = [p[0] for p in rightEye]
		y = [p[1] for p in rightEye]
		rightCenter = (sum(x) / len(rightEye), sum(y) / len(rightEye))
		angle2 = atan2(rightCenter[1]-leftCenter[1], rightCenter[0]-leftCenter[0])
		angle = degrees(angle2)
		img = im
		im = ndimage.rotate(im, angle, axes=(1, 0), reshape=False)
		newLandmarks = []
		for x in landmarks[0]:
			newLandmarks.append(rotate(img, im, np.array(x)[0], angle))
		im = np.array(im)
		angle = 0.0
		landmarks = np.array(landmarks)
		landmarks[0] = newLandmarks
		#End Rotate

	#Start Crop

	#0 is left point
	#16 is right point
	#18 is top point
	#8 is bottom point
	im = Image.fromarray(im)
	landmarks[0] = newLandmarks
	center = np.array(landmarks[0][0:19])
	im = im.crop((center[0][0] - 1, center[18][1] - 5, center[16][0] + 1, center[8][1] + 1))

	im = np.array(im)
	#End Crop

	#Whiten image for the neural network
	im = facenet.prewhiten(im)
	'''cv2.imshow('image',im)
	cv2.waitKey(0)'''
	im = cv2.resize(im, (input_image_size, input_image_size), interpolation=cv2.INTER_CUBIC)
	return getEmbedding(im)



def getEmbedding(resized):
    reshaped = resized.reshape(-1,input_image_size,input_image_size,3)
    feed_dict = {images_placeholder: reshaped, phase_train_placeholder: False}
    # print(feed_dict)
    embedding = sess.run(embeddings, feed_dict=feed_dict)
    return embedding



def getNpy(name, suppliedImg):

	name = name.replace("\\", "")
	name = name.replace("/", "")
	name = name.replace(".", "")
	
	img = cv2.imdecode(suppliedImg, flags=cv2.IMREAD_COLOR)
	im = Image.fromarray(img)
	percent = 0.5
	if float(im.size[1]) < float(1000) or float(im.size[0]) < float(1000):
		percent = 0.8
	if float(im.size[1]) > float(1600) or float(im.size[0]) > float(1600):
		percent = 0.3
	height = int((float(im.size[1]) * float(percent)))
	width = int((float(im.size[0]) * float(percent)))
	im = im.resize((width, height), PIL.Image.ANTIALIAS)
	im = np.array(im)
	return annotate_landmarks(im,get_landmarks(im, 0), name, 0)


def addAutorise(name):
	print(name)


def test(emb1, emb2):
	distance = np.sqrt(np.sum(np.square(np.subtract(emb1, emb2))))
	threshold = 0.7    # set yourself to meet your requirement
	print(distance)
	if distance <= threshold:
		return distance
	else:
		return False

def checkIfExist(name):
	name = name.replace("\\", "")
	name = name.replace("/", "")
	name = name.replace(".", "")
	my_file = Path(name + ".npy")
	if my_file.exists():
		emb1 = np.load(my_file)
		for filename in os.listdir("./autorise/"):
			if filename.endswith(".npy"):
				emb2 = np.load("./autorise/" + filename)
				threshold = 1.1    # set yourself to meet your requirement
				print("distance = "+str(distance))
				if distance <= threshold:
					return distance
		return False

	else:
		("Erreur le nom n existe pas.")
		return False

import os
import cv2
import dlib
import numpy as np
from math import atan2, degrees
from scipy import ndimage
import PIL
from PIL import Image
import tensorflow as tf
import facenet
from align import detect_face

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
cascade_path='haarcascade_frontalface_default.xml'
cascade = cv2.CascadeClassifier(cascade_path)
detector = dlib.get_frontal_face_detector()


input_image_size = 160
sess = tf.Session()


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
	im = im.crop((center[0] - 150, center[1] - 200, center[0] + 150, center[1] + 100))

	im = np.array(im)
	landmarks = get_landmarks(im, i)
	#End Crop

	#Whiten image for the neural network
	im = facenet.prewhiten(im)
	im = cv2.resize(im, (input_image_size, input_image_size), interpolation=cv2.INTER_CUBIC)
	np.save(filename + ".npy", getEmbedding(im))
	return getEmbedding(im)
	#save file
	im.save('img/Ellen/converted/' + filename)
	im = np.array(im)

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



def getEmbedding(resized):
    reshaped = resized.reshape(-1,input_image_size,input_image_size,3)
    feed_dict = {images_placeholder: reshaped, phase_train_placeholder: False}
    # print(feed_dict)
    embedding = sess.run(embeddings, feed_dict=feed_dict)
    return embedding

directory = "img/Ellen"


facenet.load_model("20180408-102900.pb")
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]

'''for filename in os.listdir(directory):
	if filename.endswith(".jpg"): 
		# print(os.path.join(directory, filename))
		im=cv2.imread(directory + "/" + filename)
		if get_number_face(im) > 0:
			for i in range(get_number_face(im)):
				cv2.imshow('Result',annotate_landmarks(im,get_landmarks(im, i), filename, i))
				im = cv2.imread(filename)

				cv2.waitKey(0)
				cv2.destroyAllWindows()
		continue
	else:
		continue
'''
cam = cv2.VideoCapture(0)   # 0 -> index of camera
s, img = cam.read()
if s:    # frame captured without any errors
    cv2.imwrite("webcam.jpg",img) #save image
im1 = annotate_landmarks(img,get_landmarks(img, 0), "webcam", 0)

s2, img2 = cam.read()
if s2:    # frame captured without any errors
    cv2.imwrite("webcam2.jpg",img2) #save image

img2 = cv2.imread("img/Ellen/ellen.jpg")
im2 = annotate_landmarks(img2,get_landmarks(img2, 0), "ellen", 0)

print(im1)
print("SEPARATEUR")
print(im2)




distance = np.sqrt(np.sum(np.square(np.subtract(im1, im2))))

threshold = 1.1    # set yourself to meet your requirement
print("distance = "+str(distance))
print("Result = " + ("Meme personne" if distance <= threshold else "Pas la meme personne"))
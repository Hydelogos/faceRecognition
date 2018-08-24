# -*- coding: utf-8 -*-
import numpy as np
import os
import Face

from io import BytesIO
import base64
from PIL import Image, ImageDraw

import psycopg2

from flask import Flask, request, jsonify, render_template, abort
from flask_socketio import SocketIO, emit

import dlib
import cv2

user = input("Postgres User Name:")
password = input("Postgres Password:")
domain = input("Postgres Domain Url:")
port = input("Postgres Port:")
try:
	DATABASE_URL = os.environ['DATABASE_URL']
except:
	DATABASE_URL = "postgresql://" + user + ":" + password + "@" + domain + ":" + port

conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor()
conn.commit()
cur.execute("select exists(select * from information_schema.tables where table_name=%s)", ('face',))
if not cur.fetchone()[0]:
	cur.execute("CREATE TABLE info (id serial PRIMARY KEY, nom varchar, prenom varchar, autorise boolean);")
	cur.execute("CREATE TABLE face (id serial PRIMARY KEY, name varchar UNIQUE, data decimal[], info_id integer, FOREIGN KEY (info_id) REFERENCES info (id));")
	
	conn.commit()
	cur.close()
else:
	cur.close()

tracker = dlib.correlation_tracker()
started = False
mask = Image.new('RGBA', (500, 500), (255, 0, 0, 0))
mask.save("static/mask.png", "PNG")
previous = [0, 0]
grabbing = False

app = Flask(__name__)
socketio = SocketIO(app)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/", methods=['POST'])
def save():
	if request.method == 'POST':
		print("Start")
		if 'name' not in request.form:
			print("name introuvable")
		name = request.form['name']
		print("Recuperation du nom!")
		print(name)
		if 'file' not in request.files:
			print("File not found")
		f = request.files['file'].read()
		image = np.fromstring(f, np.uint8)
		npydata = Face.getNpy(name, image)
		if npydata is None:
			return "Erreur, pas de visage detecté!"
		print(npydata)
		cur = conn.cursor()
		cur.execute("SELECT * FROM face WHERE name = %s", (name,))
		if cur.fetchone() is None:
			cur.execute("INSERT INTO face (name, data) VALUES (%s, %s);", (name, npydata.tolist()))
			conn.commit()
		else:
			cur.execute("UPDATE face SET data = (%s) WHERE name = (%s);", (npydata.tolist(),name,))
			conn.commit()
		cur.close()
		return jsonify(npydata.tolist())

@app.route("/test")
def test():
	return render_template("test.html")

@app.route("/test", methods=["POST"])
def testPost():
	if request.method == 'POST':
		print("Start")
		if 'file' not in request.files:
			print("File not found")
		f = request.files['file'].read()
		image = np.fromstring(f, np.uint8)
		npydata = Face.getNpy("", image)
		cur = conn.cursor()
		cur.execute("SELECT * FROM face;")
		myFaces = cur.fetchall()
		nom = "personne de connu."
		lowest = 2
		for myFace in myFaces:
			result = Face.test(np.asarray(myFace[2], dtype=float), npydata)
			if result is not False:
				
				if lowest > result:
					lowest = result
					nom = myFace[1]
		return "Ce visage appartient à " + nom
@app.route("/webcam")
def getCam():
	return render_template("cam.html")

@app.route("/webcam", methods=["POST"])
def postCam():
	if request.method == 'POST':
		if 'image' not in request.files:
			abort(400)
		f = request.files['image'].read()
		img = np.fromstring(f, np.uint8)
		if Face.get_number_face(img) > 0:
			npydata = Face.getNpy("", img)
			cur = conn.cursor()
			cur.execute("SELECT * FROM face;")
			myFaces = cur.fetchall()
			nom = "inconnu"
			lowest = 2
			for myFace in myFaces:
				result = Face.test(np.asarray(myFace[2], dtype=float), npydata)
				if result is not False:
					
					if lowest > result:
						lowest = result
						nom = myFace[1]
			return jsonify({'result': 1, 'nom': nom})
		else:
			return jsonify({'result': 0})

@app.route("/tracker")
def getTrack():
	return render_template("track.html")


@socketio.on('connect', namespace='/track')
def connect():
	emit('okay', {})
	global started
	started = False

@socketio.on('stream', namespace='/track')
def track(message):
	global started
	f = Image.open(BytesIO(message['data']))
	img = cv2.cvtColor(np.array(f), cv2.COLOR_RGB2BGR)
	if not started:
		detectedBox = test(img)
		if detectedBox is not None:
			tracker.start_track(img, detectedBox)
			(x,y,xb,yb) = [detectedBox.left(),detectedBox.top(),detectedBox.right(),detectedBox.bottom()]
			emit('response', {'data': (x, y, xb, yb)})
			started = True
	else:
		tracker.update(img)
		detectedBox = tracker.get_position()
		(x,y,xb,yb) = [detectedBox.left(),detectedBox.top(),detectedBox.right(),detectedBox.bottom()]
		emit('response', {'data': (x, y, xb, yb)})

@app.route("/draw")
def getDraw():
	return render_template("draw.html")


@socketio.on('connect', namespace='/drawing')
def connectDraw():
	emit('okay', {})
	global started
	started = False

@socketio.on('stream', namespace='/drawing')
def draw(message):
	global started
	global mask
	global previous
	f = Image.open(BytesIO(message['data']))
	img = cv2.cvtColor(np.array(f), cv2.COLOR_RGB2BGR)
	if not started:
		detectedBox = test(img)
		if detectedBox is not None:
			tracker.start_track(img, detectedBox)
			(x,y,xb,yb) = [detectedBox.left(),detectedBox.top(),detectedBox.right(),detectedBox.bottom()]
			width = xb - x
			height = yb - y
			print((x,y,xb,yb))
			previous[0] = x + round(width/2)
			previous[1] = y + round(width/2)
			emit('response', {})
			started = True
	else:
		tracker.update(img)
		detectedBox = tracker.get_position()
		(x,y,xb,yb) = [detectedBox.left(),detectedBox.top(),detectedBox.right(),detectedBox.bottom()]
		width = xb - x
		height = yb - y
		draw = ImageDraw.Draw(mask)
		draw.line((previous[0], previous[1], x + round(width/2), y + round(height/2)), fill=(255, 0, 0), width=5)
		previous[0] = x + round(width/2)
		previous[1] = y + round(height/2)
		mask.save("static/mask.png", "PNG")
		print("SAVED!")
		emit('response', {})



@app.route("/drag")
def getDrag():
	return render_template("dragdrop.html")


@socketio.on('connect', namespace='/drag')
def connectDrag():
	emit('okay', {})
	global started
	started = False

@socketio.on('stream', namespace='/drag')
def drag(message):
	global started
	global grabbing
	f = Image.open(BytesIO(message['data']))
	img = cv2.cvtColor(np.array(f), cv2.COLOR_RGB2BGR)
	if not started:
		detectedBox = test(img)
		if detectedBox is not None:
			print("DETECTED")
			tracker.start_track(img, detectedBox)
			(x,y,xb,yb) = [detectedBox.left(),detectedBox.top(),detectedBox.right(),detectedBox.bottom()]
			width = xb - x
			height = yb - y
			emit('response', {"data": [x + round(width/2), y + round(width/2)]})
			started = True
	else:
		
		if testDrag(img) is not None:
			grabbing = True
			emit("grab", {})
			detectedBox = testDrag(img)
			tracker.start_track(img, detectedBox)
			(x,y,xb,yb) = [detectedBox.left(),detectedBox.top(),detectedBox.right(),detectedBox.bottom()]
			width = xb - x
			height = yb - y
			emit('response', {"data": [x + round(width/2), y + round(width/2)]})
		elif test(img) is not None:
			grabbing = False
			emit("release", {})
			detectedBox = test(img)
			tracker.start_track(img, test(img))
			(x,y,xb,yb) = [detectedBox.left(),detectedBox.top(),detectedBox.right(),detectedBox.bottom()]
			width = xb - x
			height = yb - y
			emit('response', {"data": [x + round(width/2), y + round(width/2)]})
		else:
			tracker.update(img)
			print("UPDATED!")
			detectedBox = tracker.get_position()
			(x,y,xb,yb) = [detectedBox.left(),detectedBox.top(),detectedBox.right(),detectedBox.bottom()]
			width = xb - x
			height = yb - y
			emit('response', {"data": [x + round(width/2), y + round(width/2)]})

def test(file):
	detector = dlib.simple_object_detector("detector/cup")
	detectedBoxes = detector(file)
	for detectedBox in detectedBoxes:
		print(detectedBox)
		return detectedBox
	return None

def testDrag(file):
	detector = dlib.simple_object_detector("detector/grab")
	detectedBoxes = detector(file)
	for detectedBox in detectedBoxes:
		print(detectedBox)
		return detectedBox
	return None

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
# -*- coding: utf-8 -*-
import numpy as np
import os
import Face

from io import BytesIO
import base64
from PIL import Image

import psycopg2

from flask import Flask, request, jsonify, render_template, abort


try:
	DATABASE_URL = os.environ['DATABASE_URL']
except:
	DATABASE_URL = "postgresql://postgres:obscure@localhost:5433"

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


app = Flask(__name__)
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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
from flask import Flask, request, render_template, redirect, url_for, flash, session
import os
import cv2
import numpy as np
from datetime import date, datetime
import joblib
from sklearn.neighbors import KNeighborsClassifier
from gtts import gTTS
from io import BytesIO
import pygame
import smtplib
from email.mime.text import MIMEText
from cryptography.fernet import Fernet
import logging
from pymongo import MongoClient
import base64

#### Defining Flask App
app = Flask(__name__)
app.secret_key = os.urandom(24)

#### Logging Setup
logging.basicConfig(filename='app.log', level=logging.INFO)

#### Initialize encryption (optional)
key = Fernet.generate_key()
cipher_suite = Fernet(key)

#### MongoDB Setup
client = MongoClient('mongodb://localhost:27017/')
db = client.attendance_db
users_collection = db.users
attendance_collection = db.attendance

#### Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

#### Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#### Utility Functions
def totalreg():
    return users_collection.count_documents({})

def extract_faces(img):
    try:
        if img.shape != (0, 0, 0):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_points = face_detector.detectMultiScale(gray, 1.3, 5)
            return face_points
        else:
            return []
    except Exception as e:
        logging.error(f"Error in extract_faces: {e}")
        return []

def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

def train_model():
    faces = []
    labels = []
    userlist = users_collection.find()
    for user in userlist:
        for face_data in user['faces']:
            img_data = base64.b64decode(face_data)
            img_array = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(f"{user['name']}_{user['id']}")
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

def extract_attendance():
    attendance_records = attendance_collection.find({"date": datetoday})
    names, rolls, times = [], [], []
    for record in attendance_records:
        names.append(record['name'])
        rolls.append(record['roll'])
        times.append(record['time'])
    l = len(names)
    return names, rolls, times, l

def add_attendance(name):
    username, userid = name.split('_')
    current_time = datetime.now().strftime("%H:%M:%S")
    
    if not attendance_collection.find_one({"roll": int(userid), "date": datetoday}):
        attendance_collection.insert_one({"name": username, "roll": int(userid), "time": current_time, "date": datetoday})
        return True
    return False

def play_voice_message(message):
    tts = gTTS(text=message, lang='en')
    fp = BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)

    pygame.mixer.init()
    pygame.mixer.music.load(fp)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

def send_email(subject, body, to_email):
    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = 'sabarish.it22@bitsathy.ac.in'
        msg['To'] = to_email

        with smtplib.SMTP('smtp.example.com', 587) as server:  # Update SMTP settings
            server.starttls()
            server.login('your_email@example.com', 'your_password')  # Update this
            server.send_message(msg)
    except Exception as e:
        logging.error(f"Error in sending email: {e}")

def getallusers():
    userlist = users_collection.find()
    names, rolls = [], []
    for user in userlist:
        names.append(user['name'])
        rolls.append(user['id'])
    l = len(names)
    return userlist, names, rolls, l

#### Route Functions
@app.route('/')
def home():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    
    names, rolls, times, l = extract_attendance()    
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)  

@app.route('/start', methods=['GET'])
def start():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', totalreg=totalreg(), datetoday2=datetoday2, mess='There is no trained model in the static folder. Please add a new face to continue.')

    ret = True
    cap = cv2.VideoCapture(0)
    while ret:
        ret, frame = cap.read()
        if len(extract_faces(frame)) > 0:
            (x, y, w, h) = extract_faces(frame)[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            if add_attendance(identified_person):
                play_voice_message('Attendance marked successfully.')
                send_email('Attendance Update', f'User {identified_person} marked present at {datetime.now()}', 'recipient@example.com')
            cv2.putText(frame, f'{identified_person}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()    
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2) 

@app.route('/add', methods=['GET', 'POST'])
def add():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        newusername = request.form['newusername']
        newuserid = request.form['newuserid']
        user_faces = []
        i, j = 0, 0
        cap = cv2.VideoCapture(0)
        while 1:
            _, frame = cap.read()
            faces = extract_faces(frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                if j % 10 == 0:
                    face = frame[y:y+h, x:x+w]
                    resized_face = cv2.resize(face, (50, 50))
                    encoded_face = base64.b64encode(cv2.imencode('.jpg', resized_face)[1]).decode()
                    user_faces.append(encoded_face)
                    i += 1
                j += 1
            cv2.imshow('Adding new User', frame)
            if cv2.waitKey(1) == 27 or i == 20:
                break
        cap.release()
        cv2.destroyAllWindows()
        users_collection.insert_one({"name": newusername, "id": int(newuserid), "faces": user_faces})
        
        logging.info('Adding User and Training Model')
        train_model()
        return redirect(url_for('home'))
    return render_template('add.html')

@app.route('/edit_user/<username>', methods=['GET', 'POST'])
def edit_user(username):
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        new_username = request.form['new_username']
        new_userid = request.form['new_userid']
        
        users_collection.update_one({'name': username}, {'$set': {'name': new_username, 'id': int(new_userid)}})
        
        logging.info('Updating User and Training Model')
        train_model()
        
        return redirect(url_for('home'))
    
    user = users_collection.find_one({'name': username.split('_')[0], 'id': int(username.split('_')[1])})
    if not user:
        return 'User not found!', 404
    
    return render_template('edit_user.html', user=user)

@app.route('/delete_user/<username>', methods=['POST'])
def delete_user(username):
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    
    users_collection.delete_one({'name': username.split('_')[0], 'id': int(username.split('_')[1])})
    logging.info('Deleting User and Training Model')
    train_model()
    return redirect(url_for('home'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'admin123':
            session['logged_in'] = True
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password')
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)

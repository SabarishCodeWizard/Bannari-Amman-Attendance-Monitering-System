from flask import Flask, request, render_template, redirect, url_for, flash, session
import os
import cv2
import numpy as np
from datetime import date, datetime
import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier
from gtts import gTTS
from io import BytesIO
import pygame
import smtplib
from email.mime.text import MIMEText
from cryptography.fernet import Fernet
import logging

#### Defining Flask App
app = Flask(__name__)
app.secret_key = os.urandom(24)

#### Logging Setup
logging.basicConfig(filename='app.log', level=logging.INFO)

#### Initialize encryption (optional)
key = Fernet.generate_key()
cipher_suite = Fernet(key)

#### Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

#### Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#### If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')

#### Utility Functions
def totalreg():
    return len(os.listdir('static/faces'))

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
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l

def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')
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
        msg['Subject'] = "Opening the Attendance System"
        msg['From'] = 'sabarish.it22@bitsathy.ac.in'
        msg['To'] = 'ravik60656@gmail.com'

        with smtplib.SMTP('smtp.example.com', 587) as server:  # Update SMTP settings
            server.starttls()
            server.login('your_email@example.com', 'your_password')  # Update this
            server.send_message(msg)
    except Exception as e:
        logging.error(f"Error in sending email: {e}")

def getallusers():
    userlist = os.listdir('static/faces')
    names = []
    rolls = []
    l = len(userlist)

    for i in userlist:
        name, roll = i.split('_')
        names.append(name)
        rolls.append(roll)
    
    return userlist, names, rolls, l

def deletefolder(duser):
    pics = os.listdir(duser)
    
    for i in pics:
        os.remove(duser+'/'+i)

    os.rmdir(duser)

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
        userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
        if not os.path.isdir(userimagefolder):
            os.makedirs(userimagefolder)
        i, j = 0, 0
        cap = cv2.VideoCapture(0)
        while 1:
            _, frame = cap.read()
            faces = extract_faces(frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
                cv2.putText(frame, f'Images Captured: {i}/50', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
                if j % 10 == 0:
                    name = newusername+'_'+str(i)+'.jpg'
                    cv2.imwrite(userimagefolder+'/'+name, frame[y:y+h, x:x+w])
                    i += 1
                j += 1
            if j == 500:
                break
            cv2.imshow('Adding new User', frame)
            if cv2.waitKey(1) == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
        logging.info('Training Model')
        train_model()
        names, rolls, times, l = extract_attendance()    
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2) 

@app.route('/edit_user/<username>', methods=['GET', 'POST'])
def edit_user(username):
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        newname = request.form['newname']
        newroll = request.form['newroll']
        old_path = f'static/faces/{username}'
        new_path = f'static/faces/{newname}_{newroll}'
        os.rename(old_path, new_path)
        return redirect(url_for('home'))
    return render_template('edit_user.html', username=username)

@app.route('/delete_user/<username>', methods=['POST'])
def delete_user(username):
    try:
        user_folder = f'static/faces/{username}'
        if os.path.isdir(user_folder):
            deletefolder(user_folder)
            train_model()
            flash(f'User {username} deleted successfully.', 'success')
        else:
            flash(f'User {username} not found.', 'error')
    except Exception as e:
        logging.error(f"Error deleting user {username}: {e}")
        flash(f'Error deleting user {username}.', 'error')
    return redirect(url_for('home'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'bitsathy' and password == '1234':  
            session['logged_in'] = True
            return redirect(url_for('home'))
        else:
            flash('Invalid credentials', 'error')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)

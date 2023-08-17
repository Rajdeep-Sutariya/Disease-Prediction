from flask import Flask, render_template, flash, request, url_for, redirect, session
from passlib.hash import sha256_crypt
from functools import wraps
import gc, os
from wtforms import Form, TextField, PasswordField
from flask_sqlalchemy import SQLAlchemy
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import csv
from flask_admin import Admin
from flask_admin.contrib.sqla import ModelView
from datetime import datetime
import smtplib
from random import choice

app = Flask(__name__)
db = SQLAlchemy(app)


app.config['SECRET_KEY'] = os.urandom(24)
conn = 'sqlite:///'+ os.path.abspath(os.getcwd())+"/DataBases/test.db"   
#conn1 = 'sqlite:///'+ os.path.abspath(os.getcwd())+"/DataBases/doctor.db"   

admin = Admin(app,name='Admin')

#user
def connect_to_db(app, database):
    
    app.config['SQLALCHEMY_DATABASE_URI'] = database
    app.config['SQLALCHEMY_ECHO'] = True
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
    db.app = app
    db.init_app(app)

connect_to_db(app,conn)
EMAIL_ADDRESS = "diseaseprediction12@gmail.com"
EMAIL_PASSWORD = "dis@pred"
app.config['SECRET_KEY'] = os.urandom(24)

class User(db.Model):
    """ This is the user """
    __tablename__ = "users"
    id = db.Column(db.Integer,  autoincrement=True, primary_key=True)
    username = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    password = db.Column(db.String(120), nullable=False)
    registeron = db.Column(db.DateTime, nullable=False ,default=datetime.now)
    phno = db.Column(db.Integer, nullable=False)    
    addr = db.Column(db.String(250), nullable=False)
    gender = db.Column(db.String(20), nullable=False)
#    date_posted = db.Column(db.DateTime, nullable=False, default=datetime.now)
    
    
admin.add_view(ModelView(User, db.session))

def __repr__(self):
    return 'User' + str(self.id)

class Doctor(db.Model):
    """ This is the doctor """
    
    __tablename__ = "doctors"
    id = db.Column(db.Integer,  autoincrement=True, primary_key=True)
    doctname = db.Column(db.String(80), nullable=False)
    doctdegree = db.Column(db.String(100), nullable=False)
    doctdisexp = db.Column(db.String(100), nullable=False)
    docthospital = db.Column(db.String(100), nullable=False)
    doctphno = db.Column(db.Integer, nullable=False)    
    doctlocation = db.Column(db.String(250), nullable=False)
    date_posted = db.Column(db.DateTime, nullable=False, default=datetime.now)    

admin.add_view(ModelView(Doctor, db.session))


def __repr__(self):
    return 'Doctor' + str(self.id)    
 
class Laboratory(db.Model):
    """ This is the doctor """
    
    __tablename__ = "laboratory"
    id = db.Column(db.Integer,  autoincrement=True, primary_key=True)
    labname = db.Column(db.String(80), nullable=False)
    labphno = db.Column(db.Integer, nullable=False)    
    lablocation = db.Column(db.String(250), nullable=False)   
    date_posted = db.Column(db.DateTime, nullable=False, default=datetime.now)

admin.add_view(ModelView(Laboratory, db.session))

def __repr__(self):
    return 'Doctor' + str(self.id)    
 
class Feedback(db.Model):
    """ This contains feedbacks """

    __tablename__ = "feedbacks"

    id = db.Column(db.Integer, autoincrement=True, primary_key=True)
    feedback_name = db.Column(db.String(80), nullable=False)
    feedback_email = db.Column(db.String(120), nullable=False)
    feedback_subject = db.Column(db.String(80), nullable=False)
    feedback_message = db.Column(db.String(200), nullable=False)
    feedback_datetime = db.Column(db.DateTime, nullable=False ,default=datetime.now)
    
def __repr__(self):
    return '<Feedback {}>'.format(self.id)


class RegistrationForm(Form):
	username = TextField('Username')
	email = TextField('Email')
	password = PasswordField('Password')
	confirm = PasswordField('Confirm Password')


class LoginForm(Form):
	username = TextField('Username')
	password = PasswordField('Password')



def login_required(f):
	@wraps(f)
	def wrap(*args, **kwargs):
		if 'logged_in' in session:
			return f(*args,**kwargs)
		else:
			flash('You need to login first!', "warning")
			return redirect(url_for('login_page'))
	return wrap

def already_logged_in(f):
	@wraps(f)
	def wrap(*args, **kwargs):
		if 'logged_in' in session:
			flash("You are already logged in!", "success")
			return redirect(url_for('dashboard'))
		else:
			return f(*args, **kwargs)
	return wrap


@app.route('/logout/')
@login_required
def logout():
	flash("You have been logged out!", "success")
	session.clear()
	gc.collect()
	return redirect(url_for('main'))

def verify(_username, _password):
	if User.query.filter_by(username=_username).first() is None:
		flash("No such user found with this username", "warning")
		return False
	if not sha256_crypt.verify(_password, User.query.filter_by(username=_username).first().password):
		flash("Invalid Credentials, password isn't correct!", "danger")
		return False
	return True

@app.route('/', methods=['GET','POST'])
def main():
	return render_template('main.html')


'''@app.route('/', methods=['GET'])
def dropdown():
        return render_template('includes/default.html', symptoms=symptoms)'''
    
with open('templates/Testing.csv', newline='') as f:
        reader = csv.reader(f)
        symptoms = next(reader)
        symptoms = symptoms[:len(symptoms)-1]
        
    
@app.route('/dashboard/',methods=['GET'])
@login_required
def dashboard():
               return render_template('dashboard.html', symptoms=symptoms)
           
@app.route('/about/')
def about():
               return render_template('about.html')

@app.route('/contactus/')
def contactus():
               return render_template('contact.html')
           
@app.route('/disease_predict', methods=['POST'])
def disease_predict():
    selected_symptoms = []
    if(request.form['Symptom1']!="") and (request.form['Symptom1'] not in selected_symptoms):
        selected_symptoms.append(request.form['Symptom1'])
    if(request.form['Symptom2']!="") and (request.form['Symptom2'] not in selected_symptoms):
        selected_symptoms.append(request.form['Symptom2'])
    if(request.form['Symptom3']!="") and (request.form['Symptom3'] not in selected_symptoms):
        selected_symptoms.append(request.form['Symptom3'])
    if(request.form['Symptom4']!="") and (request.form['Symptom4'] not in selected_symptoms):
        selected_symptoms.append(request.form['Symptom4'])
    if(request.form['Symptom5']!="") and (request.form['Symptom5'] not in selected_symptoms):
        selected_symptoms.append(request.form['Symptom5'])

    disease = dosomething(selected_symptoms)
    return render_template('disease_predict.html',disease=disease,symptoms=symptoms)

 
@app.route('/find_doctor/',methods=['GET','POST'])
# @login_required
def find_doctor():
    all_doctor = Doctor.query.all()
    return render_template('find_doctor.html',doctors=all_doctor)

@app.route('/find_lab/',methods=['GET'])
# @login_required
def find_lab():
    all_lab = Laboratory.query.all()
    return render_template('find_lab.html',laboratory = all_lab)

@app.route('/drug', methods=['POST'])
def drugs():
    medicine = request.form['medicine']
    return render_template('homepage.html',medicine=medicine,symptoms=symptoms)  
    
@app.route('/login/', methods=['GET','POST'])
@already_logged_in
def login_page():
	try:
        
		form = LoginForm(request.form)            
		if request.method == 'POST':
            
			_username = request.form['username']
			_password = request.form['password']
            
            
			if verify(_username, _password) is False:
				return render_template('login.html', form=form)
			session['logged_in'] = True
			session['username'] = _username
			gc.collect()
			return redirect(url_for('dashboard'))
		
		return render_template('login.html', form=form)
    
        
	except Exception as e:
		return render_template('error.html',e=e)

@app.route('/register/', methods=['GET','POST'])
def register_page():
	try:
		form = RegistrationForm(request.form)
		if request.method == 'POST' and form.validate():
			_username = request.form['username']
			_email = request.form['email']
			_password = sha256_crypt.encrypt(str(form.password.data))
			user = User(username = _username, email = _email, password = _password)
			db.create_all()
			if User.query.filter_by(username=_username).first() is not None:
				flash('User Already registered with username {}'.format(User.query.filter_by(username=_username).first().username), "warning")
				return render_template('register.html', form=form)
			if User.query.filter_by(email=_email).first() is not None:
				flash('Email is already registered with us {}'.format(User.query.filter_by(email=_email).first().username), "warning")
				return render_template('register.html', form=form)
			flash("Thank you for registering!", "success")
			db.session.add(user)
			db.session.commit()
			db.session.close()
			gc.collect()
			session['logged_in'] = True
			session['username'] = _username
			session.modified = True
			return redirect(url_for('dashboard'))
		return render_template('register.html', form=form)
	except Exception as e:
		return render_template('error.html',e=e)

@app.route('/forget_password/', methods=['GET', 'POST'])
def forget_password():
    _email = None
    try:
        if request.method=="POST":
            if request.form['submit'] == "Send Email":
                _email = request.form['email']
                if User.query.filter_by(email=_email).first() is None:
                    flash('Email is not registered with us', "danger")
                    _email = None
                else:
                    session['username'] = User.query.filter_by(email=_email).first().username
                    with smtplib.SMTP('smtp.gmail.com', 587) as smtp:
                        smtp.ehlo()
                        smtp.starttls()
                        smtp.ehlo()
                        smtp.login(EMAIL_ADDRESS,EMAIL_PASSWORD)
                        secret_key = choice(("556257", "787637", "768686", "278672", "879745", "876876", "168373", "365262", "876721", "218982"))
                        session['otp'] = secret_key
                        session.modified = True
                        subject ='DIESEASE PREDICTION Forget Password!! !'
                        body = "Your One Time password: {} \n Valid till half an hour from the generation of the OTP.".format(secret_key)
                        msg=f'subject: {subject}\n\n{body}'
                        smtp.sendmail(EMAIL_ADDRESS, _email, msg)
                        flash("Mail Sent!", "success")
                return render_template('forget_password.html')
            if request.form['submit'] == "Verify OTP":
                otp = request.form['otp']				
                if 'username' in session:
                    if otp == session['otp']:
                        session['logged_in'] = True
                        return redirect(url_for('reset_password'))
                    else:
                        flash("OTP is incorrect. Try again!", "warning")
                        return render_template('forget_password.html')
                else:
                    flash("First enter email!")
                    return render_template('forget_password.html')
        else:
            return render_template('forget_password.html')
    except Exception as e:
            return render_template('error.html', e=e)
  

@app.route('/database/', methods=['GET','POST'])
@login_required
def database():
    try:
        flash("Welcome!")
        return render_template('database.html',data= User.query.all())
    except Exception as e:
        return render_template('error.html',e=e)		


@app.errorhandler(500)
@app.errorhandler(404)
def page_not_found(e):
	return render_template('error.html', e=e)

iris = load_iris()

X = pd.DataFrame(iris.data, columns=iris.feature_names)
#print(X)
y = pd.Categorical.from_codes(iris.target, iris.target_names)
#print(y)

#print(X.head())

y = pd.get_dummies(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)

acc = accuracy_score(y_test,y_pred)*100

print(acc)


data = pd.read_csv(os.path.join("templates", "Training.csv"))
df = pd.DataFrame(data)
cols = df.columns
cols = cols[:-1]
x = df[cols]
y = df['prognosis']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

print ("DecisionTree")
dt = DecisionTreeClassifier()
clf_dt=dt.fit(x_train,y_train)
#print ("Acurracy: ", clf_dt.score(x_test,y_test))

# with open('templates/Testing.csv', newline='') as f:
#         reader = csv.reader(f)
#         symptoms = next(reader)
#         symptoms = symptoms[:len(symptoms)-1]

indices = [i for i in range(132)]
symptoms = df.columns.values[:-1]

dictionary = dict(zip(symptoms,indices))

def dosomething(symptom):
    user_input_symptoms = symptom
    user_input_label = [0 for i in range(132)]
    for i in user_input_symptoms:
        idx = dictionary[i] -1
        user_input_label[idx] = 1

    user_input_label = np.array(user_input_label)
    user_input_label = user_input_label.reshape((-1,1)).transpose()
    return(dt.predict(user_input_label))

@app.route('/Adashboard')
def Adashboard():
    flash("HII, Welcome to Disease Prediction")
    return render_template('Admin/Adashboard.html')
    
#USER CURD
@app.route('/user')
def user():
    return render_template('Admin/user.html')

@app.route('/Usertables', methods=['GET','POST'])
def Usertables():
    all_user = User.query.all()
    return render_template('Admin/Usertables.html',users=all_user)

@app.route('/Usertables/Userdelete/<int:id>')
def Userdelete(id):
    user = User.query.get_or_404(id)
    db.session.delete(user)
    db.session.commit()
    flash("Data is Deleted")
    return redirect('/Usertables')

@app.route('/Usertables/Useredit/<int:id>' , methods = ['GET', 'POST'])
def Useredit(id):
    user = User.query.get_or_404(id)
    if request.method == 'POST' :
        user.username = request.form['username']
        user.email = request.form['email']
        user.password = sha256_crypt.encrypt(str(request.form['password']))
        db.session.commit()
        flash("Data is Updated")
        return redirect('/Usertables')
    else :
        return render_template('Admin/Useredit.html', user=user)
    
@app.route('/Usertables/Useradd/', methods=['GET','POST'])
def Useradd():
    try:
        form = RegistrationForm(request.form)
        if request.method == 'POST' and form.validate():
            _username = request.form['username']
            _email = request.form['email']
            _password = sha256_crypt.encrypt(str(form.password.data))
            user = User(username = _username, email = _email, password = _password)
            db.create_all()
            if User.query.filter_by(username=_username).first() is not None:
                flash('User Already registered with username {}'.format(User.query.filter_by(username=_username).first().username), "warning")
                return redirect(url_for('Usertables'))
            if User.query.filter_by(email=_email).first() is not None:
                flash('Email is already registered with us {}'.format(User.query.filter_by(email=_email).first().username), "warning")
                return redirect(url_for('Usertables'))
            db.session.add(user)
            db.session.commit()
            db.session.close()
            gc.collect()
            session['logged_in'] = True
            session['username'] = _username
            session.modified = True
            flash("User is Added",_username)
            return redirect(url_for('Usertables'))
        return render_template('Admin/Useradd.html', form=form)
    except Exception as e:
        return render_template('error.html',e=e)  
    
@app.route('/edit_profile', methods=['GET','POST'])
@login_required
def edit_profile():
    try:
        user = User.query.filter_by(username=session['username'])
        return render_template('edit_profile.html',user=user)
    except Exception as e:
        return render_template('error.html',e=e)  

@app.route("/insertProfile/<int:id>",methods=["POST","GET"])
def inserProfile(id):
    if request.method=='POST':
        user=User()
        user.id=id
        user.email=request.form['email']
#        user.password=sha256_crypt.encrypt(str(request.form['password']))
        user.phno=request.form['phno']
        user.addr=request.form['addr']
        user.gender=request.form['gender']
        db.session.merge(user)
        db.session.commit()
        flash("Profile is Updated")
    
        return redirect(url_for("edit_profile"))
    else:
        return render_template("edit_profile.html", user=user)
#Doctor CURD
        
@app.route('/reset_password', methods=['GET', 'POST'])
@login_required
def reset_password():
    username = session['username']
    try:
        user_password = User.query.filter_by(username=username).first().password
        if request.method == 'POST':
            username = session['username']
            if request.form['submit'] == "Save":
                new_password = request.form['password']
                User.query.filter_by(username = username).first().password = sha256_crypt.encrypt(str(new_password))
                db.session.commit()
                gc.collect()
                flash("Data is Updated")
                return render_template('/reset_password.html',user_password = user_password)
            else :
                return render_template('/reset_password.html',user_password = user_password)
        else :
            return render_template('/reset_password.html')
    except Exception as e:
        return render_template('error.html',e=e)  
    
        
@app.route('/Doctortables', methods=['GET','POST'])
def Doctortables():
    all_doctor = Doctor.query.all()
    return render_template('Admin/Doctortables.html',doctors=all_doctor)

@app.route('/Doctortables/Doctoradd', methods = ['GET', 'POST'])
def Doctoradd():
    
    if request.method == 'POST' :
        doct_name = request.form['doctname']
        doct_degree = request.form['doctdegree']
        doct_disexp = request.form['doctdisexp']
        doct_hospital = request.form['docthospital']
        doct_phno = request.form['doctphno']
        doct_location = request.form['doctlocation']
        new_doctor = Doctor(doctname=doct_name, doctdegree=doct_degree, doctdisexp= doct_disexp, docthospital=doct_hospital, doctphno= doct_phno, doctlocation= doct_location )
        db.create_all()
        if Doctor.query.filter_by(doctname=doct_name).first() is not None:
            flash('Doctor Already registered with Name {}'.format(Doctor.query.filter_by(doctname=doct_name).first().doctname), "warning")
            return redirect(url_for('Usertables'))
        if Doctor.query.filter_by(doctphno= doct_phno).first() is not None:
            flash('Phone number is already registered with us {}'.format(Doctor.query.filter_by(doctphno= doct_phno).first().doctname), "warning")
            return redirect(url_for('Usertables'))
        db.session.add(new_doctor)
        db.session.commit()
        return redirect('/Doctortables')
    else:
        all_docts = Doctor.query.order_by(Doctor.date_posted).all()
    return render_template('Admin/Doctoradd.html', doctors = all_docts)

@app.route('/Doctortables/Doctordelete/<int:id>')
def Doctordelete(id):
    doctor= Doctor.query.get_or_404(id)
    db.session.delete(doctor)
    db.session.commit()
    return redirect('/Doctortables')

@app.route('/Doctortables/Doctoredit/<int:id>' , methods = ['GET', 'POST'])
def Doctoredit(id):
    doctor = Doctor.query.get_or_404(id)
    if request.method == 'POST' :
        doctor.doctname = request.form['doctname']
        doctor.doctdegree = request.form['doctdegree']
        doctor.docthospital = request.form['docthospital']
        doctor.doctphno = request.form['doctphno']
        doctor.doctlocation = request.form['doctlocation']
        db.session.commit()
        return redirect('/Doctortables')
    else :
        return render_template('Admin/Doctoredit.html', doctor=doctor)

#Laboratory CURD
        
@app.route('/Labtables', methods=['GET','POST'])
def Labtables():
    all_lab = Laboratory.query.all()
    return render_template('Admin/Labtables.html',laboratory=all_lab)

@app.route('/Labtables/Labadd', methods = ['GET', 'POST'])
def Labadd():
    if request.method == 'POST' :
        lab_name = request.form['labname']
        lab_phno = request.form['labphno']
        lab_location = request.form['lablocation']
        db.create_all()
        new_lab = Laboratory(labname=lab_name, labphno= lab_phno, lablocation= lab_location )
        if Laboratory.query.filter_by(labname=lab_name).first() is not None:
            flash('Laboratory Already registered with Name {}'.format(Laboratory.query.filter_by(labname=lab_name).first().labname), "warning")
            return redirect(url_for('Usertables'))
        if Laboratory.query.filter_by(labphno= lab_phno).first() is not None:
            flash('Phone number is already registered with us {}'.format(Laboratory.query.filter_by(labphno= lab_phno).first().labname), "warning")
            return redirect(url_for('Usertables'))
        db.session.add(new_lab)
        db.session.commit()
        return redirect('/Labadd')
    else:
        all_lab = Laboratory.query.order_by(Laboratory.date_posted).all()
    return render_template('Admin/Labadd.html', laboratory = all_lab)

@app.route('/Labtables/Labdelete/<int:id>')
def Labdelete(id):
    lababoratory= Laboratory.query.get_or_404(id)
    db.session.delete(lababoratory)
    db.session.commit()
    return redirect('/Labtables')

@app.route('/Labtables/Labedit/<int:id>' , methods = ['GET', 'POST'])
def Labedit(id):
    laboratory = Laboratory.query.get_or_404(id)
    if request.method == 'POST' :
        laboratory.labname = request.form['labname']
        laboratory.labphno = request.form['labphno']
        laboratory.lablocation = request.form['lablocation']
        db.session.commit()
        return redirect('/Labtables')
    else :
        return render_template('Admin/Labedit.html', laboratory=laboratory)

@app.route('/maps')
def maps():
    return render_template('Admin/maps.html')

@app.route('/notifications')
def notifications():
    return render_template('Admin/notifications.html')
    
@app.route('/adminlogin', methods=['GET','POST'])
def adminlogin():
    if request.method == 'POST':
        if request.form['username']!= 'Adminraj' or request.form['password']!= 'Adminpass':
            flash("Invalid Credentials, Please try again","warning")
        else:
            return redirect(url_for('Adashboard'))
    return render_template('Admin/adminlogin.html')

@app.route('/feedback_page/', methods=['GET','POST'])
def feedback_page():
    try:
        if request.method == 'POST':
            _feedbackname = request.form['feedback_name']
            _feedbackemail = request.form['feedback_email']
            _feedbacksub = request.form['feedback_subject']
            _feedbackmsg = request.form['feedback_message']
            feedback = Feedback(feedback_name = _feedbackname, feedback_email = _feedbackemail, feedback_subject = _feedbacksub, feedback_message =  _feedbackmsg)
            db.create_all()
            db.session.add(feedback)
            db.session.commit()
            db.session.close()
            gc.collect()
            return render_template('contact.html')
    except Exception as e:
        return render_template('error.html',e=e)
    
@app.route('/feedback_table')
def feedback_table():
    all_feedback = Feedback.query.all()
    return render_template('Admin/feedback_table.html', feedbacks=all_feedback)

#@app.route('/feedback_table/fdelete/<int:id>')
#def fdelete(id):
#    feedbacks = Feedback.query.get_or_404(id)
#    db.session.delete(feedbacks)
#    db.session.commit()
#    flash("Data is Deleted")
#    return redirect('/feedback_table')
    
@app.route('/feedback_table/fdelete/<int:id>')
def fdelete(id):
    feedback = Feedback.query.get_or_404(id)
    db.session.delete(feedback)
    db.session.commit()
    flash("Data is Deleted")
    return redirect('/feedback_table') 

if __name__ == "__main__":
	db.create_all()
	app.run()

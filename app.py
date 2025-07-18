from flask import Flask, render_template, request, jsonify, session, url_for, redirect, flash
from joblib import load
import mysql.connector
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import timedelta, datetime
import os
from werkzeug.utils import secure_filename
import pickle
from sklearn.preprocessing import LabelEncoder

from flask import Flask, render_template, request, redirect, url_for
import os

app = Flask(__name__, template_folder='templates')

app.secret_key = os.urandom(24)

UPLOAD_FOLDER = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'static', 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ======================
# Manually define class labels for encoders
# ======================
protocol_types = ["icmp", "tcp", "udp"]
service_types = [
    "aol", "auth", "bgp", "courier", "csnet_ns", "ctf", "daytime", "discard", "domain", "domain_u",
    "echo", "eco_i", "ecr_i", "efs", "exec", "finger", "ftp", "ftp_data", "gopher", "harvest",
    "hostnames", "http", "http_2784", "http_443", "http_8001", "imap4", "IRC", "iso_tsap", "klogin",
    "kshell", "ldap", "link", "login", "mtp", "name", "netbios_dgm", "netbios_ns", "netbios_ssn",
    "netstat", "nnsp", "nntp", "ntp_u", "other", "pm_dump", "pop_2", "pop_3", "printer", "private",
    "red_i", "remote_job", "rje", "shell", "smtp", "sql_net", "ssh", "sunrpc", "supdup", "systat",
    "telnet", "tftp_u", "tim_i", "time", "urh_i", "urp_i", "uucp", "uucp_path", "vmnet", "whois", "X11",
    "Z39_50"
]
flag_types = [
    "OTH", "REJ", "RSTO", "RSTOS0", "RSTR", "S0", "S1", "S2", "S3", "SF", "SH"
]

# ======================
# Fit encoders in memory
# ======================
model = load('models/xgboost_best_model.pkl')
scaler = load('models/scaler.pkl')
service_encoder = load('models/service_encoder.pkl')
label_encoder = load('models/label_encoder.pkl')
flag_encoder = load('models/flag_encoder.pkl')
protocol_type_encoder = load('models/protocol_type_encoder.pkl')

feature_columns = [
    'src_bytes', 'dst_bytes', 'same_srv_rate', 'flag', 'dst_host_srv_count',
    'serror_rate', 'srv_serror_rate', 'service', 'dst_host_serror_rate',
    'dst_host_same_srv_rate', 'count', 'dst_host_rerror_rate', 'rerror_rate',
    'logged_in', 'protocol_type', 'srv_count', 'diff_srv_rate', 'srv_rerror_rate',
    'dst_host_diff_srv_rate', 'dst_host_srv_serror_rate'
]
feature_mapping = {name: idx for idx, name in enumerate(feature_columns)}


def get_db_connection():
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='',
        database='intrusion_detection_system'
    )


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    # Check if user is logged in
    if not session.get('user_id'):
        return render_template(
            "predict.html",
            # protocol_type_options=protocol_type_encoder.classes_, # Removed encoder options
            # service_options=service_encoder.classes_, # Removed encoder options
            # flag_options=flag_encoder.classes_, # Removed encoder options
            feature_mapping=feature_mapping
        )

    # Pass encoder options for dropdowns
    # protocol_type_options = protocol_type_encoder.classes_ # Removed encoder options
    # service_options = service_encoder.classes_ # Removed encoder options
    # flag_options = flag_encoder.classes_ # Removed encoder options

    conn = None
    cursor = None
    try:
        # Get database connection
        conn = get_db_connection()
        cursor = conn.cursor()

        if request.method == 'POST':
            # Get the input values from the form
            form_data = request.form.to_dict()

            print("\n=== Processing New Prediction ===")
            print(f"Input data: {form_data}")

            # Initialize a zero array for 20 features
            features = np.zeros(20)

            # Use LabelEncoders and scaler for categorical and numerical features
            try:
                protocol = form_data['protocol_type'].strip().lower()
                features[feature_mapping['protocol_type']] = protocol_type_encoder.transform([protocol])[0]
            except Exception as e:
                return render_template("predict.html", error=f"Invalid protocol_type: {form_data.get('protocol_type')}", feature_mapping=feature_mapping)

            try:
                service = form_data['service'].strip().lower()
                features[feature_mapping['service']] = service_encoder.transform([service])[0]
            except Exception as e:
                return render_template("predict.html", error=f"Invalid service: {form_data.get('service')}", feature_mapping=feature_mapping)

            try:
                flag = form_data['flag'].strip().upper()
                features[feature_mapping['flag']] = flag_encoder.transform([flag])[0]
            except Exception as e:
                return render_template("predict.html", error=f"Invalid flag: {form_data.get('flag')}", feature_mapping=feature_mapping)

            # Robustly handle logged_in field
            if 'logged_in' in form_data:
                logged_in_val = form_data['logged_in']
                if str(logged_in_val).lower() in ['no', '0', 'false']:
                    features[feature_mapping['logged_in']] = 0
                else:
                    features[feature_mapping['logged_in']] = 1

            # Process numerical features
            for field, index in feature_mapping.items():
                if field not in ['protocol_type', 'service', 'flag', 'logged_in']:
                    value = form_data.get(field, '0')
                    try:
                        features[index] = float(value)
                    except ValueError:
                        return render_template("predict.html",
                                               error=f"Invalid value for {field}. Please enter a valid number.",
                                               feature_mapping=feature_mapping)

            # Scale features
            features_scaled = scaler.transform([features])[0]
            print("Features (scaled):", features_scaled)

            # Debug: Print features and model output
            print("Features sent to model:", features_scaled)
            try:
                prediction = model.predict([features_scaled])[0]
                probabilities = model.predict_proba([features_scaled])[0]
                print("Model raw prediction:", prediction)
                print("Model probabilities:", probabilities)
            except Exception as e:
                print("Model prediction error:", e)
                return render_template("predict.html", error="Model prediction error: {}".format(e), feature_mapping=feature_mapping)

            confidence = float(max(probabilities))
            prediction_str = label_encoder.inverse_transform([prediction])[0] if hasattr(label_encoder, 'inverse_transform') else ('normal' if prediction == 0 else 'anomaly')

            print(f"\nPrediction: {prediction_str}")
            print(f"Confidence: {confidence}")

            # Save to database
            cursor.execute(
                "INSERT INTO detections (prediction, confidence, user_id) VALUES (%s, %s, %s)",
                (prediction_str, confidence, session['user_id'])
            )
            conn.commit()

            # Store prediction results in session for the result page
            session['last_prediction'] = {
                'prediction': prediction_str,
                'confidence': f"{confidence:.2%}"
            }

            return redirect(url_for('result'))

        return render_template(
            "predict.html",
            # protocol_type_options=protocol_type_options, # Removed encoder options
            # service_options=service_options, # Removed encoder options
            # flag_options=flag_options, # Removed encoder options
            feature_mapping=feature_mapping
        )
    except Exception as e:
        print(f"Database error in predict: {str(e)}")
        return render_template("predict.html", error="An error occurred while processing your request.",
                               feature_mapping=feature_mapping)
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


@app.route('/result')
def result():
    if not session.get('user_id'):
        return redirect(url_for('login'))
    prediction = session.get('last_prediction', {})
    return render_template('result.html', prediction=prediction)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()

        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['role'] = user['role']
            return redirect(url_for('index'))
        return render_template('login.html', error="Invalid credentials")

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/statistics')
def statistics():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    # Count normal and anomaly predictions
    cursor.execute("SELECT prediction, COUNT(*) as count FROM detections GROUP BY prediction")
    rows = cursor.fetchall()
    stats = {'normal': 0, 'anomaly': 0}
    total = 0
    for row in rows:
        if row['prediction'] in stats:
            stats[row['prediction']] = row['count']
            total += row['count']
    # Count total users
    cursor.execute("SELECT COUNT(*) as user_count FROM users")
    user_count = cursor.fetchone()['user_count']
    # Per-user detection statistics
    cursor.execute("""
        SELECT u.username, COUNT(d.id) as count
        FROM users u
        LEFT JOIN detections d ON u.id = d.user_id
        GROUP BY u.id, u.username
    """)
    user_rows = cursor.fetchall()
    user_stats = []
    for row in user_rows:
        percentage = (row['count'] / total * 100) if total > 0 else 0
        user_stats.append({
            'username': row['username'],
            'count': row['count'],
            'percentage': f"{percentage:.2f}%"
        })
    is_admin = session.get('role') == 'admin'
    cursor.close()
    conn.close()
    return render_template(
        'statistics.html',
        stats=stats,
        total=total,
        is_admin=is_admin,
        user_stats=user_stats,
        user_count=user_count
    )


@app.route('/admin/users', methods=['GET', 'POST'])
def admin_users():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    # Query all users and their detection counts
    cursor.execute('''
        SELECT u.id, u.username, u.email, u.role, u.full_name, u.profile_image, u.created_at, COUNT(d.id) as detections
        FROM users u
        LEFT JOIN detections d ON u.id = d.user_id
        GROUP BY u.id, u.username, u.email, u.role, u.full_name, u.profile_image, u.created_at
        ORDER BY u.created_at DESC
    ''')
    users = []
    for row in cursor.fetchall():
        users.append({
            'id': row['id'],
            'username': row['username'],
            'email': row['email'],
            'role': row['role'],
            'full_name': row['full_name'],
            'profile_image': row['profile_image'],
            'detections': row['detections'],
            'joined': row['created_at'].strftime('%Y-%m-%d') if row['created_at'] else '',
        })
    cursor.close()
    conn.close()
    return render_template('admin_users.html', users=users)


@app.route('/team')
def team():
    return render_template('team.html')


@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    message = None
    if request.method == 'POST':
        # Handle profile update (full_name, email, password, profile image, etc.)
        full_name = request.form.get('full_name')
        email = request.form.get('email')
        new_password = request.form.get('new_password')
        profile_image = None
        if 'profile_image' in request.files:
            file = request.files['profile_image']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                import time
                unique_filename = f"{session['user_id']}_{int(time.time())}_{filename}"
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], unique_filename))
                profile_image = unique_filename
                cursor.execute('UPDATE users SET profile_image=%s WHERE id=%s', (profile_image, session['user_id']))
        # Update full_name and email
        cursor.execute('UPDATE users SET full_name=%s, email=%s WHERE id=%s', (full_name, email, session['user_id']))
        # Update password if provided
        if new_password:
            hashed_pw = generate_password_hash(new_password)
            cursor.execute('UPDATE users SET password=%s WHERE id=%s', (hashed_pw, session['user_id']))
        conn.commit()
        message = 'Profile updated successfully.'
    cursor.execute('SELECT * FROM users WHERE id = %s', (session['user_id'],))
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    return render_template('profile.html', user=user, message=message)


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    # You can add your signup logic here
    return render_template('signup.html')


@app.route('/admin')
def admin():
    return render_template('admin.html')


if __name__ == '__main__':
    app.run(debug=True)

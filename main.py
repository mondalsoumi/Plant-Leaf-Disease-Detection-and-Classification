import os
from flask import Flask, request, render_template, jsonify, send_file, redirect, url_for, session, flash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import numpy as np
import csv
from datetime import datetime
import pandas as pd
import json
from collections import Counter
import matplotlib.pyplot as plt
import io
import base64
import secrets
from auth import register_user, verify_user, login_required

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PREDICTION_LOG'] = 'prediction_logs.csv'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['SECRET_KEY'] = secrets.token_hex(16)  # Generate a random secret key
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Create prediction log file if it doesn't exist
if not os.path.exists(app.config['PREDICTION_LOG']):
    with open(app.config['PREDICTION_LOG'], 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'filename', 'predicted_class', 'confidence', 'file_path', 'user_id'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def log_prediction(filename, predicted_class, confidence, file_path):
    """Log prediction details to CSV file"""
    with open(app.config['PREDICTION_LOG'], 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            filename,
            predicted_class,
            confidence,
            file_path,
            session.get('user_id', 'anonymous')
        ])

def get_prediction_stats():
    """Get statistics from prediction logs"""
    if not os.path.exists(app.config['PREDICTION_LOG']):
        return {
            'total_predictions': 0,
            'class_distribution': {},
            'average_confidence': 0,
            'recent_predictions': []
        }
    
    df = pd.read_csv(app.config['PREDICTION_LOG'])
    class_dist = df['predicted_class'].value_counts().to_dict()
    avg_confidence = df['confidence'].mean()
    recent_preds = df.tail(5).to_dict('records')
    
    return {
        'total_predictions': len(df),
        'class_distribution': class_dist,
        'average_confidence': round(avg_confidence, 2),
        'recent_predictions': recent_preds
    }

def generate_confidence_plot():
    """Generate a plot of confidence scores distribution"""
    if not os.path.exists(app.config['PREDICTION_LOG']):
        return None
    
    df = pd.read_csv(app.config['PREDICTION_LOG'])
    plt.figure(figsize=(10, 6))
    plt.hist(df['confidence'], bins=20, edgecolor='black')
    plt.title('Distribution of Prediction Confidence Scores')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    
    # Save plot to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def compare_diseases():
    """Compare different diseases based on prediction logs"""
    if not os.path.exists(app.config['PREDICTION_LOG']):
        return {
            'error': 'No prediction data available',
            'disease_comparison': {},
            'monthly_trends': {}
        }
    
    try:
        df = pd.read_csv(app.config['PREDICTION_LOG'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Disease comparison
        disease_stats = df.groupby('predicted_class').agg({
            'confidence': ['count', 'mean', 'std']
        }).round(2)
        
        disease_comparison = {
            disease: {
                'count': stats['count'],
                'avg_confidence': stats['mean'],
                'std_confidence': stats['std']
            }
            for disease, stats in disease_stats.iterrows()
        }
        
        # Monthly trends
        df['month'] = df['timestamp'].dt.to_period('M')
        monthly_trends = df.groupby(['month', 'predicted_class']).size().unstack(fill_value=0)
        monthly_trends = monthly_trends.to_dict()
        
        return {
            'disease_comparison': disease_comparison,
            'monthly_trends': monthly_trends
        }
    except Exception as e:
        return {
            'error': str(e),
            'disease_comparison': {},
            'monthly_trends': {}
        }

def generate_disease_comparison_plots():
    """Generate plots for disease comparison"""
    if not os.path.exists(app.config['PREDICTION_LOG']):
        return None
    
    try:
        df = pd.read_csv(app.config['PREDICTION_LOG'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        
        # Plot 1: Disease Distribution
        disease_counts = df['predicted_class'].value_counts()
        disease_counts.plot(kind='bar', ax=ax1)
        ax1.set_title('Distribution of Detected Diseases')
        ax1.set_xlabel('Disease Type')
        ax1.set_ylabel('Number of Cases')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Plot 2: Monthly Trends
        df['month'] = df['timestamp'].dt.to_period('M')
        monthly_trends = df.groupby(['month', 'predicted_class']).size().unstack(fill_value=0)
        monthly_trends.plot(kind='line', marker='o', ax=ax2)
        ax2.set_title('Monthly Disease Trends')
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Number of Cases')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error generating plots: {str(e)}")
        return None

# Load model
model = load_model('leaf_disease_model.h5')  # or .keras if you have that
CLASS_NAMES = [
    'Bacterial Spot', 
    'Early Blight', 
    'Healthy', 
    'Late Blight', 
    'Leaf Mold',
    'Septoria Leaf Spot',
    'Spider Mites',
    'Target Spot',
    'Tomato Yellow Leaf Curl Virus',
    'Two-Spotted Spider Mite'
]

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        success, result = verify_user(username, password)
        
        if success:
            session['user_id'] = result['id']
            session['username'] = result['username']
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash(result, 'danger')
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return render_template('signup.html')
        
        success, message = register_user(username, password, email)
        
        if success:
            flash(message, 'success')
            return redirect(url_for('login'))
        else:
            flash(message, 'danger')
    
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

@app.route('/stats')
@login_required
def get_stats():
    """Get prediction statistics"""
    stats = get_prediction_stats()
    return jsonify(stats)

@app.route('/confidence-plot')
@login_required
def get_confidence_plot():
    """Get confidence score distribution plot"""
    plot_data = generate_confidence_plot()
    if plot_data:
        return jsonify({'plot': plot_data})
    return jsonify({'error': 'No prediction data available'}), 404

@app.route('/download-logs')
@login_required
def download_logs():
    """Download prediction logs as CSV"""
    if not os.path.exists(app.config['PREDICTION_LOG']):
        return jsonify({'error': 'No logs available'}), 404
    
    return send_file(
        app.config['PREDICTION_LOG'],
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'prediction_logs_{datetime.now().strftime("%Y%m%d")}.csv'
    )

@app.route('/compare-diseases')
@login_required
def get_disease_comparison():
    """Get disease comparison statistics"""
    comparison_data = compare_diseases()
    return jsonify(comparison_data)

@app.route('/disease-comparison-plots')
@login_required
def get_disease_comparison_plots():
    """Get disease comparison visualization plots"""
    plot_data = generate_disease_comparison_plots()
    if plot_data:
        return jsonify({'plot': plot_data})
    return jsonify({'error': 'Could not generate comparison plots'}), 404

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed types: png, jpg, jpeg'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = round(np.max(predictions) * 100, 2)

        result = CLASS_NAMES[predicted_class]
        
        # Log the prediction
        log_prediction(filename, result, confidence, filepath)
        
        return jsonify({
            'result': result,
            'confidence': f"{confidence}%",
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

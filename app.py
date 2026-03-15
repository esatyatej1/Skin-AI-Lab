import os
import io
import base64
import numpy as np
import tensorflow as tf
import threading
import subprocess
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)

# Path to the generator model
MODEL_PATH = 'c:/123/dcgan_generator_epoch50.keras'

# Global variable for the model
generator = None
# Global variable for training status
training_process = None
training_output = []
current_progress = 0

def load_generator():
    global generator
    if generator is None:
        if os.path.exists(MODEL_PATH):
            print(f"Loading model from {MODEL_PATH}...")
            generator = tf.keras.models.load_model(MODEL_PATH)
            print("Model loaded successfully.")
        else:
            print(f"Error: Model not found at {MODEL_PATH}")
            return False
    return True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    global training_process, training_output
    data = request.get_json() or {}
    mode = data.get('mode', 'normal')

    if training_process and training_process.poll() is None:
        return jsonify({'status': 'running', 'message': f'Training ({mode}) is already in progress.'})
    
    training_output = ["Starting training process..."]
    
    def run_command():
        global training_process
        # We trigger the .bat file or direct wsl command
        # Using WSL command directly to capture output better
        cmd = ["wsl", "-d", "Ubuntu-22.04", "-u", "root", "bash", "/mnt/c/123/setup.sh", mode]
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            training_process = process
            
            for line in iter(process.stdout.readline, ""):
                line_text = line.strip()
                if "PROGRESS:" in line_text:
                    try:
                        # Extract 10/100 -> 10%
                        parts = line_text.split("PROGRESS:")[1].strip().split("/")
                        global current_progress
                        current_progress = int((int(parts[0]) / int(parts[1])) * 100)
                    except:
                        pass
                
                training_output.append(line_text)
                # Keep only last 100 lines for memory
                if len(training_output) > 100:
                    training_output.pop(0)
            
            process.stdout.close()
            process.wait()
            training_output.append(f"Process finished with exit code {process.returncode}")
        except Exception as e:
            training_output.append(f"Error: {str(e)}")

    thread = threading.Thread(target=run_command)
    thread.start()
    
    return jsonify({'status': 'started', 'message': 'Training started in background.'})

@app.route('/train/status', methods=['GET'])
def train_status():
    global training_process
    is_running = training_process and training_process.poll() is None
    return jsonify({
        'running': is_running,
        'progress': current_progress,
        'logs': training_output[-20:] # Return last 20 lines
    })

@app.route('/train/stop', methods=['POST'])
def stop_train():
    global training_process, training_output
    if training_process and training_process.poll() is None:
        try:
            # Terminate the process and its children
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(training_process.pid)], capture_output=True)
            training_output.append("--- Training stopped by user ---")
            return jsonify({'message': 'Training stopped.'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return jsonify({'message': 'No training process is running.'})

@app.route('/generate', methods=['GET'])
def generate():
    if not load_generator():
        return jsonify({'error': 'Model not found'}), 500

    # Generate random noise
    noise = tf.random.normal([1, 100])
    
    # Generate image
    generated_image = generator(noise, training=False)
    
    # Post-process image: [-1, 1] -> [0, 255]
    generated_image = (generated_image[0].numpy() + 1) / 2.0
    generated_image = (generated_image * 255).astype(np.uint8)
    
    # If grayscale (1 channel), convert to 3 channels for display consistency if needed
    # (Though DCGAN usually outputs 3 channels in this training script)
    
    # Convert to PIL Image
    img = Image.fromarray(generated_image)
    
    # Save to buffer
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return jsonify({'image': img_str})

if __name__ == '__main__':
    # Ensure the model exists before starting
    load_generator()
    # Create templates folder if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    app.run(debug=True, port=5000)

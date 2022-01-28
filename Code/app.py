from MathModel import imageToExcersise_FromModel
from Solver import evaluate_expression
from Precedence import precedence
import Calculation_module
import Errors
import os
import urllib.request
import socketio as socketio
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def upload_form():
    return render_template('upload.html')


@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(UPLOAD_FOLDER, filename))
        # print('upload_image filename: ' + filename)
        #return render_template('upload.html', filename=filename)
        str_exercise = imageToExcersise_FromModel(os.path.join(UPLOAD_FOLDER, filename))
        try:
            x = evaluate_expression(str_exercise)
            if x is not None:
                print("Result:", x)
        except KeyboardInterrupt:
            print(Errors.Click)

        except EOFError:
            print(Errors.Invalid_char)
        return render_template('predict.html', image_file_name = file.filename, label = str_exercise, accuracy =str(x))
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


if __name__ == "__main__":
    app.debug = True
    app.run(debug=True)
    app.debug = True
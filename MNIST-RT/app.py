from flask import Flask, request, flash, redirect, url_for, send_from_directory, render_template
from datetime import timedelta
from werkzeug.utils import secure_filename
import mnist1
import cassandrad
import socket

import os
#set folder as
UPLOAD_FOLDER = './'
#set pictures requirement
extensions = set(['png', 'jpg'])
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in extensions

@app.route('/', methods = ['GET', 'POST'])
def upload_file():
     #no file
    if file.filename == '':
        flash('empty')
        return redirect(request.url)
     #post one
    if request.method == 'POST':
        if 'file' not in request.files:
            return ({"error": "check your file type" })            #if the file type is not allowed
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            predict = mnist1.do_predict(filename)              # predict with the mnist1
            cassandrad.write_cassandra(filename, predict)      #generate the reult
            return ('The predicted result is: ' + str(predict) + '\n')
    return render_template('upload.html') #put them in the same folder

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
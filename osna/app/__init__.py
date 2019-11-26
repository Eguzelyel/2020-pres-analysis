from flask import Flask
import os
from .. import osna_path
app = Flask(__name__)
app.config['SECRET_KEY'] = 'you-will-never-guess'  # for CSRF

from . import routes
### Getting Started with your Github Repository.

You should have been assigned a project repository for your work. In the examples below, we'll assume the repository is called <https://github.com/iit-cs579/sample-project>. This is where all your code will live. 

Your repository has been setup with a lot of starter code so you can get started more easily. To use it, follow the instructions below. We'll assume your project name is 'project1'.

1. Clone your repo:  `git clone https://github.com/iit-cs579/project1`
2. Start a [virtual environment](https://virtualenv.pypa.io/en/stable/).
  - First, make sure you have virtual env installed. `pip install virtualenv`
  - Next, outside of the team repository, create a new virtual environment folder by `virtualenv osna-virtual`. 
  - Activate your virtual environment by `source osna-virtual/bin/activate`
  - Now, when you install python software, it will be saved in your `osna-virtual` folder, so it won't conflict with the rest of your system.
3. Install your project code by
```
cd project1   # enter your project repository folder
python setup.py develop # install the code. 
```

This may take a while, as all dependencies listed in the `requirements.txt` file will also be installed.

**Windows users**: if you're having troubles, try reading [this](http://timmyreilly.azurewebsites.net/python-flask-windows-development-environment-setup/). It looks like you will need to:
- install `pip install virtualenvwrapper-win`
- instead of `virtualenv osna-virtual` above, do `mkvirtualenv osna-virtual`
- other students have also had luck starting environments with the command `py -3 -m venv env env\scripts\activate`

4. If everything worked properly, you should now be able to run your project's command-line tool by typing:  
```
osna --help
```
which should print
```
Usage: osna [OPTIONS] COMMAND [ARGS]...

  Console script for osna.

Options:
  --help  Show this message and exit.

Commands:
  collect   Collect data and store in given directory.
  evaluate  Report accuracy and other metrics of your approach.
  network   Perform the network analysis component of your project.
  stats     Read all data and print statistics.
  train     Train a classifier on all of your labeled data and save it for...
  web       Launch a web app for your project demo.
```

As part of your project, you'll implement each of the above 6 commands. How you do so will in large part be up to you.

### Configuration
The app will stores configuration files in  `~/.osna/` (on windows, this is `C:\Users\<username>`.) See `__init__.py` for an example of how you can add variables to this file and read them for use in your app.


### Flask Web UI

Your tool currently has one command called `web`. This launches a simple web server which we will use to make a demo of your project. You can launch it with:  
`osna web`  
which should produce output like:
```
* Serving Flask app "osna.app" (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: on
 * Running on http://0.0.0.0:9999/ (Press CTRL+C to quit)
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 606-066-960
```

If you open your web browser and go to `http://0.0.0.0:9999/` you should see something like:

![img/web.png](img/web.png)


### Tutorials

Here are a few tutorials that will help you with the project:

1. Complete the scikit-learn tutorial from <https://www.datacamp.com/community/tutorials/machine-learning-python> (2 hours)
2. Understand how python packages work by going through the [Python Packaging User Guide](https://packaging.python.org/tutorials/) (you can skip the "Creating Documentation" section). (1 hour)
3. Complete Part 1 of the [Flask tutorial](https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world), which is the library we will use for making a web demo for your project.


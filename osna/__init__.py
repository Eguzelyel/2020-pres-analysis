# -*- coding: utf-8 -*-

"""Top-level package for elevate-osna."""

__author__ = """Salem Abuammer, Ekrem Guzelyel, Muhammad Shareef"""
__email__ = 'sabuammer@hawk.iit.edu, eguzelyel@hawk.iit.edu, msharee3@hawk.iit.edu'
__version__ = '0.1.0'

# -*- coding: utf-8 -*-
import os
import json


def write_default_credentials(path):
    cred_file = open(path, 'wt')
    creds = {
        "Twitter": {
            "consumer_key": "Az4GvKnMDywiT5CinddLkil23",
            "consumer_secret": "h0domk7JfpRSmJISP3Q2DtsmUq1x2f4yDGsJCg99BiRRyjhCPV",
            "access_token": "1163693072028487680-DuPz1APMFz2Kel9Ep5pdJO1W2rTZ3p",
            "access_token_secret": "EK8CH4FTcr3eeyVs846VGq8mX8zowoY8lxNF4G0y9hMJ3"
        },
        "IBM-Cloud": {
            "api_key": "SnwmF6iLlOmdBXaYm8CiHewut3DF5JuMrxl69clrGLcc",
            "service_url": "https://gateway.watsonplatform.net/natural-language-understanding/api"
        }
    }
    json.dump(creds, cred_file)
    cred_file.close()


# Find OSNA_HOME path
if 'OSNA_HOME' in os.environ:
    osna_path = os.environ['OSNA_HOME']
else:
    osna_path = os.environ['HOME'] + os.path.sep + '.osna' + os.path.sep

# Make osna directory if not present
try:
    os.makedirs(osna_path)
except:
    pass

# twitter and ibm cloud credentials
credentials_path = osna_path + 'credentials.json'

if not os.path.isfile(credentials_path):
    write_default_credentials(credentials_path)

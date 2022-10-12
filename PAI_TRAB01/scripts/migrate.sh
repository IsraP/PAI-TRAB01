#!/bin/sh
python3 manage.py makemigrations PAI
python3 manage.py migrate

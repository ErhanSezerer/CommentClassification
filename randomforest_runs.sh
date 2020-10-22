#!/bin/sh

python main.py --classifier="randomforest" --estimators=10
python main.py --classifier="randomforest" --estimators=25
python main.py --classifier="randomforest" --estimators=50
python main.py --classifier="randomforest" --estimators=75
python main.py --classifier="randomforest" --estimators=100
python main.py --classifier="randomforest" --estimators=200
python main.py --classifier="randomforest" --estimators=300
python main.py --classifier="randomforest" --estimators=500
python main.py --classifier="randomforest" --estimators=1000

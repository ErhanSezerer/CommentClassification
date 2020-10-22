#!/bin/sh

python main.py --classifier="bert" --use_gpu=True --epochs=2 --lr=5e-5 
python main.py --classifier="bert" --use_gpu=True --epochs=3 --lr=5e-5
python main.py --classifier="bert" --use_gpu=True --epochs=4 --lr=5e-5
python main.py --classifier="bert" --use_gpu=True --epochs=5 --lr=5e-5
python main.py --classifier="bert" --use_gpu=True --epochs=6 --lr=5e-5 
python main.py --classifier="bert" --use_gpu=True --epochs=7 --lr=5e-5
python main.py --classifier="bert" --use_gpu=True --epochs=8 --lr=5e-5
python main.py --classifier="bert" --use_gpu=True --epochs=9 --lr=5e-5

python main.py --classifier="bert" --use_gpu=True --epochs=2 --lr=1e-5 
python main.py --classifier="bert" --use_gpu=True --epochs=3 --lr=1e-5
python main.py --classifier="bert" --use_gpu=True --epochs=4 --lr=1e-5
python main.py --classifier="bert" --use_gpu=True --epochs=5 --lr=1e-5
python main.py --classifier="bert" --use_gpu=True --epochs=6 --lr=1e-5 
python main.py --classifier="bert" --use_gpu=True --epochs=7 --lr=1e-5
python main.py --classifier="bert" --use_gpu=True --epochs=8 --lr=1e-5
python main.py --classifier="bert" --use_gpu=True --epochs=9 --lr=1e-5

python main.py --classifier="bert" --use_gpu=True --epochs=2 --lr=1e-4 
python main.py --classifier="bert" --use_gpu=True --epochs=3 --lr=1e-4
python main.py --classifier="bert" --use_gpu=True --epochs=4 --lr=1e-4
python main.py --classifier="bert" --use_gpu=True --epochs=5 --lr=1e-4
python main.py --classifier="bert" --use_gpu=True --epochs=6 --lr=1e-4 
python main.py --classifier="bert" --use_gpu=True --epochs=7 --lr=1e-4
python main.py --classifier="bert" --use_gpu=True --epochs=8 --lr=1e-4
python main.py --classifier="bert" --use_gpu=True --epochs=9 --lr=1e-4

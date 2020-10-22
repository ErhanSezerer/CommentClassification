#!/bin/sh

python main.py --classifier="doc2vec" --emb_dimension=50 --epochs=10
python main.py --classifier="doc2vec" --emb_dimension=50 --epochs=25
python main.py --classifier="doc2vec" --emb_dimension=50 --epochs=50
python main.py --classifier="doc2vec" --emb_dimension=50 --epochs=100

python main.py --classifier="doc2vec" --emb_dimension=100 --epochs=10
python main.py --classifier="doc2vec" --emb_dimension=100 --epochs=25
python main.py --classifier="doc2vec" --emb_dimension=100 --epochs=50
python main.py --classifier="doc2vec" --emb_dimension=100 --epochs=100

python main.py --classifier="doc2vec" --emb_dimension=300 --epochs=10
python main.py --classifier="doc2vec" --emb_dimension=300 --epochs=25
python main.py --classifier="doc2vec" --emb_dimension=300 --epochs=50
python main.py --classifier="doc2vec" --emb_dimension=300 --epochs=100

python main.py --classifier="doc2vec" --emb_dimension=500 --epochs=10
python main.py --classifier="doc2vec" --emb_dimension=500 --epochs=25
python main.py --classifier="doc2vec" --emb_dimension=500 --epochs=50
python main.py --classifier="doc2vec" --emb_dimension=500 --epochs=100

python main.py --classifier="doc2vec" --emb_dimension=1000 --epochs=10
python main.py --classifier="doc2vec" --emb_dimension=1000 --epochs=25
python main.py --classifier="doc2vec" --emb_dimension=1000 --epochs=50
python main.py --classifier="doc2vec" --emb_dimension=1000 --epochs=100

import logging
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import BertTokenizer
from args import args
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
#doc2vec
from gensim.models.doc2vec import TaggedDocument
from nltk.tokenize import word_tokenize

logger = logging.getLogger('preprocess.py')
import numpy as np
import os



def read_files(file_path):
	#read the dataset and find the labels
	#description=0 quality=1, planning=2, currentInfo=3, generalInfo=4
	try: 
		train_file = os.path.join(file_path, "train.csv")
		test_file = os.path.join(file_path, "test.csv")
		train_df = pd.read_csv(train_file, sep='\t')
		test_df = pd.read_csv(test_file, sep='\t')
		train_text = train_df.text.values
		train_labels = train_df.label.values
		test_text = test_df.text.values
		test_labels = test_df.label.values
		#split into (0.7,0.1,0.2) partitions
		train_text, dev_text, train_labels, dev_labels = train_test_split(train_text, train_labels, test_size=0.125, random_state=42)
	except:
		annotation_file = os.path.join(file_path, "annotations.csv")
		annotations_df = pd.read_csv(annotation_file, sep='\t')
		train_data_raw = annotations_df.values

		train_text = []
		train_labels = []
		for line in tqdm(train_data_raw):
			if line[1]==1:
				train_labels.append(0)
				train_text.append(line[0])
			elif line[2]==1:
				train_labels.append(1)
				train_text.append(line[0])
			elif line[3]==1:
				train_labels.append(2)
				train_text.append(line[0])
			elif line[4]==1:
				train_labels.append(3)
				train_text.append(line[0])
			elif line[5]==1:
				train_labels.append(4)
				train_text.append(line[0])
		#split into (0.7,0.1,0.2) partitions
		train_text, test_text, train_labels, test_labels = train_test_split(train_text, train_labels, test_size=0.30, random_state=42)
		dev_text, test_text, dev_labels, test_labels = train_test_split(test_text, test_labels, test_size=0.66, random_state=42)	

	return train_text, dev_text, test_text, train_labels, dev_labels, test_labels 






def read_samples_bert(file_path):
	train_text, dev_text, test_text, train_labels, dev_labels, test_labels = read_files(file_path)

	#tokenize for bert
	logging.info("Tokenizing the Dataset for BERT...")
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

	train_ids = []
	train_att_mask = []
	dev_ids = []
	dev_att_mask = []
	test_ids = []
	test_att_mask = []
	for article in train_text:
		encoded_article = tokenizer.encode_plus(article, add_special_tokens=True, max_length=args.MAX_LEN,
                                                pad_to_max_length=True,
                                                return_attention_mask=True, return_tensors='pt')
		train_ids.append(encoded_article['input_ids'])
		train_att_mask.append(encoded_article['attention_mask'])
	for article in test_text:
		encoded_article = tokenizer.encode_plus(article, add_special_tokens=True, max_length=args.MAX_LEN,
                                                pad_to_max_length=True,
                                                return_attention_mask=True, return_tensors='pt')
		test_ids.append(encoded_article['input_ids'])
		test_att_mask.append(encoded_article['attention_mask'])
	for article in dev_text:
		encoded_article = tokenizer.encode_plus(article, add_special_tokens=True, max_length=args.MAX_LEN,
                                                pad_to_max_length=True,
                                                return_attention_mask=True, return_tensors='pt')
		dev_ids.append(encoded_article['input_ids'])
		dev_att_mask.append(encoded_article['attention_mask'])

	#prepare torch tensors from the data
	train_ids = torch.cat(train_ids, dim=0)
	test_ids = torch.cat(test_ids, dim=0)
	train_att_mask = torch.cat(train_att_mask, dim=0)
	test_att_mask = torch.cat(test_att_mask, dim=0)
	train_labels = torch.tensor(train_labels)
	test_labels = torch.tensor(test_labels)
	dev_ids = torch.cat(dev_ids, dim=0)
	dev_att_mask = torch.cat(dev_att_mask, dim=0)
	dev_labels = torch.tensor(dev_labels)

	train_dataset = TensorDataset(train_ids, train_att_mask, train_labels)
	test_dataset = TensorDataset(test_ids, test_att_mask, test_labels)
	dev_dataset = TensorDataset(dev_ids, dev_att_mask, dev_labels)

	return train_dataset, dev_dataset, test_dataset







def read_samples_ML(file_path):
	train_text, dev_text, test_text, train_labels, dev_labels, test_labels = read_files(file_path)

	vectorizer = TfidfVectorizer(ngram_range=(1,2))
	train_features = vectorizer.fit_transform(train_text)
	dev_features = vectorizer.transform(dev_text)
	test_features = vectorizer.transform(test_text)

	return train_features, dev_features, test_features, train_labels, dev_labels, test_labels





def read_samples_doc2vec(file_path):
	train_text, dev_text, test_text, train_labels, dev_labels, test_labels = read_files(file_path)

	train_samples = [TaggedDocument(words=word_tokenize(train_text[i].lower()), tags=[train_labels[i]]) for i in range(len(train_text))]
	dev_samples = [word_tokenize(dev_text[i].lower()) for i in range(len(dev_text))]
	test_samples = [word_tokenize(test_text[i].lower()) for i in range(len(test_text))]

	return train_samples, dev_samples, test_samples, dev_labels, test_labels










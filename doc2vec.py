from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import logging
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report
import csv
import os
from args import args
logger = logging.getLogger('doc2vec.py')
stats_columns = '{0:>5}|{1:>5}|{2:>5}|{3:>5}|{4:>5}|{5:>5}|{6:>5}|{7:>5}|{8:>5}|{9:>5}|{10:>5}'

stats_template = '{mode} Accuracy: {acc}\n' \
                 '{mode} F1: {f1}\n' \
                 '{mode} ave. F1: {f1_ave}\n' \
                 '{mode} Recall: {recall}\n' \
                 '{mode} ave. Recall: {recall_ave}\n' \
                 '{mode} Precision: {prec}\n' \
                 '{mode} ave. Precision: {prec_ave}\n'






def train_doc2vec(epoch, model, logreg, train_samples, dev_samples, dev_labels, checkpoint_dir, results_file):
	
	#training
	model.build_vocab(train_samples)
	for epoch in range(epoch):
		logger.info('iteration {0}'.format(epoch))
		model.train(train_samples, total_examples=model.corpus_count, epochs=model.iter)

		# decrease the learning rate
		model.alpha -= 0.0002
		# fix the learning rate, no decay
		model.min_alpha = model.alpha

	#train LogReg
	train_vectors = [model.infer_vector(training_sentence.words) for training_sentence in train_samples]
	train_labels = [training_sentence.tags for training_sentence in train_samples]
	logreg.fit(train_vectors, train_labels)


	#validation
	dev_vectors = []
	for i in range(len(dev_samples)):
		dev_vectors.append(model.infer_vector(dev_samples[i]))	
	predictions = logreg.predict(dev_vectors)

	acc, f1, recall, prec, f1_ave, recall_ave, prec_ave = calculate_metrics(dev_labels, predictions, average=None)
	print(stats_template.format(mode='eval', acc=acc, f1=f1, f1_ave=f1_ave, recall=recall,
                  recall_ave=recall_ave, prec=prec, prec_ave=prec_ave))
	
	with open(results_file, 'a') as output_file:
		cw = csv.writer(output_file, delimiter='\t')
		cw.writerow(["valid",
				'%0.4f' % acc,
				'%0.4f' % prec_ave,
				'%0.4f' % recall_ave,
				'%0.4f' % f1_ave,
				'%0.4f' % f1[0],
				'%0.4f' % f1[1],
				'%0.4f' % f1[2],
				'%0.4f' % f1[3],
				'%0.4f' % f1[4],
				'%0.4f' % prec[0],
				'%0.4f' % prec[1],
				'%0.4f' % prec[2],
				'%0.4f' % prec[3],
				'%0.4f' % prec[4],
				'%0.4f' % recall[0],
				'%0.4f' % recall[1],
				'%0.4f' % recall[2],
				'%0.4f' % recall[3],
				'%0.4f' % recall[4]])



def test_doc2vec(test_samples, test_labels, model, logreg):
	test_vectors = []

	for i in range(len(test_samples)):
		test_vectors.append(model.infer_vector(test_samples[i]))	
	predictions = logreg.predict(test_vectors)

	if args.print_preds == True:
		path = os.path.join(args.checkpoint_dir, "preds_doc2vec.csv")
		with open(path, 'w') as output_file: 
			cw = csv.writer(output_file, delimiter='\t')
			for pred in predictions:
				cw.writerow(str(pred))

	acc, f1, recall, prec, f1_ave, recall_ave, prec_ave = calculate_metrics(test_labels, predictions, average=None)
	print(stats_template.format(mode='eval', acc=acc, f1=f1, f1_ave=f1_ave, recall=recall,
                  recall_ave=recall_ave, prec=prec, prec_ave=prec_ave))

	return acc, f1, recall, prec, f1_ave, recall_ave, prec_ave













def calculate_metrics(label, pred, average='binary'):
	#pred_class = np.concatenate([np.argmax(numarray, axis=1) for numarray in pred]).ravel()
	#label_class = np.concatenate([numarray for numarray in label]).ravel()

	logging.debug('Expected: \n{}'.format(label[:20]))
	logging.debug('Predicted: \n{}'.format(pred[:20]))

	acc = round(accuracy_score(label, pred), 4)
	f1 = [round(score, 4) for score in f1_score(label, pred, average=average)]
	recall = [round(score, 4) for score in recall_score(label, pred, average=average)]
	prec = [round(score, 4) for score in precision_score(label, pred, average=average)]

	f1_ave = f1_score(label, pred, average='weighted')
	recall_ave = recall_score(label, pred, average='weighted')
	prec_ave = precision_score(label, pred, average='weighted')

	return acc, f1, recall, prec, f1_ave, recall_ave, prec_ave


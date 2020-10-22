import pandas as pd
from args import args
import os
from scipy.stats import f_oneway,spearmanr, friedmanchisquare
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import confusion_matrix
import numpy as np
import csv


def read_files():
	#doc2vec
	file_path = os.path.join(args.checkpoint_dir, "preds_doc2vec.csv")
	pred_d2v = pd.read_csv(file_path, sep='\t')
	pred_d2v.columns =['pred']
	pred_d2v = pred_d2v.pred.values
	print(pred_d2v)

	#random forest
	file_path = os.path.join(args.checkpoint_dir, "preds_randomforest.csv")
	pred_rf = pd.read_csv(file_path, sep='\t')
	pred_rf.columns =['pred']
	pred_rf = pred_rf.pred.values
	print(pred_rf)

	#svm
	file_path = os.path.join(args.checkpoint_dir, "preds_svm.csv")
	pred_svm = pd.read_csv(file_path, sep='\t')
	pred_svm.columns =['pred']
	pred_svm = pred_svm.pred.values
	print(pred_svm)

	#bert
	file_path = os.path.join(args.checkpoint_dir, "preds_bert.csv")
	pred_bert = pd.read_csv(file_path, sep='\t')
	pred_bert.columns =['pred']
	pred_bert = pred_bert.pred.values
	print(pred_bert)

	return np.vstack((pred_d2v, pred_rf, pred_svm, pred_bert))





if __name__ == '__main__':
	preds = read_files()
	
	Fscores=[]
	pvalues=[]
	for i in range(len(preds)):
		Frow=[]
		prow=[]
		    
		#friedman test
		if args.test=="friedman":
			preds_temp = preds.tolist()
			preds_temp.pop(i)
			Frow, prow = friedmanchisquare(preds_temp[0], preds_temp[1], preds_temp[2])
		#rest of the tests
		else:
			for j in range(len(preds)):
				if args.test=="anova":
					F, p = f_oneway(preds[i], preds[j])
				elif args.test=="spearman":
					F, p = spearmanr(preds[i], preds[j])
				elif args.test=="mcnemar":
					contingency = confusion_matrix(preds[i], preds[j]).tolist()
					b = mcnemar(contingency, exact=True)
					F = b.statistic
					p = b.pvalue
				Frow.append(F)
				prow.append(p)
			    
		Fscores.append(Frow)
		pvalues.append(prow)

	if args.test=="friedman":
		df_F = pd.DataFrame(Fscores, index=["wo-doc2vec","wo-randomforest","wo-svm","wo-bert"])
		df_p = pd.DataFrame(pvalues, index=["wo-doc2vec","wo-randomforest","wo-svm","wo-bert"])
	else:
		df_F = pd.DataFrame(Fscores, index=["doc2vec","randomforest","svm","bert"], columns=["doc2vec","randomforest","svm","bert"])
		df_p = pd.DataFrame(pvalues, index=["doc2vec","randomforest","svm","bert"], columns=["doc2vec","randomforest","svm","bert"])


	if args.test=="anova":
		output_path_F = os.path.join(args.checkpoint_dir, "anova_F.csv")
		output_path_p = os.path.join(args.checkpoint_dir, "anova_p.csv")
		df_F.to_csv(output_path_F, sep='\t')	
		df_p.to_csv(output_path_p, sep='\t')
	elif args.test=="spearman":
		output_path_F = os.path.join(args.checkpoint_dir, "spearman_corr.csv")
		output_path_p = os.path.join(args.checkpoint_dir, "spearman_p.csv")
		df_F.to_csv(output_path_F, sep='\t')	
		df_p.to_csv(output_path_p, sep='\t')
	elif args.test=="mcnemar":
		output_path_F = os.path.join(args.checkpoint_dir, "mcnemar_stat.csv")
		output_path_p = os.path.join(args.checkpoint_dir, "mcnemar_p.csv")
		df_F.to_csv(output_path_F, sep='\t')	
		df_p.to_csv(output_path_p, sep='\t')
	elif args.test=="friedman":
		output_path_F = os.path.join(args.checkpoint_dir, "friedman_stat.csv")
		output_path_p = os.path.join(args.checkpoint_dir, "friedman_p.csv")
		df_F.to_csv(output_path_F, sep='\t')	
		df_p.to_csv(output_path_p, sep='\t')
	    	


















from source.utils.metric import F1Metric
import pandas as pd
# import argparse
# parser = argparse.ArgumentParser(description='F1-score')
# parser.add_argument('--file', default= '../Data/default.pickle', type=str, metavar='N', help='test data path')
# args = parser.parse_args()


# d = pd.read_csv(args.file,',')
# print('calculating ...')
# print(d.columns)
# tar_text = d['tar_text'].tolist()
# pred_text = d['pred_text'].tolist()

def F1_Score(tar_text, pred_text):
	references = [[tar_text[i]] for i in range(len(tar_text))]
	candidates = [pred_text[i] for i in range(len(pred_text))]
	scores = 0
	i = 0
	for candidate, reference in zip(candidates,references):
		i += 1
		# print(candidate,reference)
		# print('f1', F1Metric.compute(candidate, reference))
		value = F1Metric.compute(candidate, reference)
		# if value == 0:
		# 	print(candidate,reference)

		scores = scores + int(value)
	f1_score = (scores/i)

	return f1_score

# print(F1_Score(tar_text,pred_text))
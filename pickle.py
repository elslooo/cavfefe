import cPickle as pkl

with open("meteor_scores/meteor_score_dict_explanation_1006_pred.p") as f:
	data = pkl.load(f)

print data
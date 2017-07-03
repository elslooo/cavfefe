import cPickle as pkl 
from lib.cv import FeatureCache

feature_cache = FeatureCache()
feature_cache.restore("data/cv/features.csv")
matching = dict()

# 11789 len(labels)
pred = [feature_cache.get(i) for i in range(1, 11789)]

with open('data/CUB_200_2011/CUB_200_2011/images.txt') as file:
	for line in file:
		id, image = line[:-1].split(' ')  
		matching[id] = {image:pred[int(id) - 1][0]}

# Lelijk maar noodzakelijk
predictions = {entry.keys()[0]:entry.values()[0] for entry in matching.values()}

pkl.dump(predictions, open( "data/predictions.p", "wb" ) )

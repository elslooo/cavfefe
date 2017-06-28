import sys
from lib.init import *
import cPickle as pkl
import time

sys.path.insert(0, coco_eval_path)
from pycocoevalcap.meteor import meteor
from pycocoevalcap.cider import cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer


class EvaluationScorer(object):
	""" label_path is the path leading to the ground truth sentences (labels)
		pred_path is path leading to generated sentences (predictions)
		(both in json format)"""
	def __init__(self, label_path, pred_path):
		
		self.labels = read_json(label_path)
		self.predictions = read_json(pred_path)
		self.labeled_class_annotations = {}
		self.pred_class_annotations = {}
		self.tfidf_dict = {}
		self.get_annotations()

	def get_annotations(self):

		for a in self.labels['annotations']:
			cl = int(a['image_id'].split('/')[0].split('.')[0]) - 1
			if cl not in self.labeled_class_annotations:
				self.labeled_class_annotations[cl] = {}
				self.labeled_class_annotations[cl]['all_images'] = [] 
			self.labeled_class_annotations[cl]['all_images'].append({'caption': a['caption'], 'id': a['image_id'], 'image_id': a['image_id']})
	
		for a in self.predictions:
			cl = int(a['image_id'].split('/')[0].split('.')[0]) - 1 
			im = a['image_id']
			if cl not in self.pred_class_annotations:
				self.pred_class_annotations[cl] = {}
			if im not in self.pred_class_annotations[cl].keys():
				self.pred_class_annotations[cl][im] = []
			self.pred_class_annotations[cl][im].append({'caption': a['caption'], 'id': a['image_id'], 'image_id': a['image_id']})

		tokenizer = PTBTokenizer()
		for key in self.labeled_class_annotations:
			self.labeled_class_annotations[key] = tokenizer.tokenize(self.labeled_class_annotations[key])
		for key in self.pred_class_annotations:
			self.pred_class_annotations[key] = tokenizer.tokenize(self.pred_class_annotations[key])

		print "Done tokenizing"

	def compute_meteor(self, gen, gt):

	  meteor_scorer = meteor.Meteor() 
	  scores, imgIds = meteor_scorer.compute_score(gt, gen)
	  
	  return scores, imgIds

	def compute_cider(self, gen, gt):
    
	    cider_scorer = cider.Cider()
	    scores, imgIds = cider_scorer.compute_score(gt, gen)

	    return scores, imgIds

	def get_scores(self):
		meteor_score = {}
		cider_score = {}
		for cl in sorted(self.pred_class_annotations.keys()):
			gts = {}
			gen = {}
			t = time.time()
			for cl_gt in sorted(self.labeled_class_annotations.keys()):
				for im in sorted(self.pred_class_annotations[cl].keys()):
					gen[im+('_%d' %cl_gt)] = self.pred_class_annotations[cl][im]
					gts[im+('_%d' %cl_gt)] = self.labeled_class_annotations[cl_gt]['all_images']


			scores, im_ids = self.compute_meteor(gen, gts) 
			for s, ii in zip(scores, im_ids):
				meteor_score[ii] = s

			scores, im_ids = self.compute_cider(gen, gts) 
			for s, ii in zip(scores, im_ids):
				cider_score[ii] = s

			print "Class %s took %f s." %(cl, time.time() -t)
		
		pkl.dump(meteor_score, open('meteor_scores/meteor_score_dict_test2.p', 'w'))
		pkl.dump(cider_score, open('cider_scores/cider_score_dict_test2.p', 'w'))


label_path = './data/TMP_descriptions_bird.train_noCub.fg.json'
pred_path = './generated_sentences/TMP_generation_result.json'

scorer = EvaluationScorer(label_path, pred_path)
scorer.get_scores()
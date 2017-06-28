import sys
import time
import pickle as pkl
import os
# sys.path.append('utils/')
from lib.init import *

sys.path.insert(0, coco_eval_path)
from pycocoevalcap.meteor import meteor
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

# def determine_anno_path(dataset_name, split_name):
#   if dataset_name == 'coco':
#     return COCO_ANNO_PATH % split_name 
#   if dataset_name == 'birds':
#     return bird_anno_path % split_name 
#   if dataset_name == 'birds_fg':
#     return bird_anno_path_fg % (split_name)
#   else:
#     raise Exception ("do not annotation path for dataset %s." %dataset_name)

def compute_meteor(gen, gt):
  meteor_scorer = meteor.Meteor() 
  score, scores = meteor_scorer.compute_score(gt, gen)
  im_ids = range(len(scores))
  # return scores, meteor_scorer.imgIds
  return scores, im_ids

def eval_class_meteor(tag):
  #Make the reference for an image all reference sentences from the corresponding class.  

  ground_truth_annotation_path = './data/TMP_descriptions_bird.train_noCub.fg.json'

  # TODO: Convert the generated sentence to the same format
  gen_annotations = './generated_sentences/TMP_generation_result.json'

  if not os.path.isdir('meteor_scores'):
    os.mkdir('meteor_scores')

  # dataset = 'birds_fg'
  # # split = 'test'
  # gt_comp = 'train_noCub' #alternatively could use sentences from val or test. Can think of this metric as a measure between a generated sentence and the NN class in the reference set.  

  # image_root = eval_generation.determine_image_pattern(dataset, split) 
  # vocab_file = 'data/vocab.txt'
  # vocab = open_txt(vocab_file)

  #combine gt annotations for each class
  # anno_path_ref = determine_anno_path(dataset, gt_comp)
  ref_annotations = read_json(ground_truth_annotation_path)
  gen_annotations = read_json(gen_annotations)

  #create dict which has all annotations which correspond to a certain class in the reference set
  gt_class_annotations = {}
  
  # Get class for every image
  for a in ref_annotations['annotations']:
    cl = int(a['image_id'].split('/')[0].split('.')[0]) - 1
    if cl not in gt_class_annotations:
      gt_class_annotations[cl] = {}
      gt_class_annotations[cl]['all_images'] = [] 
    gt_class_annotations[cl]['all_images'].append({'caption': a['caption'], 'id': a['image_id'], 'image_id': a['image_id']})


  gen_class_annotations = {}
  for a in gen_annotations:
    cl = int(a['image_id'].split('/')[0].split('.')[0]) - 1 
    im = a['image_id']
    if cl not in gen_class_annotations:
      gen_class_annotations[cl] = {}
    if im not in gen_class_annotations[cl].keys():
      gen_class_annotations[cl][im] = []
    gen_class_annotations[cl][im].append({'caption': a['caption'], 'id': a['image_id'], 'image_id': a['image_id']})

  # for im in sorted(gen_class_annotations[0].keys()):
  #   print im

  t = time.time()
  tokenizer = PTBTokenizer()
  for key in gt_class_annotations:
    gt_class_annotations[key] = tokenizer.tokenize(gt_class_annotations[key])
  for key in gen_class_annotations:
    gen_class_annotations[key] = tokenizer.tokenize(gen_class_annotations[key])

  # for im in sorted(gen_class_annotations[0].keys()):
  #   print im

  # print gen_class_annotations[0]
  # print "new\n"
  # print gen_class_annotations[1]
  # print gt_class_annotations[0]['all_images']
  print "Time for tokenization: %f." %(time.time()-t)
  score_dict = {}
  for cl in sorted(gen_class_annotations.keys()):
    gts = {}
    gen = {}
    t = time.time()
    for cl_gt in sorted(gt_class_annotations.keys()):
      for im in sorted(gen_class_annotations[cl].keys()):
        gen[im+('_%d' %cl_gt)] = gen_class_annotations[cl][im]
        gts[im+('_%d' %cl_gt)] = gt_class_annotations[cl_gt]['all_images']

      # print len(gen_class_annotations)
      # print gen_class_annotations
    scores, im_ids = compute_meteor(gen, gts) 
    for s, ii in zip(scores, im_ids):
      score_dict[ii] = s
    print "Class %s took %f s." %(cl, time.time() -t)
    pkl.dump(score_dict, open('meteor_scores/meteor_score_dict_%s.p' %(tag), 'w'))

eval_class_meteor('explanation_1006_pred')
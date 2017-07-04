from lib.eval import eval_scorer

label_path = './data/TMP_descriptions_bird.train_noCub.fg.json'
pred_path = './products/sentences.json'

scorer = eval_scorer.EvaluationScorer(label_path, pred_path)
scorer.get_scores()

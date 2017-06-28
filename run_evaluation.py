from lib.eval import eval_scorer

label_path = './data/TMP_descriptions_bird.train_noCub.fg.json'
pred_path = './generated_sentences/TMP_generation_result.json'

scorer = EvaluationScorer(label_path, pred_path)
scorer.get_scores()


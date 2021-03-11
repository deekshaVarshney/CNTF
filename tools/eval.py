# m
# python eval.py --eval_dir ../outputs/cmu_new

import sys
sys.path.append('../')
import argparse
import json
import numpy as np
from source.utils.metrics import moses_multi_bleu
from source.utils.f1_score import F1_Score
from nlgeval import compute_metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", type=str)
    parser.add_argument("--pred_file", type=str)
    parser.add_argument("--ref_file", type=str)
    args = parser.parse_args()

    eval_dir = args.eval_dir

    eval_file = "%s/output.txt" % eval_dir

    # calculate metrics
    hyps = []
    refs = []
    with open(eval_file, 'r') as fr:
        for line in fr:
            dialog = json.loads(line.strip())
            pred_str = dialog["result"]
            gold_str = dialog["target"]
            hyps.append(pred_str)
            refs.append(gold_str)
    
    assert len(hyps) == len(refs)
    
    f_score = F1_Score(refs, hyps)

    hyp_arrys = np.array(hyps)
    ref_arrys = np.array(refs)
    bleu_score = moses_multi_bleu(hyp_arrys, ref_arrys, lowercase=True)
    metrics_dict = compute_metrics(hypothesis=args.pred_file, references=[args.ref_file])
    
    output_str = "BLEU SCORE(PERL): %.3f\n" % bleu_score
    output_str += "F1 SCORE: %.2f%%\n" % (f_score * 100)
    output_str += str(metrics_dict)
    
    # write evaluation results to file
    out_file = "%s/eval.result1.txt" % eval_dir
    with open(out_file, 'w') as fw:
        fw.write(output_str)
    print("Saved evaluation results to '{}.'".format(out_file))


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
    parser.add_argument("--result_file", type=str, default='result')
    args = parser.parse_args()

    eval_dir = args.eval_dir

    eval_file = f"{eval_dir}/output.json"

    # calculate metrics
    hyps = []
    refs = []
    
    refs = [l.strip().split("\n")[0][1:-1] for l in open(f'{eval_dir}/{args.ref_file}.txt', 'r').readlines()]
    hyps = [l.strip().split("\n")[0][1:-1] for l in open(f'{eval_dir}/{args.pred_file}.txt', 'r').readlines()]
    print(hyps[0])
    print(refs[0])
    assert len(hyps) == len(refs)
    
    f_score = F1_Score(refs, hyps)
    print("F1score", f_score)
    hyp_arrys = np.array(hyps)
    ref_arrys = np.array(refs)
    bleu_score = moses_multi_bleu(hyp_arrys, ref_arrys, lowercase=True)
    print("bleu", bleu_score)
    metrics_dict = compute_metrics(hypothesis=f'{eval_dir}/{args.pred_file}.txt', references=[f'{args.eval_dir}/{args.ref_file}.txt'])
    print(metrics_dict)
    
    output_str = "BLEU SCORE(PERL): %.3f\n" % bleu_score
    
    output_str += "F1 SCORE: %.2f%%\n" % (f_score * 100)
    output_str += str(metrics_dict)
    
    out_file = f"{eval_dir}/{args.result_file}.txt"
    with open(out_file, 'w') as fw:
        fw.write(output_str)
    print("Saved evaluation results to '{}.'".format(out_file))


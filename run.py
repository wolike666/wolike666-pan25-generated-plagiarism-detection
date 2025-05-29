#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # 直接使用 Hugging Face 地址进行推理
    cmd = [
        "python", "scripts/predict-fast.py",
        "--model_dir", "jrluo/PlagiarismDetection-bert-base-train10000",
        "--data_dir",  args.input,
        "--truth_dir", args.input,
        "--output_dir",args.output,
        "--threshold_tfidf",        "0.7",
        "--threshold_jaccard_simple","0.4",
        "--threshold_jaccard_other", "0.7",
        "--threshold_bert",         "0.8",
        "--threshold_bert_simple",  "0.75",
        "--batch_size",             "64",
        "--min_len",                "50",
        "--merge_gap",              "20",
        "--chunk_size",             "100",
    ]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()

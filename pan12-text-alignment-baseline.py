#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基线程序：针对近似复制型抄袭的检测示例

作者: Arnd Oberlaender
邮箱: arnd.oberlaender@uni-weimar.de
版本: 1.1

此脚本提供 PAN 2013 文本对齐任务的参考实现，可作为抄袭检测流程的模板。
"""

import os
import string
import sys
import xml.dom.minidom
from pathlib import Path

# -------------------------------------------------------------------
# 全局常量
# -------------------------------------------------------------------
# 要移除的字符：标点符号和空白字符
DELETECHARS = ''.join([string.punctuation, string.whitespace])
# N-gram 长度（字符级）
LENGTH = 50

# -------------------------------------------------------------------
# 辅助函数
# -------------------------------------------------------------------

def tokenize(text, length):
    """
    将文本分割为固定长度的字符 n-gram，并记录每个 n-gram 的起止位置。

    参数:
      text   -- 输入原文字符串
      length -- 每个 n-gram 的字符长度

    返回:
      字典：{ ngram: [(start_pos, end_pos), ...], ... }
    """
    tokens = {}
    token = []

    for i, ch in enumerate(text):
        if ch not in DELETECHARS:
            token.append((i, ch))
        if len(token) == length:
            # 拼接 n-gram 字符串并全小写
            ngram = ''.join([c.lower() for _, c in token])
            tokens.setdefault(ngram, []).append((token[0][0], token[-1][0]))
            # 移动窗口：丢弃最早一个字符
            token = token[1:]

    return tokens


def serialize_features(susp, src, features, outdir):
    """
    将检测结果序列化为 XML 文件，符合 PAN 文本对齐语料库格式。

    参数:
      susp     -- 可疑文档文件名
      src      -- 源文档文件名
      features -- 检测到的特征列表，格式为 [((src_start, src_end), (susp_start, susp_end)), ...]
      outdir   -- 输出目录
    """
    impl = xml.dom.minidom.getDOMImplementation()
    doc = impl.createDocument(None, 'document', None)
    root = doc.documentElement
    root.setAttribute('reference', susp)

    for src_span, susp_span in features:
        feature = doc.createElement('feature')
        feature.setAttribute('name', 'detected-plagiarism')
        feature.setAttribute('this_offset', str(susp_span[0]))
        feature.setAttribute('this_length', str(susp_span[1] - susp_span[0]))
        feature.setAttribute('source_reference', src)
        feature.setAttribute('source_offset', str(src_span[0]))
        feature.setAttribute('source_length', str(src_span[1] - src_span[0]))
        root.appendChild(feature)

    # 输出文件：suspicious-documentXYZ-source-documentABC.xml
    out_path = os.path.join(outdir, f"{susp.split('.')[0]}-{src.split('.')[0]}.xml")
    with open(out_path, 'w', encoding='utf-8') as f:
        doc.writexml(f, encoding='utf-8', addindent="  ", newl="\n")

class Baseline:
    """
    基线检测类：实现一个近似复制检测流程示例。

    方法:
      process()    -- 执行完整的处理流程
      preprocess() -- 文档预处理与 n-gram 建立
      compare()    -- 基于 n-gram 的匹配及最长扩展
      postprocess()-- 序列化并输出 XML
    """
    def __init__(self, susp_id, susp_text, src_id, src_text, outdir):
        self.susp_text = susp_text
        self.src_text = src_text
        self.susp_id = susp_id
        self.src_id = src_id
        self.outdir = outdir
        self.detections = []

    def process(self):
        # 执行检测流程
        self.preprocess()
        self.detections = self.compare()
        self.postprocess()

    def preprocess(self):
        """
        文本预处理：
          - 去除 DELETECHARS 中的字符
          - 对可疑文档切分 n-gram 并建立字典索引
        """
        self.tokens = tokenize(self.susp_text, LENGTH)

    def compare(self):
        """
        对源文档滑动生成 n-gram，与可疑文档索引比较，
        一旦匹配就尝试向后扩展以获取最长连续匹配区间。
        """
        detections = []
        skipto = -1
        token = []

        for i, ch in enumerate(self.src_text):
            if i > skipto:
                if ch not in DELETECHARS:
                    token.append((i, ch))
                if len(token) == LENGTH:
                    ngram = ''.join([c.lower() for _, c in token])
                    if ngram in self.tokens:
                        # 找到匹配位置
                        src_start, src_end = token[0][0], token[-1][0]
                        # 扩展匹配
                        susp_positions = self.tokens[ngram]
                        best = (src_start, src_end, susp_positions[0][0], susp_positions[0][1])
                        for s_start, s_end in susp_positions:
                            cur_src, cur_susp = src_start, s_start
                            # 向后匹配字符
                            while (cur_src < len(self.src_text) and
                                   cur_susp < len(self.susp_text) and
                                   self.src_text[cur_src] == self.susp_text[cur_susp]):
                                cur_src += 1
                                cur_susp += 1
                                # 跳过 DELETECHARS
                                while cur_src < len(self.src_text) and self.src_text[cur_src] in DELETECHARS:
                                    cur_src += 1
                                while cur_susp < len(self.susp_text) and self.susp_text[cur_susp] in DELETECHARS:
                                    cur_susp += 1
                            # 更新最佳
                            if cur_src - src_start > best[1] - best[0]:
                                best = (src_start, cur_src, s_start, cur_susp)
                        detections.append(((best[0], best[1]), (best[2], best[3])))
                        skipto = best[1]
                        token = []
                    else:
                        token.pop(0)

        return detections

    def postprocess(self):
        """
        将检测结果转为 XML 并写入输出目录
        """
        serialize_features(self.susp_id, self.src_id, self.detections, self.outdir)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python pan12-text-alignment-baseline.py <corpus_dir> <output_dir>")
        sys.exit(1)

    corpus_dir = sys.argv[1]
    outdir = sys.argv[2]
    if not outdir.endswith(os.sep):
        outdir += os.sep
    Path(outdir).mkdir(exist_ok=True, parents=True)

    # 读取 pairs 文件
    pairs_file = os.path.join(corpus_dir, "pairs")
    with open(pairs_file, 'r', encoding='utf-8') as pf:
        for line in pf:
            susp_id, src_id = line.strip().split()
            susp_path = os.path.join(corpus_dir, "susp", susp_id)
            src_path = os.path.join(corpus_dir, "src", src_id)
            with open(susp_path, 'r', encoding='utf-8') as f:
                susp_text = f.read()
            with open(src_path, 'r', encoding='utf-8') as f:
                src_text = f.read()
            baseline = Baseline(susp_id, susp_text, src_id, src_text, outdir)
            baseline.process()
    print("处理完成，结果保存在:", outdir)

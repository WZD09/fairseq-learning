# encoding: utf-8
"""
@author: Xiaofei Sun

@time: 2022/05/07
@desc: 这只飞很懒
"""
import json
import os
from collections import Counter

import numpy
import torch
from fairseq import utils
from fairseq.data import Dictionary, LanguagePairDataset, FairseqDataset
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask


class MyJsonDataset(FairseqDataset):
    """
    从json文件中读取数据
    """

    def __init__(self, tokens_list):
        self.tokens_list = [torch.LongTensor(tokens) for tokens in tokens_list]
        self.sizes = numpy.array([len(tokens) for tokens in tokens_list])
        self.size = len(self.tokens_list)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError("index out of range")

    def __getitem__(self, i):
        self.check_index(i)
        return self.tokens_list[i]

    def __del__(self):
        pass

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

from fairseq.tasks.translation import TranslationConfig
@register_task("my_translation", dataclass=TranslationConfig)
class MyTranslationTask(TranslationTask):

    # @staticmethod
    # def add_args(parser):
    #     # Add some command-line arguments for specifying where the data is
    #     # located and the maximum supported input length.
    #     parser.add_argument('data', metavar='FILE',
    #                         help='file prefix for data')
    #     parser.add_argument('--max_source_positions', default=1024, type=int,
    #                         help='max input length')
    #     parser.add_argument('--max_target_positions', default=1024, type=int,
    #                         help='max input length')
    #     parser.add_argument('--eval_bleu', default=False, type=bool,
    #                         help='max input length')
    #     parser.add_argument('--left_pad_source', default=False, type=bool,
    #                         help='max input length')
    #     parser.add_argument('--left_pad_target', default=False, type=bool,
    #                         help='max input length')



    @staticmethod
    def build_dictionary(file_name, split):
        with open(file_name) as input_file:
            data = json.load(input_file)
        dict = Dictionary()
        counter = Counter()
        for datum in data:
            counter.update(datum[split].split())
            counter.update([dict.eos_word])
        for w, c in sorted(counter.items()):
            dict.add_symbol(w, c)
        dict.finalize()
        return dict

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = utils.eval_bool(args.left_pad_source)
        args.left_pad_target = utils.eval_bool(args.left_pad_target)

        paths = utils.split_paths(args.data)
        assert len(paths) == 1
        # load dictionaries
        src_dict = cls.load_dictionary(
            os.path.join(paths[0], f"dict_src.txt")
        )
        tgt_dict = cls.load_dictionary(
            os.path.join(paths[0], f"dict_tgt.txt")
        )
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()

        return cls(args, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        print("my load_dataset")
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        path = os.path.join(paths[0], f"{split}.json")
        print("path", path)
        with open(path) as input_file:
            data = json.load(input_file)
        src_tokens_list = []
        tgt_tokens_list = []
        for datum in data:
            src_tokens_list.append(datum["src_ids"])
            tgt_tokens_list.append(datum["tgt_ids"])
        src = MyJsonDataset(src_tokens_list)
        tgt = MyJsonDataset(tgt_tokens_list)
        self.datasets[split] = LanguagePairDataset(
            src=src,
            src_sizes=src.sizes,
            src_dict=self.src_dict,
            tgt=tgt,
            tgt_sizes=tgt.sizes,
            tgt_dict=self.tgt_dict,
        )

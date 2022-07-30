# 五、真实的翻译任务

下载解压：http://dl.fbaipublicfiles.com/fairseq/data/iwslt14/de-en.tgz

英语-德语翻译

## 1 处理数据

源码 examples/translation/prepare-iwslt14.sh 提供了一个处理数据的脚本，做分词 BPE 等，这里去掉这些自己做。

运行 1-prepare-data.sh

得到 iwslt14.tokenized.de-en/tmp

新建几个文件夹

```bash
mkdir -p iwslt14/raw iwslt14/tokenized iwslt14/bpe iwslt14/preprocessed
```

## 2 tokenize

2-tokenize.py 得到iwslt14/tokenized

用到了 sacremoses进行分词，是fairseq例子脚本（1-lstm用到的）moses-smt/mosesdecoder tokenizer的python部分。分词后输出两个文件给BPE用。

## 3 BPE

这里似乎有一点麻烦 windows不支持fastBPE, 所以我还使用了subword nmt 做BPE 要慢一些，然后写到json文件

## 4 add id

还是用上一个写好的tasks，存dict文件 给数据加上token id

``` 
[
  {
    "src": "wissen sie , eines der großen vern@@ ü@@ gen beim reisen und eine der freu@@ den bei der ...
    "tgt": "you know , one of the inten@@ se pleas@@ ures of travel and one of the deli@@ ghts of ...
    "src_ids": [
      22,
      7,
      9,
      ...
      25,
      23,
      2
    ],
    "tgt_ids": [
      19,
      7,
      18,
      ...
      28,
      2
    ]
  },
  {
```

## 5 model and task 同上

## 6 train

```
fairseq-train iwslt14/preprocessed --user-dir ./my_fq_module --task my_translation  --arch my_small_transformer   --optimizer adam  --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000  --max-tokens 12000  --max-epoch 2  --save-interval 2 --tensorboard-logdir logs  --skip-invalid-size-inputs-valid-test  --num-workers 1  --fp16

```

## 遇到的问题

不加tensorboard可以成功

```shell
Traceback (most recent call last):
  File "C:\install\anaconda_2\envs\p38gpu\Scripts\fairseq-train-script.py", line 33, in <module>
    sys.exit(load_entry_point('fairseq', 'console_scripts', 'fairseq-train')())
  File "d:\codes\fairseq-codes\fairseq\fairseq_cli\train.py", line 557, in cli_main
    distributed_utils.call_main(cfg, main)
  File "d:\codes\fairseq-codes\fairseq\fairseq\distributed\utils.py", line 369, in call_main
    main(cfg, **kwargs)
  File "d:\codes\fairseq-codes\fairseq\fairseq_cli\train.py", line 190, in main
    valid_losses, should_stop = train(cfg, trainer, task, epoch_itr)
  File "C:\install\anaconda_2\envs\p38gpu\lib\contextlib.py", line 75, in inner
    return func(*args, **kwds)
  File "d:\codes\fairseq-codes\fairseq\fairseq_cli\train.py", line 330, in train
    valid_losses, should_stop = validate_and_save(
  File "d:\codes\fairseq-codes\fairseq\fairseq_cli\train.py", line 421, in validate_and_save
    valid_losses = validate(cfg, trainer, task, epoch_itr, valid_subsets)
  File "d:\codes\fairseq-codes\fairseq\fairseq_cli\train.py", line 515, in validate
    progress.print(stats, tag=subset, step=trainer.get_num_updates())
  File "d:\codes\fairseq-codes\fairseq\fairseq\logging\progress_bar.py", line 454, in print
    self._log_to_tensorboard(stats, tag, step)
  File "d:\codes\fairseq-codes\fairseq\fairseq\logging\progress_bar.py", line 475, in _log_to_tensorboard
    writer.flush()
AttributeError: 'SummaryWriter' object has no attribute 'flush'
```
升级tensorboardx解决

```bash
 pip install --upgrade tensorboardx
```


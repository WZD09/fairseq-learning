# 一、 入门（Getting Started）


## 1.1 评估预训练模型（Evaluating Pre-trained Models）

这个模型用了BPE编码 a Byte Pair Encoding (BPE) vocabulary，所以我们翻译一个文本之前必须先给他编码 encoding 了。。

@@是BPE的分隔符，比如 b@@ est 所以恢复原始文本也很简单，直接sed命令给@@删了或者传个flag (--remove-bpe) 给 fairseq-generate.

在BPE之前，输入文本需要使用 mosesdecoder 中的tokenizer.perl来分词。

**英文分词：**
    
英文的标点符号和单词之间是没空格分隔的，所以如果直接对英文按照空格进行分词，cat和cat.就可能占据词典中两个词的位置，这些都是不合理的，会浪费词典的位置

moses分词小小例子：

    $ echo "A Republican 'strategy' to counter the re-election of Obama." | perl ~/script/mosesdecoder/scripts/tokenizer/tokenizer.perl
    >>>A Republican &apos; strategy &apos; to counter the re-election of Obama .

不指定任何参数的话会默认认为是英文，同时把标点分开，把引号转成 &apos。但是连字符不特别指定的话是不进行分割的。

**fairseq-interactive 交互的生成翻译**

用 fairseq-interactive 交互的生成翻译。 beam 5 用Moses分词 给定的BPE编码词汇表。自动删除BPE延续标记并对输出进行detokenize。

运行结果：
```
> fairseq-interactive    --path pre-trained-model/wmt14.en-fr.fconv-py/model.pt pre-trained-model/wmt14.en-fr.fconv-py   --beam 5 --source-lang en --target-lang fr    --tokenizer moses    --bpe subword_nmt --bpe-codes pre-trained-model/wmt14.en-fr.fconv-py/bpecodes
  INFO | fairseq_cli.interactive | {'_name': None, 'common': {'_name': None, 'no_progress_bar': False, 'log_interval': 100, ...
  INFO | fairseq.tasks.translation | [en] dictionary: 43771 types
  INFO | fairseq.tasks.translation | [fr] dictionary: 43807 types
  INFO | fairseq_cli.interactive | loading model(s) from pre-trained-model/wmt14.en-fr.fconv-py/model.pt

Why is it rare to discover new marine mammal species?
S-0     Why is it rare to discover new marine mam@@ mal species ?
W-0     1.319   seconds
H-0     -0.22002381086349487    Pourquoi est @-@ il rare de découvrir de nouvelles espèces de mammifères marins ?
D-0     -0.22002381086349487    Pourquoi est-il rare de découvrir de nouvelles espèces de mammifères marins ?
P-0     -0.3204 -0.4503 -0.1860 -0.3856 -0.2467 -0.2785 -0.1589 -0.2397 -0.1447 -0.1068 -0.1587 -0.1175 -0.1787 -0.1421 -0.1858
A-0     0-0 1-1 1-2 3-3 3-4 3-5 5-6 6-7 6-8 8-9 8-10 8-11 7-12 11-13
a b c D
S-1     a b c D
W-1     0.369   seconds
H-1     -0.5809122920036316     a b c D
D-1     -0.5809122920036316     a b c D
P-1     -1.2610 -0.5349 -0.2290 -0.6504 -0.2292
A-1     0-0 1-1 2-2 3-3


```

S: 原始源句的副本;H是生成的翻译输出（称为Hypothesis）

P: 每个token位置的positional score，包括文本中省略的句末token。

D: 解码后的输出

T: 参考目标

A：对齐信息  --print-alignment

E：生成步骤的历史 



## 1.2 训练一个新的模型（Training a New Model）

对IWSLT数据集进行预处理和二值化

先运行一个脚本 examples/translation/prepare-iwslt14.sh ，然后再用 fairseq-preprocess 就得到一个 data-bin/iwslt14.tokenized.de-en

脚本的内容是先下个Moses分词，再下个Subword NMT 做BPE预处理，下载数据 解压 分词 学BPE applyBPE

脚本需要一点时间运行完 很多时间 

脚本运行完得到：
```shell
$ ls iwslt14.tokenized.de-en/
code  test.de  test.en  tmp/  train.de  train.en  valid.de  valid.en
```
一句英语和德语的BPE，可以看到@@分隔符 而且标点是分开的，脚本还进行了小写 清洗标签 不合适长度的句子等。
```tsv
you know , one of the inten@@ se pleas@@ ures of travel and one of the deli@@ ghts of eth@@ no@@ graphic research is the opportunity to live am@@ on@@ gst those who have not for@@ gotten the old ways , who still feel their past in the wind , touch it in st@@ ones poli@@ shed by rain , taste it in the bit@@ ter leaves of plants .
wissen sie , eines der großen vern@@ ü@@ gen beim reisen und eine der freu@@ den bei der eth@@ no@@ graph@@ ischen forschung ist , gemeinsam mit den menschen zu leben , die sich noch an die alten tage erinnern können . die ihre vergangenheit noch immer im wind spüren , sie auf vom regen ge@@ gl@@ ä@@ t@@ teten st@@ einen berü@@ hren , sie in den bit@@ teren blä@@ ttern der pflanzen schme@@ cken .

```
fairseq-preprocess 指定train dev test 等的前缀 指定输出.  在Windows下是执行C:\install\anaconda_2\envs\p38gpu\Scripts下的fairseq-preprocess.exe这个脚本，别的命令行程序同理。

结果：

```
(p38gpu) D:\codes\fairseq-codes\fairseq>fairseq-preprocess --source-lang de --target-lang en    --trainpref examples/translation/iwslt14.tokenized.de-en/train --validpref examples/translation/iwslt14.tokenized.de-en/valid --testpref examples/translation/iwslt14.tokenized.de-en/test   --destdir data-bin/iwslt14.tokenized.de-en
  INFO | fairseq_cli.preprocess | Namespace(aim_repo=None, aim_run_hash=None, align_suffix=None, alignfile=None, all_gather_list_size=16384, amp=False, amp_batch_retries=2, amp_init_scale=128, amp_scale_window=None, azureml_logging=False, bf16=False, bpe=None, cpu=False, criterion='cross_entropy', dataset_impl='mmap', destdir='data-bin/iwslt14.tokenized.de-en', dict_only=False, empty_cache_freq=0, fp16=False, fp16_init_scale=128, fp16_no_flatten_grads=False, fp16_scale_tolerance=0.0, fp16_scale_window=None, joined_dictionary=False, log_file=None, log_format=None, log_interval=100, lr_scheduler='fixed', memory_efficient_bf16=False, memory_efficient_fp16=False, min_loss_scale=0.0001, model_parallel_size=1, no_progress_bar=False, nwordssrc=-1, nwordstgt=-1, on_cpu_convert_precision=False, only_source=False, optimizer=None, padding_factor=8, plasma_path='/tmp/plasma', profile=False, quantization_config_path=None, reset_logging=False, scoring='bleu', seed=1, simul_type=None, source_lang='de', srcdict=None, suppress_crashes=False, target_lang='en', task='translation', tensorboard_logdir=None, testpref='examples/translation/iwslt14.tokenized.de-en/test', tgtdict=None, threshold_loss_scale=None, thresholdsrc=0, thresholdtgt=0, tokenizer=None, tpu=False, trainpref='examples/translation/iwslt14.tokenized.de-en/train', use_plasma_view=False, user_dir=None, validpref='examples/translation/iwslt14.tokenized.de-en/valid', wandb_project=None, workers=1)
  INFO | fairseq_cli.preprocess | [de] Dictionary: 8848 types
  INFO | fairseq_cli.preprocess | [de] examples/translation/iwslt14.tokenized.de-en/train.de: 160250 sents, 4034263 tokens, 0.0% replaced (by <unk>)
  INFO | fairseq_cli.preprocess | [de] Dictionary: 8848 types
  INFO | fairseq_cli.preprocess | [de] examples/translation/iwslt14.tokenized.de-en/valid.de: 7284 sents, 185522 tokens, 0.0129% replaced (by <unk>)
  INFO | fairseq_cli.preprocess | [de] Dictionary: 8848 types
  INFO | fairseq_cli.preprocess | [de] examples/translation/iwslt14.tokenized.de-en/test.de: 6750 sents, 161890 tokens, 0.0642% replaced (by <unk>)
  INFO | fairseq_cli.preprocess | [en] Dictionary: 6640 types
  INFO | fairseq_cli.preprocess | [en] examples/translation/iwslt14.tokenized.de-en/train.en: 160250 sents, 3946310 tokens, 0.0% replaced (by <unk>)
  INFO | fairseq_cli.preprocess | [en] Dictionary: 6640 types
  INFO | fairseq_cli.preprocess | [en] examples/translation/iwslt14.tokenized.de-en/valid.en: 7284 sents, 182023 tokens, 0.00385% replaced (by <unk>)
  INFO | fairseq_cli.preprocess | [en] Dictionary: 6640 types
  INFO | fairseq_cli.preprocess | [en] examples/translation/iwslt14.tokenized.de-en/test.en: 6750 sents, 156957 tokens, 0.00828% replaced (by <unk>)
  INFO | fairseq_cli.preprocess | Wrote preprocessed data to data-bin/iwslt14.tokenized.de-en

```


训练，用可用的GPU 指定的话用CUDA_VISIBLE_DEVICES，batch大小根据--max-tokens（每batch最大token数量）来定！！GPU太小要调。

生成，fairseq-generate 用刚刚的二进制数据和训练好的模型。没处理过的文本用 fairseq-interactive


## 总结 

就是在学习 他的几个 Command-line Tools 怎么用  fairseq-generate  fairseq-interactive fairseq-preprocess fairseq-train 此外还有 fairseq-score 算翻译的BLEU分数 还有 fairseq-eval-lm

学会了这些怎么用，自己写的model和tasks等可以直接注册然后用这些 Command-line Tools 

所以下一步是学习user-supplied plug-ins怎么用：

具体来说，就是 模型 criterion（突然想不起来咋翻译了）  任务（存字典 dataset的迭代 初始化模型 criterion 算loss） 优化器（更新模型参数用梯度） 还有个学习率调度器 这五个插件

给出上面那五个插件，fairseq 有一个训练的flow

New plug-ins 需要用 @register 装饰器注册，可以直接写在fairseq源码下面，但是更方便的是写在用户目录然后指定路径。


## 处理数据的脚本+详细注释

```shell

#!/usr/bin/env bash
#
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh


# 下载Moses 和 Subword NMT 这两个仓库，等下要用来做分词和BPE

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git

# 定义一些变量
# SCRIPTS 是mosesdecoder库的脚本的路径 下面的tokenizer/tokenizer.perl是分词程序的路径 LC是小写 CLEAN删除长句和空语句

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl

# 刚刚下载的 subword_nmt 的位置
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=10000

# 下载数据集， 下de-en.tgz这个压缩包然后解压

URL="http://dl.fbaipublicfiles.com/fairseq/data/iwslt14/de-en.tgz"
GZ=de-en.tgz

# 检查Moses脚本路径在不在
if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

src=de
tgt=en
lang=de-en
prep=iwslt14.tokenized.de-en  #处理之后的数据的位置
tmp=$prep/tmp
orig=orig  # 下载原始数据压缩包和解压的位置

# 下载压缩包到orig目录 判断下成功没有 解压
mkdir -p $orig $tmp $prep

echo "Downloading data from ${URL}..."
cd $orig
wget "$URL"

if [ -f $GZ ]; then
    echo "Data successfully downloaded."
else
    echo "Data not successfully downloaded."
    exit
fi

tar zxvf $GZ
cd ..

# 处理 orig/de-en/train.tags.de-en.de train.tags.de-en.en 这两个文件
# grep -v 查找不匹配这个的行， -e s是替换 这里是去掉有URL什么的行，然后把title什么的标签给去掉。
#然后交给Moses的分词脚本 输出在iwslt14.tokenized.de-en/tmp文件夹下面

echo "pre-processing train data..."
for l in $src $tgt; do
    f=train.tags.$lang.$l
    tok=train.tags.$lang.tok.$l

    cat $orig/$lang/$f | \
    grep -v '<url>' | \
    grep -v '<talkid>' | \
    grep -v '<keywords>' | \
    sed -e 's/<title>//g' | \
    sed -e 's/<\/title>//g' | \
    sed -e 's/<description>//g' | \
    sed -e 's/<\/description>//g' | \
    perl $TOKENIZER -threads 8 -l $l > $tmp/$tok
    echo ""
done

# 对句子长度不合适的清洗  刚刚分词好的文件train.tags.de-en.tok.de train.tags.de-en.tok.en 放在 train.tags.de-en.clean.de xxx.en (语法参考http://www2.statmt.org/moses/?n=FactoredTraining.PrepareTraining)

perl $CLEAN -ratio 1.5 $tmp/train.tags.$lang.tok $src $tgt $tmp/train.tags.$lang.clean 1 175

# 小写 输入是刚刚Clean的文件 输出到train.tags.de-en.de train.tags.de-en.en
for l in $src $tgt; do
    perl $LC < $tmp/train.tags.$lang.clean.$l > $tmp/train.tags.$lang.$l
done

# 处理orig下面的原始的训练集测试集，然后只分词和小写 不去掉句子
# 有处理seg标签和中文单引号 （因为测试的数据集是这种格式的训练集没有）
# 输出给 temp/IWSLT14.TED*.en de

echo "pre-processing valid/test data..."
for l in $src $tgt; do
    for o in `ls $orig/$lang/IWSLT14.TED*.$l.xml`; do
    fname=${o##*/}
    f=$tmp/${fname%.*}
    echo $o $f
    grep '<seg id' $o | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" | \
    perl $TOKENIZER -threads 8 -l $l | \
    perl $LC > $f
    echo ""
    done
done

# 划分训练集测试集验证集
# train.tags.de-en.XX 1:22划分出来valid和train
# 剩下的原始数据 valid/test 的那些都给test
# 5*2+2 12个文件输出到 3*2 6个文件

echo "creating train, valid, test..."
for l in $src $tgt; do
    awk '{if (NR%23 == 0)  print $0; }' $tmp/train.tags.de-en.$l > $tmp/valid.$l
    awk '{if (NR%23 != 0)  print $0; }' $tmp/train.tags.de-en.$l > $tmp/train.$l

    cat $tmp/IWSLT14.TED.dev2010.de-en.$l \
        $tmp/IWSLT14.TEDX.dev2012.de-en.$l \
        $tmp/IWSLT14.TED.tst2010.de-en.$l \
        $tmp/IWSLT14.TED.tst2011.de-en.$l \
        $tmp/IWSLT14.TED.tst2012.de-en.$l \
        > $tmp/test.$l
done

TRAIN=$tmp/train.en-de  # 训练集德语英语都输出在这里
BPE_CODE=$prep/code  # $prep是数据输出的目录
rm -f $TRAIN
for l in $src $tgt; do
    cat $tmp/train.$l >> $TRAIN
done

echo "learn_bpe.py on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE  # $BPE_TOKENS是10000，刚刚处理好的训练集学BPE放在 iwslt14.tokenized.de-en/code

# 对训练 验证 测试 *2 6个文件apply BPE 放在iwslt14.tokenized.de-en/
for L in $src $tgt; do
    for f in train.$L valid.$L test.$L; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $prep/$f
    done
done
```




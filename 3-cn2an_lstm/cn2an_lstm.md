# 三、cn2an lstm

之前都是文档的例子，现在根据xiaofei的练习代码生成简单数据。

## 1 生成简单数据

1-generate-simple-data.py 产生数据写入easy-dataset/raw_data，具体做法是生成1W个随机六位数，去掉重复的，用cn2an这个包得到对应的汉字，然后手动每个字符加上空格写入文件。

```shell
$ ls easy-dataset/raw_data/
test.in  test.out  train.in  train.out  valid.in  valid.out

```

产生六个文件：

``` 
└───easy-dataset
    └───raw_data
            test.in
            test.out
            train.in
            train.out
            valid.in
            valid.out
```
数据：
```
test.in: 7 2 0 9 2 4
test.out: 七 十 二 万 零 九 百 二 十 四

```
fairseq-preprocess --source-lang in --target-lang out   --trainpref easy-dataset/raw_data/train --validpref easy-dataset/raw_data/valid --testpref easy-dataset/raw_data/test   --destdir easy-dataset/preprocessed

用fairseq的脚本进行预处理，命令写在2-process.sh。这个脚本处理的文件应该是 train or test 这样的前缀（--xxxpref指定）  +点+ in or out(表示源或者目标语言 --xxxlang 指定)
两个词典一个log加上3*4个二进制文件
``` 
───easy-dataset
    ├───preprocessed
    │       dict.in.txt
    │       dict.out.txt
    │       preprocess.log
    │       test.in-out.in.bin
    │       test.in-out.in.idx
    │       test.in-out.out.bin
    │       test.in-out.out.idx
    │       train.in-out.in.bin
    │       train.in-out.in.idx
    │       train.in-out.out.bin
    │       train.in-out.out.idx
    │       valid.in-out.in.bin
    │       valid.in-out.in.idx
    │       valid.in-out.out.bin
    │       valid.in-out.out.idx
```
dict文件：
```
5 4931
9 4909
6 4891
2 4876
8 4873
4 4868
3 4823
7 4760
1 4737
0 3930
madeupword0000 0
madeupword0001 0
```
第二列数字表示出现的次数。

## 构造模型 注册模型

模型和注册模型的代码在 3-fq-model/__init__.py，跟教程中的一样。

同前有个bug pack_padded_sequence的第二个参数序列长度要放在CPU，似乎torch 1.5之后会有这个bug.

## train

4-train.sh,`但是这里因为把代码没写在fairseq目录下面所以用命令行程序train的时候要加一个flag --user-dir

```shell
fairseq-train easy-dataset/preprocessed  --user-dir ./my_fairseq_module --arch tutorial_simple_lstm  --encoder-dropout 0.2 --decoder-dropout 0.2  --optimizer adam --lr 0.005 --lr-shrink 0.5   --max-tokens 12000  --max-epoch 3   --source-lang in   --target-lang out

fairseq-train easy-dataset/preprocessed  --user-dir 3-fq-model --arch an2cn_simple_lstm  --encoder-dropout 0.2 --decoder-dropout 0.2  --optimizer adam --lr 0.005 --lr-shrink 0.5   --max-tokens 12000  --max-epoch 3   --source-lang in   --target-lang out
```

模型存在\checkpoints

# todo 这个命令Windows找不到这个相对目录

## 评估

fairseq-generate easy-dataset/preprocessed  --path checkpoints/checkpoint_best.pt  --batch-size 128 --beam 5  

```shell

S-990   5 2 4 1 8 5
T-990   五 十 二 万 四 千 一 百 八 十 五
H-990   -0.9469358325004578     五 十 一 万 二 千 五 百 四 十 五
D-990   -0.9469358325004578     五 十 一 万 二 千 五 百 四 十 五
P-990   -1.5801 -0.0017 -2.3802 -0.0025 -2.0811 -0.0289 -1.0876 -0.0176 -2.1604 -0.0141 -2.0000 -0.0090
S-992   8 5 1 9 1 9
T-992   八 十 五 万 一 千 九 百 一 十 九
H-992   -0.7361486554145813     九 十 一 万 九 千 九 百 九 十 一
D-992   -0.7361486554145813     九 十 一 万 九 千 九 百 九 十 一
P-992   -1.9370 -0.0029 -1.5274 -0.0053 -1.6530 -0.0182 -1.0301 -0.0420 -0.8760 -0.0386 -1.6965 -0.0068
  INFO | fairseq_cli.generate | NOTE: hypothesis and token scores are output in base 2
  INFO | fairseq_cli.generate | Translated 996 sentences (11,752 tokens) in 1.5s (683.80 sentences/s, 8068.32 tokens/s)
Generate test with beam=5: BLEU4 = 23.77, 81.5/31.7/16.7/7.4 (BP=1.000, ratio=1.030, syslen=10756, reflen=10441)
```

## 交互

fairseq-interactive --path checkpoints/checkpoint_best.pt easy-dataset/preprocessed --beam 5

## 总结

使用一个小小的英文-中文数字转换数据，使用注册的LSTM模型训练、评估、交互式的生成一下。

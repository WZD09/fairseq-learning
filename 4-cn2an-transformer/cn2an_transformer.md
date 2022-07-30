# 四、 cn2an Transformer

## 生成数据

### 1-gemerate-data-no-space.py, 不加空格
``` 
test.in: 688147
test.out: 六十八万八千一百四十七
```
### 2-tokenize.py 处理成json格式
``` 
 ───easy-dataset
    ├───json_data
    │       test.json
    │       train.json
    │       valid.json

```
test.json:
``` 
[
  {
    "src": [
      "6",
      "8",
      "8",
      "1",
      "4",
      "7"
    ],
    "tgt": [
      "六",
      "十",
      "八",
      "万",
      "八",
      "千",
      "一",
      "百",
      "四",
      "十",
      "七"
    ]
  },
  {...
  } ,
  ...
 ]
```
### 3-add-index.py 生成tokenid和dict

用了自己的task的dictionary的encode_line方法。

``` 
[
  {
    "src": [
      "6",
      "8",
      "8",
      "1",
      "4",
      "7"
    ],
    "tgt": [
      "六",
      "十",
      "八",
      "万",
      "八",
      "千",
      "一",
      "百",
      "四",
      "十",
      "七"
    ],
    "src_ids": [
      9,
      11,
      11,
      8,
      10,
      6,
      2
    ],
    "tgt_ids": [
      12,
      4,
      14,
      5,
      14,
      7,
      16,
      6,
      13,
      4,
      10,
      2
    ]
  },
```

## 4 继承Transformer模型并且注册

4-my-fq-module这个文件夹

my_small_transformer 这个 arch my_translation这个task

## 5 训练和生成

``` shell
fairseq-train  easy-dataset/preprocessed --user-dir  my_fq_module --task my_translation   --arch my_small_transformer  --optimizer adam  --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000   --max-tokens 12000 --max-epoch 200 
fairseq-generate easy-dataset/preprocessed --path checkpoints/checkpoint_best.pt    --task my_translation   --batch-size 128 --beam 5  
```


``` 
-908    2 2 7 4 2 5
T-908   二 十 二 万 七 千 四 百 二 十 五
H-908   -1.1956623792648315     二 十 二 万 二 千 二 百 二 十 二
D-908   -1.1956623792648315     二 十 二 万 二 千 二 百 二 十 二
P-908   -0.6004 -0.5432 -0.6230 -1.8822 -0.9893 -2.0649 -0.5785 -2.3122 -0.9195 -1.4486 -0.8687 -1.5174
S-37    7 8 7 9 5 2
T-37    七 十 八 万 七 千 九 百 五 十 二
H-37    -1.5420867204666138     七 十 万 七 千 七 千 七 千 七
D-37    -1.5420867204666138     七 十 万 七 千 七 千 七 千 七
P-37    -1.4143 -0.6463 -1.5340 -1.7040 -0.9766 -1.6902 -1.1405 -1.8691 -1.8665 -2.1073 -2.0142
Generate test with beam=5: BLEU4 = 17.29, 66.8/27.7/12.8/4.6 (BP=0.954, ratio=0.955, syslen=9802, reflen=10261)

```

## 7 demo

需要复制 4-cn2an-transformer/easy-dataset/preprocessed 下面的两个dict文件。

## 遇到的问题

不知道为啥缺一些命令行参数 对这个新写的task 可能是版本不同(我猜)，手动加上这些参数问题解决！具体：

- 首先会报错 我们指定的data的目录无法解析，发现是命令行参数没有data这一项，于是手动在 MyTranslationTask 类加一个添加命令行参数的成员函数，但是又有一些别的参数出现问题，在translate.py中被使用到但是没有这个参数。

  于是继续手动加入了这些找不到的参数，默认都给他整成False，对任务没有影响。

```python
@staticmethod
def add_args(parser):
    # Add some command-line arguments for specifying where the data is
    # located and the maximum supported input length.
    parser.add_argument('data', metavar='FILE',
                        help='file prefix for data')
    parser.add_argument('--max_source_positions', default=1024, type=int,
                        help='max input length')
    parser.add_argument('--max_target_positions', default=1024, type=int,
                        help='max input length')
    parser.add_argument('--eval_bleu', default=False, type=bool,
                        help='max input length')
    parser.add_argument('--left_pad_source', default=False, type=bool,
                        help='max input length')
    parser.add_argument('--left_pad_target', default=False, type=bool,
                        help='max input length')


```


**更新：真正解决这个问题应该是加上config** 

```python
from fairseq.tasks.translation import TranslationConfig, TranslationTask
@register_task("my_translation", dataclass=TranslationConfig)
class MyTranslationTask(TranslationTask):
    # 这里就不用写add_arg()了
```

看trian.py cli_main() 命令行参数怎么添加的，是在fairseq.options parse_args_and_arch() 里面，我们--tasks指定了task 就会给parser添加这个tasks额外添加的参数。

另外arch也会有用户自己添加的参数，同理


还有实现tasks时 self.args.data 改成 self.cfg.data.

这个应该跟版本有关，新版本的Fairseq的FairseqTask改用了XXXConfig初始化，所以self.args这种参数都不能用了要用self.cfg

官方给的教程用的是self.args因为继承的 LegacyFairseqTask 这个类没有用 FairseqTask 的初始化，自己写了args的初始化。

# 二、 lstm 翻译

总结：extending一个自己的模型的流程就是 继承模型 注册模型 注册arch也就是config

这个教程讲了怎么用fairseq写自己的模型然后注册，注册好的新插件就能在命令行直接用了。这个教程是直接在 fairseq/models 目录下面写自己的模型，就不需要指定位置。可以用--user-dir指定自定义位置。

## 数据

数据：这个教程还是用了上一个教程处理好的 data-bin下面的iwslt14德语英语翻译数据集。

## 写编码器和解码器

SimpleLSTMEncoder 和 SimpleLSTMDncoder，分别继承了 FairseqEncoder FairseqDecoder

## 注册模型

定义了编码器和解码器，我们必须使用函数装饰器将我们的模型注册到 fairseq，注册以后爱能用命令行程序。

要注册一个模型 必须继承 BaseFairseqModel 这个基类，对于这个翻译任务可以继承 FairseqEncoderDecoderModel.

现在我们要注册的模型有几个类方法：
    
    重载 add_args() 可以添加新的命令行参数
    build_model() 是fairseq初始化模型的方法，这是很灵活的因为返回的模型实例可以不是我们call的这个类型
    forward() 这里没用重载，因为 FairseqEncoderDecoderModel 有一个简单的forward。

写好wrapper class（SimpleLSTMModel）以后用 @register_model(注册名字) 注册

最后要用register_model_architecture() 函数装饰器定义新的模型架构（architecture），第一个参数是我们刚刚注册的模型 第二个是architecture的名字，这样注册好以后这个模型架构直接 在 --arch 命令行参数里面指定就能用了。
（就是使用模型的配置定义一个命名架构 说是arch就是config吧）
刚刚的 SimpleLSTMModel 里面加了一些命令行参数，可以指定embed维度 编码器解码器hidden维度，这些没写默认参数，dropout有些默认参数。这个教程配置里面就是给这些没写默认参数的用getattr()写一点默认的参数。

## 训练

写完可以直接用命令行train或者生成一下

## 用增量解码加快解码速度

自回归的生成文本是很慢的，用 caching the previous hidden states 的方法来改进。这个方法叫做  Incremental decoding， 具体做法是让Decoder现在继承FairseqIncrementalDecoder这个类，这样forward函数多了一个参数incremental_state。

用增量解码确实有快很多！


## 一个bug:

可能是torch版本问题

```shell
  File "d:\codes\fairseq-codes\fairseq\fairseq\models\simple_lstm.py", line 56, in forward
    x = nn.utils.rnn.pack_padded_sequence(x, src_lengths, batch_first=True)
  File "C:\install\anaconda_2\envs\p38gpu\lib\site-packages\torch\nn\utils\rnn.py", line 249, in pack_padded_sequence
    _VF._pack_padded_sequence(input, lengths, batch_first)
RuntimeError: 'lengths' argument should be a 1D CPU int64 tensor, but got 1D cuda:0 Long tensor
```

models\simple_lstm.py pack_padded_sequence() 的第二个参数 lengths 放在CPU上解决



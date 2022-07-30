# decoder

from fairseq.models.transformer import TransformerDecoder


class MyTransformerDecoder(TransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super(MyTransformerDecoder, self).__init__(args, dictionary, embed_tokens, no_encoder_attn)
        print("*****************************Decoder!!!*****************************")


# encoder
from fairseq.models.transformer import TransformerEncoder


class MyTransformerEncoder(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super(MyTransformerEncoder, self).__init__(args, dictionary, embed_tokens)
        print("*****************************Encoder!!!*****************************")


# model

from fairseq.models import register_model
from fairseq.models.transformer import TransformerModel


@register_model("my_transformer")
class MyTransformerModel(TransformerModel):
    def __init__(self, args, encoder, decoder):
        super(MyTransformerModel, self).__init__(args, encoder, decoder)
        print("*****************************Transformer!!!*****************************")

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return MyTransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return MyTransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )

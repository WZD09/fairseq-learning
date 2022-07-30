from fairseq.hub_utils import GeneratorHubInterface

from my_fq_module.model.my_transformer_model import MyTransformerModel


def run():
    model: GeneratorHubInterface = MyTransformerModel.from_pretrained(
        model_name_or_path="checkpoints",
        checkpoint_file="checkpoint_best.pt",
        tokenizer=None,
    )
    print(type(model))

    while True:
        sentence = input('\nInput: ')
        translation = model.translate([" ".join(sentence)])
        print(translation)


if __name__ == '__main__':
    run()

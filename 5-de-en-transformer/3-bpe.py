import json
from tqdm import tqdm


def run():
    for split in ["train", "valid", "test"]:
        with open(f"iwslt14/bpe_text/{split}.de") as src_input_file:
            src_data = src_input_file.readlines()
        with open(f"iwslt14/bpe_text/{split}.en") as tgt_input_file:
            tgt_data = tgt_input_file.readlines()
        data = []
        for s, t in tqdm(zip(src_data, tgt_data)):
            data.append({
                "src": s,
                "tgt": t,

            })
        with open(f"iwslt14/bpe/{split}.json", "w") as output_file:
            json.dump(data, output_file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    run()

import cn2an
import numpy


def generate_cn2an_data():
    # 生成1W个随机数
    number_list = numpy.random.randint(100000, 1000000 - 1, 10000)
    # 去掉重复的
    number_list = list(set(number_list))
    print(len(number_list), "个数字")
    dataset = {
        "train": [[], []],
        "valid": [[], []],
        "test": [[], []],
    }
    chinese_number_list = [cn2an.an2cn(str(num)) for num in number_list]
    for i in range(len(number_list)):
        p = numpy.random.random()
        if p < 0.8:
            dataset_type = "train"
        elif p < 0.9:
            dataset_type = "valid"
        else:
            dataset_type = "test"
        dataset[dataset_type][0].append(str(number_list[i]))
        dataset[dataset_type][1].append(chinese_number_list[i])
    # 8：1：1划分训练集 测试集 验证集，英语写入in 中文写入out后缀的文件
    for dataset_type in dataset:
        with open(f"easy-dataset/raw_data/{dataset_type}.in", "w") as f:
            f.write("\n".join(dataset[dataset_type][0]))
        with open(f"easy-dataset/raw_data/{dataset_type}.out", "w") as f:
            f.write("\n".join(dataset[dataset_type][1]))


if __name__ == '__main__':
    generate_cn2an_data()

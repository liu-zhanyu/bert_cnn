import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset


# 查询长度：txt
def count_text_len():
    text_len = []
    with open('./data/input/train.txt') as f:
        for line in f.readlines():
            # 这个可以打印 line 具体符号，eg："\t"
            # 方便确定下文 split 具体用什么符号划分
            # print(line.split("a"))  # ['中华女子学院：本科层次仅1专业招男生\t3\n']
            # exit()

            # 去掉 line 两边空余
            # text, _ = line.strip().split("\t")
            text, _ = line.split('\t')
            text_len.append(len(text))
    plt.hist(text_len)
    plt.show()
    print(max(text_len))


# 查询长度：csv
def count_csv_len():
    data = pd.read_csv('./data/input/train.csv')
    # print(data.head())
    # 注意 map、apply、applymap 三者区别
    x_len = data['text'].map(lambda x: len(x))
    # x_len 是一个 series
    # print(type(x_len))
    x_len.plot(kind='hist')
    plt.show()
    print(x_len.max())


# loadset 方法
def count_len():
    data = load_dataset(
        'csv',
        data_files='./data/input/train.csv',
        split='train'
    )

    def f(data):
        # 创建新的一列
        data['len'] = len(data['text'])
        return data

    data_len = data.map(f)
    print(max(data_len['len']), min(data_len['len']))


if __name__ == '__main__':
    # count_text_len()
    # count_csv_len()
    count_len()

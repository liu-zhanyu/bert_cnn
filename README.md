「文本分类」是 NLP 最基础、最重要的任务之一，非常适合作为 NLP 入门的第一个项目。本文使用了清华大学的「THUCNews」新闻文本分类数据集，训练集 18w，验证集 1w，测试集 1w，有 10 个标签类别：金融、房产、股票、教育、科学、社会、政治、体育、游戏、娱乐，样本示例如下。本文打算用「BERT + TextCNN」作为 Baseline，先解释基础理论储备，然后给出基线代码，算是抛砖引玉。
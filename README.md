# TFX Implementation of Attention is All You Need
A TFX implementation of the paper on transformers, [Attention is All You Need](https://arxiv.org/abs/1706.03762)

## To Understand Transformers
I will be writing a blog post on this soon to explain what Transformers themselves do. In this example, I have followed the paper exactly and haven't used teacher forcing during training. Anyone looking for an example with teacher forcing should try and consider the official [Tensorflow Guide on Transformers](https://www.tensorflow.org/text/tutorials/transformer)

There are also some amazing resources I found on [Transformers](https://jalammar.github.io/illustrated-transformer/) and [Attention in general](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html#:~:text=The%20attention%20mechanism%20was%20born%20to%20help%20memorize%20long%20source,and%20the%20entire%20source%20input.). These two should suffice for an overview of the whole concept.

## To Understand TFX Pipelines
I have not described the functioning of the pipeline here, but anyone who might be inclined to understand it can refer an older repository I built: [example-tfx-pipeline-text-classifier](https://github.com/microcoder-py/example-tfx-pipeline-text-classifier)

It is an example on text classification, but the difference between that and Neural Machine Translation should be apparent once you start reading the code carefully. Wherever needed, I have added additional comments in the code for better understanding, feel free to drop in questions if any

## CITATION

```citation
@misc{vaswani2017attention,
      title={Attention Is All You Need}, 
      author={Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and Llion Jones and Aidan N. Gomez and Lukasz Kaiser and Illia Polosukhin},
      year={2017},
      eprint={1706.03762},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

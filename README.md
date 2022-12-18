# ML_2022_NLP
Sentiment Analysis for Customer Service

## About
NLP classifier based on FastText approach ([Link](https://arxiv.org/abs/1612.03651))

## Usage

Firstly you need to specify the model. Model can be found by the following [link](https://drive.google.com/file/d/1gHuam8iRaGZZJ3jwAGrHy6E40a8_k8yT/view?usp=sharing).

1. Download the model
2. Load model into python script `classifier.py your/path/to/model.bin`

Then you are able to use the classifier.

```console
classifier.py [-h] [-f [path]] [text]
```

You can directly pass the text to classify or specify file of texts.
But not both simultaneously.

All texts in file should be separated with '\n'

In case of error like `error: unrecognized arguments: text` enclose your text with double quotes.

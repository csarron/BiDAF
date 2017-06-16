# Bi-directional Attention Flow for Machine Comprehension
 
- This the original implementation of [Bi-directional Attention Flow for Machine Comprehension][paper] (Seo et al., 2016).
- This is tensorflow v1.1.0 comaptible version. This is not compatible with previous trained models, 
so if you want to use them, go to [v0.2.1][v0.2.1]. 
- The CodaLab worksheet for the [SQuAD Leaderboard][squad] submission is available [here][worksheet].
- Please contact [Minjoon Seo][minjoon] ([@seominjoon][minjoon-github]) for questions and suggestions.

## 0. Requirements
#### General
- Python (developed on 3.5.2. Issues have been reported with Python 2!)
- unzip

#### Python Packages
- tensorflow (deep learning library, verified on 1.1.0)
- nltk (NLP tools, verified on 3.2.1)
- tqdm (progress bar, verified on 4.7.4)
- jinja2 (for visaulization; if you only train and test, not needed)

## 1. Pre-processing
Donwload GloVe, nltk corpus and SQuAD data (~850 MB, this will download files to `$PWD/data/` in not downloaded).
Preprocess Stanford QA dataset (along with GloVe vectors) and save them in `$PWD/data` (~5 minutes):
:
```
python3 prepare.py
```

## 2. Training
The model was trained with NVidia Titan X (Pascal Architecture, 2016).
The model requires at least 12GB of GPU RAM.
If your GPU RAM is smaller than 12GB, you can either decrease batch size (performance might degrade),
or you can use multi GPU (see below).
The training converges at ~18k steps, and it took ~4s per step (i.e. ~20 hours).

Before training, it is recommended to first try the following code to verify everything is okay and memory is sufficient:
```
python main.py --mode train --noload --debug
```

Then to fully train, run:
```
python main.py --mode train --noload
```

You can speed up the training process with optimization flags:
```
python main.py --mode train --noload --len_opt --cluster
```
You can still omit them, but training will be much slower.


## 3. Test
To test, run:
```
python main.py
```

Similarly to training, you can give the optimization flags to speed up test (5 minutes on dev data):
```
python main.py --len_opt --cluster
```

This command loads the most recently saved model during training and begins testing on the test data.
After the process ends, it prints F1 and EM scores, and also outputs a json file (`$PWD/out/basic/00/answer/test-####.json`,
where `####` is the step # that the model was saved).
Note that the printed scores are not official (our scoring scheme is a bit harsher).
To obtain the official number, use the official evaluator (copied in `squad` folder) and the output json file:

```
python squad/evaluate.py $HOME/data/dev-v1.1.json out/basic/00/answer/test-####.json
```

### 3.1 Loading from pre-trained weights

## Results

### Dev Data (using provided pre-trained weights)

|          | EM (%) | F1 (%) |
| -------- |:------:|:------:|
| single   | 67.9   | 77.3   |

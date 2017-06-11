#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import glob
import json
import nltk
import os
import sys
import urllib.request
import zipfile
from collections import Counter
from shutil import copyfile
from tqdm import tqdm
from utils import get_word_span, get_word_idx, process_tokens, word_tokenize, prettify_json

data_path = "data"
data_version = "v1.1"
squad_base_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"
glove_base_url = "http://nlp.stanford.edu/data/"
glove_6b = "glove.6B.zip"
train = "train-{}.json".format(data_version)
dev = "dev-{}.json".format(data_version)
nltk_path = os.path.join(data_path, "nltk")
nltk.data.path.append(os.path.abspath(nltk_path))


def _dl_progress(count, block_size, total_size):
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r%3d%%" % percent)
    sys.stdout.flush()


def maybe_download_data():
    print("preparing data...")

    for data_name, base_url in zip([train, dev, glove_6b], [squad_base_url, squad_base_url, glove_base_url]):
        data_file = os.path.join(data_path, data_name)
        if os.path.exists(data_file):
            print("{} already downloaded".format(data_name))
            continue
        else:
            print("Downloading {} dataset...".format(data_name))
            if not os.path.exists(data_path):
                os.mkdir(data_path)
            urllib.request.urlretrieve(os.path.join(base_url, data_name), data_file, reporthook=_dl_progress)
            print("\n{} downloaded".format(data_name))
            prettify_json(data_file)

    glove_txt_files = glob.glob(os.path.join(data_path, "glove.6B.*.txt"))
    if not glove_txt_files:
        glove_6b_file = os.path.join(data_path, glove_6b)
        with zipfile.ZipFile(glove_6b_file, "r") as zip_ref:
            zip_ref.extractall(data_path)
            print("{} extracted".format(glove_6b))
    else:
        print("{} already extracted to: {}".format(glove_6b, glove_txt_files))

    if not os.path.exists(nltk_path):
        nltk.download("punkt", download_dir=nltk_path)


def save_data(target_dir, data, shared, data_type):
    data_save_path = os.path.join(target_dir, "data_{}.json".format(data_type))
    shared_path = os.path.join(target_dir, "shared_{}.json".format(data_type))
    json.dump(data, open(data_save_path, 'w'))
    json.dump(shared, open(shared_path, 'w'))


def get_word2vec(args, word_counter):
    glove_path = os.path.join(args.glove_dir, "glove.{}.{}d.txt".format(args.glove_corpus, args.glove_vec_size))
    sizes = {'6B': int(4e5), '42B': int(1.9e6), '840B': int(2.2e6), '2B': int(1.2e6)}
    total = sizes[args.glove_corpus]
    word2vec_dict = {}
    with open(glove_path, 'r', encoding='utf-8') as fh:
        for line in tqdm(fh, total=total):
            array = line.lstrip().rstrip().split(" ")
            word = array[0]
            vector = list(map(float, array[1:]))
            if word in word_counter:
                word2vec_dict[word] = vector
            elif word.capitalize() in word_counter:
                word2vec_dict[word.capitalize()] = vector
            elif word.lower() in word_counter:
                word2vec_dict[word.lower()] = vector
            elif word.upper() in word_counter:
                word2vec_dict[word.upper()] = vector

    print("{}/{} of word vocab have corresponding vectors in {}".format(len(word2vec_dict), len(word_counter),
                                                                        glove_path))
    return word2vec_dict


def prepare_each(args, data_type, start_ratio=0.0, stop_ratio=1.0, out_name="default", in_path=None):
    # if not args.split:
    #     sent_tokenize = lambda para: [para]

    source_path = in_path or os.path.join(data_path, data_type)
    source_data = json.load(open(source_path, 'r'))

    q, cq, y, rx, rcx, ids, idxs = [], [], [], [], [], [], []
    na = []
    cy = []
    x, cx = [], []
    answerss = []
    p = []
    word_counter, char_counter, lower_word_counter = Counter(), Counter(), Counter()
    start_ai = int(round(len(source_data['data']) * start_ratio))
    stop_ai = int(round(len(source_data['data']) * stop_ratio))
    for ai, article in enumerate(tqdm(source_data['data'][start_ai:stop_ai])):
        xp, cxp = [], []
        pp = []
        x.append(xp)
        cx.append(cxp)
        p.append(pp)
        for pi, para in enumerate(article['paragraphs']):
            # wordss
            context = para['context']
            context = context.replace("''", '" ')
            context = context.replace("``", '" ')
            xi = list(map(word_tokenize, nltk.sent_tokenize(context)))
            xi = [process_tokens(tokens) for tokens in xi]  # process tokens
            # given xi, add chars
            cxi = [[list(xijk) for xijk in xij] for xij in xi]
            xp.append(xi)
            cxp.append(cxi)
            pp.append(context)

            for xij in xi:
                for xijk in xij:
                    word_counter[xijk] += len(para['qas'])
                    lower_word_counter[xijk.lower()] += len(para['qas'])
                    for xijkl in xijk:
                        char_counter[xijkl] += len(para['qas'])

            rxi = [ai, pi]
            assert len(x) - 1 == ai
            assert len(x[ai]) - 1 == pi
            for qa in para['qas']:
                # get words
                qi = word_tokenize(qa['question'])
                qi = process_tokens(qi)
                cqi = [list(qij) for qij in qi]
                yi = []
                cyi = []
                answers = []
                for answer in qa['answers']:
                    answer_text = answer['text']
                    answers.append(answer_text)
                    answer_start = answer['answer_start']
                    answer_stop = answer_start + len(answer_text)
                    # TODO : put some function that gives word_start, word_stop here
                    yi0, yi1 = get_word_span(context, xi, answer_start, answer_stop)
                    # yi0 = answer['answer_word_start'] or [0, 0]
                    # yi1 = answer['answer_word_stop'] or [0, 1]
                    assert len(xi[yi0[0]]) > yi0[1]
                    assert len(xi[yi1[0]]) >= yi1[1]
                    w0 = xi[yi0[0]][yi0[1]]
                    w1 = xi[yi1[0]][yi1[1] - 1]
                    i0 = get_word_idx(context, xi, yi0)
                    i1 = get_word_idx(context, xi, (yi1[0], yi1[1] - 1))
                    cyi0 = answer_start - i0
                    cyi1 = answer_stop - i1 - 1
                    # print(answer_text, w0[cyi0:], w1[:cyi1+1])
                    assert answer_text[0] == w0[cyi0], (answer_text, w0, cyi0)
                    assert answer_text[-1] == w1[cyi1]
                    assert cyi0 < 32, (answer_text, w0)
                    assert cyi1 < 32, (answer_text, w1)

                    yi.append([yi0, yi1])
                    cyi.append([cyi0, cyi1])

                if len(qa['answers']) == 0:
                    yi.append([(0, 0), (0, 1)])
                    cyi.append([0, 1])
                    na.append(True)
                else:
                    na.append(False)

                for qij in qi:
                    word_counter[qij] += 1
                    lower_word_counter[qij.lower()] += 1
                    for qijk in qij:
                        char_counter[qijk] += 1

                q.append(qi)
                cq.append(cqi)
                y.append(yi)
                cy.append(cyi)
                rx.append(rxi)
                rcx.append(rxi)
                ids.append(qa['id'])
                idxs.append(len(idxs))
                answerss.append(answers)

        if args.debug:
            break

    word2vec_dict = get_word2vec(args, word_counter)
    lower_word2vec_dict = get_word2vec(args, lower_word_counter)

    # add context here
    data = {'q': q, 'cq': cq, 'y': y, '*x': rx, '*cx': rcx, 'cy': cy,
            'idxs': idxs, 'ids': ids, 'answerss': answerss, '*p': rx, 'na': na}
    shared = {'x': x, 'cx': cx, 'p': p,
              'word_counter': word_counter, 'char_counter': char_counter, 'lower_word_counter': lower_word_counter,
              'word2vec': word2vec_dict, 'lower_word2vec': lower_word2vec_dict}

    print("saving ...")
    save_data(args.target_dir, data, shared, out_name)


def prepare_data(args):
    if args.mode == 'full':
        prepare_each(args, train, out_name='train')
        prepare_each(args, dev, out_name='dev')
        copyfile('data/data_dev.json', 'test')
    else:
        prepare_each(args, train, 0.0, args.train_ratio, out_name='train')
        prepare_each(args, train, args.train_ratio, 1.0, out_name='dev')
        prepare_each(args, dev, out_name='test')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="full", type=str)
    parser.add_argument('-s', "--source_dir", default=data_path)
    parser.add_argument('-t', "--target_dir", default=data_path)
    parser.add_argument("--train_name", default=train)
    parser.add_argument('-d', "--debug", action='store_true')
    parser.add_argument("--train_ratio", default=0.9, type=int)
    parser.add_argument("--glove_corpus", default="6B")
    parser.add_argument("--glove_dir", default=data_path)
    parser.add_argument("--glove_vec_size", default=100, type=int)

    return parser.parse_args()


def prettify_json_files():
    for f in glob.glob(os.path.join(data_path, "*.json")):
        prettify_json(f)

if __name__ == '__main__':
    maybe_download_data()
    # prepare_data(get_args())
    print("all data prepared\n")
    prettify_json_files()

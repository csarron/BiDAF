#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import json
import os
import sys
import urllib.request
import zipfile


def _dl_progress(count, block_size, total_size):
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r%3d%%" % percent)
    sys.stdout.flush()


def maybe_download_data():
    print("preparing data...")
    data_path = "data"
    data_version = "v1.1"
    squad_base_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"
    glove_base_url = "http://nlp.stanford.edu/data/"
    train = "train-{}.json".format(data_version)
    dev = "dev-{}.json".format(data_version)
    glove_6b = "glove.6B.zip"

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
            pretty_json(data_file)

    glove_txt_files = glob.glob(os.path.join(data_path, "glove.6B.*.txt"))
    if not glove_txt_files:
        glove_6b_file = os.path.join(data_path, glove_6b)
        with zipfile.ZipFile(glove_6b_file, "r") as zip_ref:
            zip_ref.extractall(data_path)
            print("{} extracted".format(glove_6b))
    else:
        print("{} already extracted to: {}".format(glove_6b, glove_txt_files))
    print("all data prepared\n")


def pretty_json(f):
    if not f.endswith(".json"):
        return
    print("prettifying : {}".format(f))
    parsed = json.load(open(f, 'r'))
    pretty_path = "{}.txt".format(f)

    with open(pretty_path, 'w') as p:
        p.write(json.dumps(parsed, indent=2))
        print("saved to : {}\n".format(pretty_path))


def prepare_data():
    pass


if __name__ == '__main__':
    maybe_download_data()

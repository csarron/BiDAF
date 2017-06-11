import glob
import json
import os
import sys
import urllib.request


def _dl_progress(count, block_size, total_size):
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r%3d%%" % percent)
    sys.stdout.flush()


def maybe_download_data():
    print("")
    data_path = "data"
    base_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"
    data_version = "v1.1"
    train = "train-{}.json".format(data_version)
    dev = "dev-{}.json".format(data_version)

    train_file = os.path.join(data_path, train)
    dev_file = os.path.join(data_path, dev)

    if os.path.exists(train_file) and os.path.exists(dev_file):
        print("train and dev data already downloaded")
        print("")
        return
    else:
        print("Downloading SQuAD Dataset...")
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        urllib.request.urlretrieve(os.path.join(base_url, train), train_file, reporthook=_dl_progress)
        print("\n{} downloaded".format(train))
        pretty_json(train_file)
        urllib.request.urlretrieve(os.path.join(base_url, dev), dev_file, reporthook=_dl_progress)
        print("\n{} downloaded".format(dev))
        pretty_json(dev_file)

    print("")


def pretty_json(f):
    print("pretty : {}".format(f))
    parsed = json.load(open(f, 'r'))
    pretty_path = "{}.txt".format(f)

    with open(pretty_path, 'w') as p:
        p.write(json.dumps(parsed, indent=2))
        print("saved to : {}".format(pretty_path))


def prepare_data():
    pass

if __name__ == '__main__':
    maybe_download_data()

import json
import random
from my.utils import prettify_json
import os


def extract_json(input_json, size=100, seed=0):

    if not input_json.endswith(".json"):
        print("{} is not json file".format(input_json))
        return
    json_data = json.load(open(input_json, 'r'))

    article_size = len(json_data['data'])
    paragraph_size = 0
    question_size = 0
    for article_index, article in enumerate(json_data['data']):
        paragraph_size += len(article["paragraphs"])
        for paragraph_index, paragraph in enumerate(article["paragraphs"]):
            question_size += len(paragraph["qas"])
    print("total articles: {}".format(article_size))
    print("total paragraphs: {}".format(paragraph_size))
    print("total questions: {}".format(question_size))

    random.seed(seed)
    selected_indices = [random.randint(0, question_size) for _ in range(0, size)]
    print("selected_indices: {}".format(selected_indices))

    question_index = 0
    selected_data = {}
    articles = []
    for article_index, article in enumerate(json_data['data']):
        # print("article index:{}, title:{}".format(article_index, article["title"]))
        # print("    paragraphs: {}".format(len(article["paragraphs"])))
        paragraph_size += len(article["paragraphs"])
        paragraphs = []
        append_article = False
        for paragraph_index, paragraph in enumerate(article["paragraphs"]):
            # print("paragraph index:{}".format(paragraph_index))
            # print("       questions: {}".format(len(paragraph["qas"])))
            question_size += len(paragraph["qas"])
            qas = []
            append_paragraph = False
            for qa_index, qa in enumerate(paragraph["qas"]):
                if question_index in selected_indices:
                    # this is the question we want
                    append_article = True
                    append_paragraph = True
                    qas.append(qa)
                question_index += 1
            if append_paragraph:
                paragraph["qas"] = qas
                paragraphs.append(paragraph)
        if append_article:
            article["paragraphs"] = paragraphs
            articles.append(article)
    selected_data["data"] = articles
    selected_data["version"] = "1.1"
    output_dir = os.path.dirname(input_json)
    output_name = "{}-{}.json".format(os.path.splitext(os.path.basename(input_json))[0], size)
    output_path = os.path.join(output_dir, output_name)
    json.dump(selected_data, open(output_path, 'w'))
    prettify_json(output_path)


if __name__ == '__main__':
    extract_json("data/dev-v1.1.json")

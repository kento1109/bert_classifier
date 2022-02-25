import csv
import os
from typing import List

import hydra
import spacy
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

from trainer import BertExample, Trainer, build_data_loader

nlp = spacy.load("ja_ginza")


def basic_tokenize(text, use_lemma=False):
    doc = nlp(text)
    return [token.lemma_ if use_lemma else token.orth_ for token in doc]


def build_examples(file_path, tokenizer, max_seq_length):
    example_list = []
    with open(file_path) as file:
        reader = csv.reader(file, delimiter="\t")
        _ = next(reader)
        for row in reader:
            text = row[0]
            tokens = basic_tokenize(text)
            label = row[1] if row[1] else None
            input_ids = tokenizer(
                tokens, padding="max_length", max_length=max_seq_length
            )["input_ids"][:max_seq_length]
            example_list.append(BertExample(input_ids, label))
    return example_list


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    tokenizer = BertTokenizer.from_pretrained(cfg.pretrained_model_path)
    bert_config = BertConfig.from_json_file(
        os.path.join(cfg.pretrained_model_path, "config.json")
    )
    bert_config.num_labels = cfg.num_labels
    model = BertForSequenceClassification.from_pretrained(
        cfg.pretrained_model_path, config=bert_config
    )

    train_example_list = build_examples(
        cfg.train_file_path, tokenizer, cfg.max_seq_length
    )

    # test_example_list = list()
    # with open(cfg.trest_file_path) as test_file:
    #     reader = csv.reader(test_file, delimiter="\t")
    #     for row in reader:
    #         test_example_list.append(
    #             convert_to_bert_example(tokenizer, row[0], row[1], 128)
    #         )

    train_loader = build_data_loader(
        train_example_list, batch_size=cfg.batch_size, shuffle=True
    )

    # test_loader = build_data_loader(test_example_list, batch_size=16, shuffle=False)

    trainer = Trainer(model, cfg)

    trainer.train(train_loader, train_loader, lr=5e-5, num_epochs=5)

    # trainer.evaluate(test_loader)


if __name__ == "__main__":
    main()

import csv
import os

import hydra
import MeCab
import spacy
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

from trainer import Trainer, build_data_loader

# nlp = spacy.load("ja_ginza")
mc = MeCab.Tagger("-Owakati")

# def basic_tokenize(text, use_lemma=False):
#     doc = nlp(text)
#     return [token.lemma_ if use_lemma else token.orth_ for token in doc]


def basic_tokenize(text):
    return mc.parse(text).strip()


def build_examples(file_path, tokenizer, max_seq_length, label_dict):
    example_list = []
    with open(file_path) as file:
        reader = csv.reader(file, delimiter="\t")
        _ = next(reader)
        for row in reader:
            text = row[0]
            tokens = basic_tokenize(text)
            label = label_dict[row[1]] if row[1] else None
            encoded_dict = tokenizer(
                tokens,
                padding="max_length",
                max_length=max_seq_length,
                truncation=True,
                is_split_into_words=True,
            )
            example_list.append(
                {
                    "input_ids": encoded_dict["input_ids"],
                    "attention_mask": encoded_dict["attention_mask"],
                    "labels": label,
                }
            )
    return example_list


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    tokenizer = BertTokenizer.from_pretrained(cfg.pretrained_path)
    bert_config = BertConfig.from_json_file(
        os.path.join(cfg.pretrained_path, "config.json")
    )

    label_dict = {label: i for i, label in enumerate(cfg.labels)}

    bert_config.num_labels = len(label_dict)
    model = BertForSequenceClassification.from_pretrained(
        cfg.pretrained_path, config=bert_config
    )

    train_example_list = build_examples(
        cfg.train_params.train_file_path,
        tokenizer,
        cfg.train_params.max_seq_length,
        label_dict,
    )

    test_example_list = build_examples(
        cfg.train_params.test_file_path,
        tokenizer,
        cfg.train_params.max_seq_length,
        label_dict,
    )

    train_loader = build_data_loader(
        train_example_list, batch_size=cfg.train_params.batch_size, shuffle=True
    )

    test_loader = build_data_loader(test_example_list, batch_size=1, shuffle=False)

    trainer = Trainer(model, cfg.train_params)

    trainer.train(train_loader)

    if cfg.save_path is not None:
        os.makedirs(cfg.save_path, exist_ok=True)
        model.save_pretrained(cfg.save_path)
        tokenizer.save_pretrained(cfg.save_path)

    label_dict = {i: label for label, i in label_dict.items()}

    trainer.evaluate(test_loader, label_dict)


if __name__ == "__main__":
    main()

import argparse
import re
import os
import hashlib
import json

import nltk
import pandas as pd
import spacy
import fasttext

from emot.emo_unicode import UNICODE_EMOJI, EMOTICONS_EMO
from nltk.corpus import stopwords


nlp = None
model = None
model_path = None

rep_emoji = None
pat_emoji = None

rep_emoti = None
pat_emoti = None

prog = ""

def get_file_hash(path, iterations=float('inf')):
    sha1 = hashlib.sha1()
    with open(path, "rb") as f:
        chunk = 0
        i = 0
        while chunk != b'' and i < iterations:
            chunk = f.read(1024)
            sha1.update(chunk)
            i+=1
    return sha1.hexdigest()

def clean_data(df, emoji: str="replace", emoticons: str="replace"):
    df_c = df.copy()
    df_c.text = df_c.text.str.replace(r"<[^>]*>", "", regex=True)
    df_c.text = df_c.text.str.replace(r"http\S+", "", regex=True)
    df_c.text = df_c.text.str.replace(r"http", "", regex=True)
    df_c.text = df_c.text.str.replace(r"@\S+", "", regex=True)

    # change emoji and emoticons
    emoti_replacer = "" if emoticons ==  "remove" else lambda m: "("+rep_emoti[re.escape(m.group(0))].replace(" ", "_")+")"
    df_c.text = df_c.text.str.replace(pat_emoti, emoti_replacer)

    emoji_replacer = "" if emoji ==  "remove" else lambda m: "("+rep_emoji[re.escape(m.group(0))]+")"
    df_c.text = df_c.text.str.replace(pat_emoji, emoji_replacer)

    # replace dates
    df_c.text = df_c.text.str.replace(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", "datevalue", regex=True) # m*d*Y
    df_c.text = df_c.text.str.replace(r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-zA-Z.,-]*[\s-]?(\d{1,2})?[,\s-]?[\s]?\d{4}", "datevalue", regex=True) # Month*d*Y
    df_c.text = df_c.text.str.replace(r"\d{1,2}[/-]\d{4}", "datevalue", regex=True) # m*Y
    df_c.text = df_c.text.str.replace(r"\d{4}", "datevalue", regex=True) # Y

    # # replace residual non text character
    df_c.text = df_c.text.str.replace("`", "\'", regex=True)
    df_c.text = df_c.text.str.replace(r"[^A-Za-z0-9@\'\_]", " ", regex=True)
    df_c.text = df_c.text.str.replace(r"@", "at", regex=True)
    
    df_c.text = df_c.text.str.lower()

    return df_c


def tokenize_text(text):
    doc = nlp(text)
    restricted_tokens = stopwords.words("english") + ["\'", "\'s", "`", "`s"]
    lemmatized = [token.lemma_.lower() for token in doc if not token.lemma_.isspace()]
    return [token for token in lemmatized if
        token not in restricted_tokens and
        not token.isnumeric()
    ]


def handle_data(data):
    nltk.download('stopwords', quiet=True)
    global nlp

    global rep_emoji
    global pat_emoji

    global rep_emoti
    global pat_emoti

    global model

    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
    rep_emoji = {re.escape(k): v for k, v in UNICODE_EMOJI.items()}
    pat_emoji = re.compile('('+'|'.join(rep_emoji.keys()).replace('/', '\\/')+')')

    rep_emoti = {re.escape(k): v for k, v in EMOTICONS_EMO.items()}
    pat_emoti = re.compile('|'.join(rep_emoti.keys()).replace('/', '\\/'))

    model = fasttext.load_model(os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        model_path
    ))

    data_clean = clean_data(data)
    data_clean["tokens"] = data_clean.text.apply(tokenize_text)
    for tokens in data_clean["tokens"]:
        if len(tokens) == 0:
            print()
            continue
        prediction = model.predict(" ".join(tokens))
        print(prediction[0][0].replace("__label__", ""))


def handle_text(text):
    data = pd.DataFrame(data=[text], columns=["text"])
    handle_data(data)

def handle_file(path):
    try:
        f = open(path, "r")
    except IOError:
        print(prog, ": error: cant read file")
        exit(1)
    with f:
        lines = f.readlines()
        data = pd.DataFrame(data=lines, columns=["text"])
        handle_data(data)

def check_binary():
    config_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "config.json"
    )

    parser = argparse.ArgumentParser(
        prog=os.path.basename(__file__),
        description="""\
        Program to classify customer satisfaction. 
        Firstly enter binary model path.""",
        epilog="Grishin K., Samoilova A., 2022"
    )

    if  os.path.exists(config_path):
        with open(config_path) as f:
            try:
                config = json.load(f)
            except:
                print(parser.prog, ": error: config cannot be parsed! Please, re-run program and specify model path again.")
                os.remove(config_path)
                exit(1)
            if not os.path.exists(config["model_path"]):
                print(parser.prog, ": error: model file does not exist! Please, re-run program and specify model path again.")
                os.remove(config_path)
                exit(1)
            if get_file_hash(config["model_path"], iterations=5) != config["hash"]:
                print(parser.prog, ": error: model file has wrong hash! Please, re-run program and specify model path again.")
                os.remove(config_path)
                exit(1)
            global model_path
            model_path = config["model_path"]
    else:
        parser.add_argument("model", metavar="model path", type=str, nargs=1,
                            help="path to fasttext model")
        args = parser.parse_args()
        path = os.path.abspath(os.path.join(os.getcwd(), args.model[0]))
        try:
            fasttext.load_model(path)
        except:
            print(parser.prog, ": error: wrong path or model can not be parsed!")
            exit(1)
        config = {
            "model_path": path,
            "hash": get_file_hash(path, iterations=5)
        }

        with open(config_path, "w") as f:
            json.dump(config, f)
            print("Model added successfully! Print", os.path.basename(__file__), "-h for help.")
            exit(0)
        



if __name__ == "__main__":
    check_binary()

    parser = argparse.ArgumentParser(
        prog=os.path.basename(__file__),
        description="""\
        Program to classify customer satisfaction. 
        0 means negative satisfaction, 1 - otherwise.""",
        epilog="Grishin K., Samoilova A., 2022"
    )
    parser.add_argument("text", type=str, nargs="?", default=None,
                        help="Text in quotes")
    parser.add_argument("-f","--file", metavar="path", type=str, nargs="?", action="store", default=None,
                        help="Path to text delimeted with \\n")

    prog = parser.prog
    args = parser.parse_args()
    if args.text is None and args.file is None:
        print(parser.prog, ": error: neither text nor file are defined")
        exit(1)
    if args.text is not None and args.file is not None:
        print(parser.prog, ": error: both text and file are defined")
        exit(1)
    if args.text is not None and not args.text:
        print(parser.prog, ": error: text is empty string")
        exit(1)

    if args.text:
        handle_text(args.text)
    if args.file:
        handle_file(args.file)
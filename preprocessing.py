import pandas as pd
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.model_selection import train_test_split

import re
import pandas as pd
import nltk
from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WordPunctTokenizer
def extract(text):


    nltk.download("stopwords", quiet = True)

    # needed for nltk.pos_tag function nltk.download(’averaged_perceptron_tagger’)
    nltk.download("wordnet", quiet = True)
    nltk.download("averaged_perceptron_tagger", quiet = True)
    nltk.download("omw-1.4", quiet = True)


    word_punct_token = WordPunctTokenizer().tokenize(text)
    clean_token = []
    for token in word_punct_token:
        token = token.lower()
        # remove any value that are not alphabetical
        new_token = re.sub(r"[^a-zA-Z]+", "", token)
        # remove empty value and single character value
        if new_token != "" and len(new_token) >= 2:
            vowels = len([v for v in new_token if v in "aeiou"])
            if vowels != 0:  # remove line that only contains consonants
                clean_token.append(new_token)
    #print(clean_token)
    # Get the list of stop words
    stop_words = stopwords.words("english")
    # add new stopwords to the list
    stop_words.extend(["could", "though", "would", "also", "many", "much"])
    # Remove the stopwords from the list of tokens
    tokens = [x for x in clean_token if x not in stop_words]
    #print(tokens)
    data_tagset = nltk.pos_tag(tokens)
    df_tagset = pd.DataFrame(data_tagset, columns=["Word", "Tag"])

    # Create lemmatizer object
    lemmatizer = WordNetLemmatizer()
    # Lemmatize each word and display the output
    lemmatize_text = []
    for word in tokens:
        output = [
            word,
            lemmatizer.lemmatize(word, pos="n"),
            lemmatizer.lemmatize(word, pos="a"),
            lemmatizer.lemmatize(word, pos="v"),
        ]
        lemmatize_text.append(output)
    # create DataFrame using original words and their lemma words
    df = pd.DataFrame(
        lemmatize_text,
        columns=["Word", "Lemmatized Noun", "Lemmatized Adjective", "Lemmatized Verb"],
    )

    df["Tag"] = df_tagset["Tag"]

    # replace with single character for simplifying
    df = df.replace(["NN", "NNS", "NNP", "NNPS"], "n")
    df = df.replace(["JJ", "JJR", "JJS"], "a")
    df = df.replace(["VBG", "VBP", "VB", "VBD", "VBN", "VBZ"], "v")
    # '''
    # define a function where take the lemmatized word when tagset is noun, and take lemmatized adjectives when tagset is adjective
    # '''
    df_lemmatized = df.copy()
    df_lemmatized["Tempt Lemmatized Word"] = (
        df_lemmatized["Lemmatized Noun"]
        + " | "
        + df_lemmatized["Lemmatized Adjective"]
        + " | "
        + df_lemmatized["Lemmatized Verb"]
    )
    df_lemmatized.head(5)
    lemma_word = df_lemmatized["Tempt Lemmatized Word"]
    tag = df_lemmatized["Tag"]
    i = 0
    new_word = []
    while i < len(tag):
        words = lemma_word[i].split("|")
        if tag[i] == "n":
            word = words[0]
        elif tag[i] == "a":
            word = words[1]
        elif tag[i] == "v":
            word = words[2]
        new_word.append(word)
        i += 1
    df_lemmatized["Lemmatized Word"] = new_word
    #print(df_lemmatized["Lemmatized Word"])
    # calculate frequency distribution of the tokens
    lemma_word = [str(x).strip() for x in df_lemmatized["Lemmatized Word"]]
    return lemma_word


def clean(df):
    for k in range(len(df["Symptoms"])):
        if isinstance(df.loc[k, "Symptoms"], float):
            df = df.drop(k, axis=0)
        else:
            if not isinstance(df.loc[k, "Reason for Exam"], float):
                df.loc[k, "Symptoms"] = (
                    df.loc[k, "Symptoms"] + ", " + df.loc[k, "Reason for Exam"]
                )
            vals = extract(df.loc[k, "Symptoms"])
            res = []
            [res.append(x) for x in vals if x not in res]
            df.loc[k, "Symptoms"] = " ".join(res)

    df = df[["PROC_NAME", "Symptoms"]]
    return df


def embed(df):
    import openai

    def get_embedding(text, model="text-embedding-ada-002"):
        # print(text)
        if isinstance(text, float):
            return "NaN"
        else:
            text = text.replace("\n", " ")
            return openai.Embedding.create(input=[text], model=model)["data"][0][
                "embedding"
            ]

    df["embedded_symptoms"] = df.Symptoms.apply(
        lambda x: get_embedding(x, model="text-embedding-ada-002")
    )
    return df


def balance(df):
    df["embedded_symptoms"] = df.embedded_symptoms.apply(eval).apply(np.array)

    X, X_test, y, y_test = train_test_split(
        list(df.embedded_symptoms.values),
        df.label,
        test_size=1,
        random_state=42,
    )

    sm = SMOTE(sampling_strategy="minority", random_state=7)
    X_res, y_res = sm.fit_resample(X, y)

    pd.Series(y_res).value_counts().plot.bar()

    df = pd.DataFrame(
        X_res, columns=["feature_" + str(i) for i in range(np.array(X_res).shape[1])]
    )
    df["label"] = list(y_res)

    print(df["label"].value_counts())

    df["embedded_symptoms"] = df[df.columns[1:-1]].apply(
        lambda x: ",".join(x.dropna().astype(str)), axis=1
    )

    df["embedded_symptoms"] = df["embedded_symptoms"].apply(lambda x: "[" + x + "]")

    df = df[["label", "embedded_symptoms"]]

    return df
df = pd.read_excel("full_data.xlsx")
df = balance(embed(clean(df)))
df.to_csv("balanced_embed.csv")
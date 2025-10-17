"""
Helper functions for training and chat:
- tokenization
- vocabulary expansion (WordNet)
- Markov chain build/generate
- simple entity detection
"""

import re
import random
import json
from collections import defaultdict, Counter
import nltk

# Ensure required NLTK data is available
nltk_packages = ["punkt", "wordnet", "omw-1.4"]
for pkg in nltk_packages:
    try:
        nltk.data.find(pkg)
    except LookupError:
        nltk.download(pkg)

from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize

WORD_RE = re.compile(r"[A-Za-z0-9_'’@#\-\+]+")

def simple_tokenize(text):
    # Lowercase, keep mentions and hashtags, split on whitespace and punctuation
    text = text.replace("’", "'")
    tokens = [tok.lower() for tok in WORD_RE.findall(text)]
    return tokens

def build_word_bank(corpora_lines, expand_with_wordnet=True, max_synonyms_per_word=5):
    """
    Returns a set of words (expanded). corpora_lines is iterable of strings.
    Expands using WordNet synonyms, some crude morphological variants.
    """
    base = set()
    for line in corpora_lines:
        for tok in simple_tokenize(line):
            base.add(tok)

    if expand_with_wordnet:
        additions = set()
        for w in list(base):
            # only try alph tokens
            if not w.isalpha():
                continue
            try:
                syns = wn.synsets(w)
            except Exception:
                syns = []
            cnt = 0
            for s in syns:
                for lemma in s.lemmas():
                    lw = lemma.name().lower().replace("_", "")
                    if lw and lw not in base:
                        additions.add(lw)
                        cnt += 1
                        if cnt >= max_synonyms_per_word:
                            break
                if cnt >= max_synonyms_per_word:
                    break
        base.update(additions)

    # add some common slang and contractions heuristics
    heuristics = {"u", "ur", "lol", "brb", "idk", "ikr", "tho", "cuz", "omg", "wtf", "yolo", "ok", "okay", "pls"}
    base.update(heuristics)

    # morphological naive forms
    morph_adds = set()
    for w in list(base):
        if w.isalpha() and len(w) > 2:
            morph_adds.add(w + "s")
            morph_adds.add(w + "ing")
            morph_adds.add(w + "ed")
    base.update(morph_adds)

    return sorted(base)

def build_markov_chain(lines, order=2):
    """Build a simple word-level Markov chain dictionary for given lines."""
    model = defaultdict(list)
    for line in lines:
        tokens = ["<s>"] + simple_tokenize(line) + ["</s>"]
        if len(tokens) <= order:
            continue
        for i in range(len(tokens) - order):
            key = tuple(tokens[i:i+order])
            next_tok = tokens[i+order]
            model[key].append(next_tok)
    return dict(model)

def generate_markov_sentence(chain, order=2, max_len=30):
    """Generate a sentence from a Markov chain (may be noisy)."""
    if not chain:
        return ""
    start_keys = [k for k in chain.keys() if k[0] == "<s>"]
    if start_keys:
        key = random.choice(start_keys)
    else:
        key = random.choice(list(chain.keys()))
    out = list(key)
    for _ in range(max_len):
        key_t = tuple(out[-order:])
        choices = chain.get(key_t) or chain.get(random.choice(list(chain.keys())))
        if not choices:
            break
        nxt = random.choice(choices)
        if nxt == "</s>":
            break
        out.append(nxt)
    # remove start token and join
    out = [t for t in out if t != "<s>" and t != "</s>"]
    # capitalise first word
    if not out:
        return ""
    s = " ".join(out)
    s = s.replace(" i ", " I ")
    return s.capitalize()

def detect_names(text):
    """Very simple name detection: a set of tokens that look like names."""
    tokens = simple_tokenize(text)
    # common: look for capitalized forms in original text? but we lowered tokens.
    names = set()
    for tok in tokens:
        if tok.isalpha() and len(tok) <= 20:
            # crude heuristic: treat tokens not in common stopwords as names if capitalized originally
            if tok.lower() in {"kya", "colin", "val", "galaxy", "miles", "connor", "ry", "ryan", "asher"}:
                names.add(tok.lower())
    return names

def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def load_file_lines(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        # strip empty lines, keep as utterance-per-line
        return [line.strip() for line in f if line.strip()]
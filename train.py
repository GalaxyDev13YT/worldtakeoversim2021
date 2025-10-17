"""
Train script: builds models from two persona files.

Outputs to directory (default: ./models/):
- vocab.json
- tfidf_vectorizer.joblib
- nn_bot1.joblib, nn_bot2.joblib (NearestNeighbors indices)
- docs_bot1.pkl, docs_bot2.pkl (original utterances)
- markov_bot1.pkl, markov_bot2.pkl (Markov chain dicts)
"""

import os
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from joblib import dump
import pickle

from utils import load_file_lines, build_word_bank, build_markov_chain, save_json

def fit_vectorizer(all_lines):
    # Use word tokenization based on simple_tokenize implicitly via analyzer='word'
    v = TfidfVectorizer(ngram_range=(1,2), max_features=60000)
    v.fit(all_lines)
    return v

def fit_nearest_neighbors(tfidf, docs):
    X = tfidf.transform(docs)
    nn = NearestNeighbors(n_neighbors=3, metric="cosine")
    nn.fit(X)
    return nn

def main(bot1_file, bot2_file, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    bot1_lines = load_file_lines(bot1_file)
    bot2_lines = load_file_lines(bot2_file)
    all_lines = bot1_lines + bot2_lines

    print(f"[+] Bot1 utterances: {len(bot1_lines)}")
    print(f"[+] Bot2 utterances: {len(bot2_lines)}")

    print("[+] Building expanded word bank (this can take a moment)...")
    vocab = build_word_bank(all_lines, expand_with_wordnet=True)
    save_json(os.path.join(out_dir, "vocab.json"), vocab)
    print(f"[+] Saved vocabulary with {len(vocab)} tokens to {out_dir}/vocab.json")

    print("[+] Fitting TF-IDF vectorizer...")
    tfidf = fit_vectorizer(all_lines)
    dump(tfidf, os.path.join(out_dir, "tfidf_vectorizer.joblib"))
    print("[+] TF-IDF saved.")

    print("[+] Fitting NearestNeighbors for bot1...")
    nn1 = fit_nearest_neighbors(tfidf, bot1_lines)
    dump(nn1, os.path.join(out_dir, "nn_bot1.joblib"))
    with open(os.path.join(out_dir, "docs_bot1.pkl"), "wb") as f:
        pickle.dump(bot1_lines, f)
    print("[+] Bot1 index saved.")

    print("[+] Fitting NearestNeighbors for bot2...")
    nn2 = fit_nearest_neighbors(tfidf, bot2_lines)
    dump(nn2, os.path.join(out_dir, "nn_bot2.joblib"))
    with open(os.path.join(out_dir, "docs_bot2.pkl"), "wb") as f:
        pickle.dump(bot2_lines, f)
    print("[+] Bot2 index saved.")

    print("[+] Building Markov chains...")
    markov1 = build_markov_chain(bot1_lines, order=2)
    markov2 = build_markov_chain(bot2_lines, order=2)
    with open(os.path.join(out_dir, "markov_bot1.pkl"), "wb") as f:
        pickle.dump(markov1, f)
    with open(os.path.join(out_dir, "markov_bot2.pkl"), "wb") as f:
        pickle.dump(markov2, f)
    print("[+] Markov models saved.")

    print("[+] All artifacts written to:", out_dir)
    print("You can now run chatbot.py --models", out_dir)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bot1", required=True, help="Text file for bot1 (one utterance per line).")
    p.add_argument("--bot2", required=True, help="Text file for bot2 (one utterance per line).")
    p.add_argument("--out", default="models", help="Output directory for model artifacts.")
    args = p.parse_args()
    main(args.bot1, args.bot2, args.out)
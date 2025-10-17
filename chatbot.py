"""
Interactive chatbot CLI to talk to either persona.

Behavior:
- Loads models from a models dir (produced by train.py).
- For each user input:
  - Detect some names/entities (very simple).
  - If 'kya' is in the message (case-insensitive), apply persona-specific override replies.
  - Else: use TF-IDF + NearestNeighbors to find similar utterances from the persona's corpus.
    - If similarity is high (cosine distance low) return the retrieved utterance (stylized).
    - Otherwise, fall back to Markov generation using the persona's Markov chain.
"""

import argparse
import os
import pickle
import random
import numpy as np
from joblib import load
from utils import simple_tokenize, detect_names, generate_markov_sentence

# Personality override templates
PERSONALITY_OVERRIDES = {
    "bot1": {
        # Bot1 is "madly in love with Kya" and defends her
        "kya_mention": [
            "Kya? She's amazing. Don't you dare say anything bad about her.",
            "If anyone talks trash about Kya, they have to answer to me.",
            "I care about Kya so much. Please be kind to her."
        ],
        "kya_question": [
            "Honestly? Kya is perfect to me. I'd do anything to protect her."
        ]
    },
    "bot2": {
        # Bot2 dislikes Kya outwardly; secretly jealous and soft toward bot1
        "kya_mention": [
            "Ugh, Kya? She's always in the way. I don't like her.",
            "Kya again... I can't stand her, but whatever, do what you want.",
            "I don't trust Kya. Keep your distance from her."
        ],
        "kya_secret_towards_bot1": [
            "I hate that Kya and you get along. Not that it matters.",
            "Fine, Kya can have you. I'm not jealous at all... 🙄"
        ]
    }
}

SIMILARITY_DISTANCE_THRESHOLD = 0.45  # lower cosine distance = more similar (cosine metric used)

def load_models(models_dir):
    artifacts = {}
    artifacts['tfidf'] = load(os.path.join(models_dir, "tfidf_vectorizer.joblib"))
    artifacts['nn1'] = load(os.path.join(models_dir, "nn_bot1.joblib"))
    artifacts['nn2'] = load(os.path.join(models_dir, "nn_bot2.joblib"))
    with open(os.path.join(models_dir, "docs_bot1.pkl"), "rb") as f:
        artifacts['docs1'] = pickle.load(f)
    with open(os.path.join(models_dir, "docs_bot2.pkl"), "rb") as f:
        artifacts['docs2'] = pickle.load(f)
    with open(os.path.join(models_dir, "markov_bot1.pkl"), "rb") as f:
        artifacts['markov1'] = pickle.load(f)
    with open(os.path.join(models_dir, "markov_bot2.pkl"), "rb") as f:
        artifacts['markov2'] = pickle.load(f)
    return artifacts

def reply_from_bot(user_input, which_bot, artifacts):
    names = detect_names(user_input)
    lower = user_input.lower()

    # Kya override handling (case-insensitive token match)
    if "kya" in lower or "k.y.a" in lower:
        if which_bot == "bot1":
            # always defend and be affectionate
            return random.choice(PERSONALITY_OVERRIDES['bot1']['kya_mention'])
        else:
            # bot2 outwardly dislikes Kya; but if bot2 sees "galaxydev" or "bot1" referenced it might show jealousy
            if "galaxydev" in lower or "colin" in lower or "bot1" in lower:
                # reveal a softer, jealous line sometimes
                if random.random() < 0.5:
                    return random.choice(PERSONALITY_OVERRIDES['bot2']['kya_secret_towards_bot1'])
            return random.choice(PERSONALITY_OVERRIDES['bot2']['kya_mention'])

    # otherwise retrieval
    tfidf = artifacts['tfidf']
    if which_bot == "bot1":
        nn = artifacts['nn1']
        docs = artifacts['docs1']
    else:
        nn = artifacts['nn2']
        docs = artifacts['docs2']

    vec = tfidf.transform([user_input])
    dists, idxs = nn.kneighbors(vec, n_neighbors=1)
    dist = float(dists[0][0])
    idx = int(idxs[0][0])

    # similarity check (NearestNeighbors is fit with metric='cosine', so dist is cosine distance)
    if dist <= SIMILARITY_DISTANCE_THRESHOLD:
        candidate = docs[idx]
        # Slight polishing: if candidate is short, maybe append small Markov fragment
        if len(candidate.split()) < 4:
            # combine with markov fragment
            markov = artifacts['markov1'] if which_bot == "bot1" else artifacts['markov2']
            gen = generate_markov_sentence(markov, order=2, max_len=12)
            if gen:
                return candidate + " " + gen
        return candidate
    else:
        # fall back to markov generation
        markov = artifacts['markov1'] if which_bot == "bot1" else artifacts['markov2']
        gen = generate_markov_sentence(markov, order=2, max_len=20)
        # if markov failed, just do a canned reply
        if not gen:
            return random.choice([
                "Hmm, tell me more.",
                "Oh? Interesting. Go on.",
                "I don't have much to say about that, but I'm listening."
            ])
        return gen

def interactive_loop(artifacts):
    print("Dual-Persona Chatbot\nCommands: /bot1, /bot2, /quit")
    which_bot = "bot1"
    print(f"Default persona: {which_bot} (bot1 = galaxydev13, bot2 = daydreaming_val_76222)")
    while True:
        try:
            inp = input(f"[You] (active={which_bot})> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break
        if not inp:
            continue
        if inp.startswith("/"):
            # commands
            if inp.lower().startswith("/bot1"):
                which_bot = "bot1"
                print("Switched to bot1.")
                continue
            if inp.lower().startswith("/bot2"):
                which_bot = "bot2"
                print("Switched to bot2.")
                continue
            if inp.lower().startswith("/quit"):
                print("Goodbye.")
                break
            print("Unknown command. Use /bot1, /bot2, /quit")
            continue

        response = reply_from_bot(inp, which_bot, artifacts)
        # simple postprocessing: ensure punctuation at end
        if response and response[-1] not in ".!?":
            response = response + "."
        print(f"[{which_bot}] {response}")

def main(models_dir):
    if not os.path.isdir(models_dir):
        raise SystemExit("models dir not found. Run train.py first.")
    artifacts = load_models(models_dir)
    interactive_loop(artifacts)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="models", help="Directory with trained artifacts.")
    args = p.parse_args()
    main(args.models)
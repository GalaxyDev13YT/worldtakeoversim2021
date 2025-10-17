```markdown
# Bare-bones Custom Dual-Persona Chatbot — Complete Setup

This repository contains:
- train.py        (TF-IDF + NearestNeighbors training + Markov fallback)
- chatbot.py      (interactive CLI)
- utils.py        (tokenization, markov, simple utilities)
- split_logs.py   (parser to extract per-user lines from a single chat log)
- run_all.sh      (convenience script to set up venv, parse, train, run)
- requirements.txt

Goal: take your single chat log (thecapicornist.txt), split into two persona files, train persona models and run an interactive chatbot that can behave like:
- Bot1 — GalaxyDev13 style (trained on messages from galaxydev / galaxydev13)
- Bot2 — Val style (trained on messages from daydreaming_val_76222 / Val)

Important: This system runs fully locally on your machine (no external API required). It uses classical NLP (TF-IDF + retrieval + Markov fallback) to produce persona-consistent replies.

Quick start (Unix / macOS)
1. Put your chat log file in the project root and name it:
   thecapicornist.txt

2. Make the run script executable:
   chmod +x run_all.sh

3. Run the script:
   ./run_all.sh

What run_all.sh does
- Creates a Python venv (.venv_chatbot)
- Installs packages from requirements.txt
- Runs split_logs.py to create:
  - data/galaxydev13.txt
  - data/daydreaming_val_76222.txt
- Runs train.py to build models into `models/`
- Launches chatbot.py with the models directory

Manual steps (if you prefer not to use run_all.sh)
1. Create & activate a virtualenv:
   python3 -m venv .venv_chatbot
   source .venv_chatbot/bin/activate

2. Install requirements:
   pip install -r requirements.txt

3. Parse the log:
   python3 split_logs.py --infile thecapicornist.txt --outdir data

4. Train:
   python3 train.py --bot1 data/galaxydev13.txt --bot2 data/daydreaming_val_76222.txt --out models

5. Chat:
   python3 chatbot.py --models models

How to chat
- /bot1 — switch to GalaxyDev13 persona
- /bot2 — switch to Val persona
- /quit — exit

Notes & troubleshooting
- NLTK downloads: the first run may download NLTK data (punkt, wordnet). If you see prompts, allow the downloads or run:
  python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')"

- If split_logs.py doesn't capture everything correctly, open data/galaxydev13.txt and data/daydreaming_val_76222.txt and manually edit or add lines (one message per line) — the system performs best when training files contain many short utterances in the persona style.

- The Kya personality override is built into chatbot.py. You can edit PERSONLITY_OVERRIDES in chatbot.py to change how each bot reacts to mentions of Kya (or add more named-entity rules).

Privacy & safety
- This system reproduces patterns from your training logs. If the logs contain sensitive content, profanity, or mentions of self-harm, sanitize before training.
- If you want a content filter, add a blacklist filter in chatbot.py before printing responses.

Want me to:
- Tailor the parser to any quirks in your thecapicornist.txt format? (e.g., different separators, emojis, or embedded images)
- Add a small web UI or Discord adapter (I can add a simple Flask app or a Discord bot wrapper — note Discord integration requires an API token and following their guidelines)
```
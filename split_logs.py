#!/usr/bin/env python3
"""
split_logs.py

Parse a single chat log (like thecapicornist.txt) and extract per-user
utterances into two files suitable for training:

- data/galaxydev13.txt          (bot1)
- data/daydreaming_val_76222.txt (bot2)

Heuristics / behavior:
- Handles lines formatted like:
    "galaxydev — 8/5/2025 5:46 PM"
    "Val — 8/6/2025 7:41 AM"
    "galaxydev" [on its own line] followed by a timestamp line and content lines
- Collects multi-line messages until the next username/timestamp line.
- Normalizes short messages, strips images markers, and deduplicates outputs.
- You can extend NAME_MAP to include more name variants if needed.
"""
import re
import os
import argparse
from collections import defaultdict

NAME_MAP = {
    # map variants to bot keys
    # Bot1: galaxydev13 (Colin)
    "bot1": [
        "galaxydev13", "galaxydev", "colin", "galaxydev13yt", "galvin"
    ],
    # Bot2: daydreaming_val_76222 (Val)
    "bot2": [
        "daydreaming_val_76222", "daydreaming_val", "val", "daydreaming_val_76222"
    ]
}

# Reverse lookup for quick mapping
def username_to_bot(name):
    if not name:
        return None
    n = name.lower().strip()
    n = re.sub(r'^@', '', n)  # strip leading @
    n = re.split(r'\s+', n)[0]  # only consider first token
    for bot, variants in NAME_MAP.items():
        for v in variants:
            if n == v.lower():
                return bot
    return None

# Helper to decide if a line is a username or username + timestamp
USERNAME_WITH_DASH_RE = re.compile(r'^([^—\n\r]+?)\s*—\s*(.+)$')
USERNAME_ONLY_RE = re.compile(r'^[A-Za-z0-9_@]{1,60}$')
POSSIBLE_DATE_RE = re.compile(r'\d{1,2}[/\-]\d{1,2}[/\-]?\d{2,4}|\d{4}')  # crude date detector

def parse_log(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        raw_lines = f.readlines()

    # normalize lines (strip trailing newlines but keep empties as '')
    lines = [ln.rstrip('\n').rstrip('\r') for ln in raw_lines]

    outputs = defaultdict(list)  # bot -> list of messages
    current_bot = None
    i = 0
    N = len(lines)

    def is_username_line(s):
        if not s:
            return False
        # username — timestamp or username only (no spaces except underscores)
        if USERNAME_WITH_DASH_RE.match(s):
            return True
        if USERNAME_ONLY_RE.match(s.strip()):
            return True
        return False

    while i < N:
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # Case 1: "username — rest"
        m = USERNAME_WITH_DASH_RE.match(line)
        if m:
            username = m.group(1).strip()
            after_dash = m.group(2).strip()
            mapped = username_to_bot(username)
            current_bot = mapped
            # If after_dash appears to be a timestamp (starts with digits / contains year),
            # then message likely begins on next non-empty line(s)
            if POSSIBLE_DATE_RE.search(after_dash):
                # gather message lines after this
                j = i + 1
                msg_lines = []
                while j < N:
                    nl = lines[j].strip()
                    if not nl:
                        j += 1
                        continue
                    # if next line looks like a username (start of next message), break
                    if USERNAME_WITH_DASH_RE.match(nl) or USERNAME_ONLY_RE.match(nl):
                        break
                    msg_lines.append(nl)
                    j += 1
                i = j
                if current_bot and msg_lines:
                    msg = " ".join(msg_lines).strip()
                    if msg:
                        outputs[current_bot].append(msg)
                continue
            else:
                # after_dash likely contains the start of the message (rare)
                msg = after_dash
                j = i + 1
                while j < N:
                    nl = lines[j].strip()
                    if not nl:
                        j += 1
                        continue
                    if USERNAME_WITH_DASH_RE.match(nl) or USERNAME_ONLY_RE.match(nl):
                        break
                    msg += " " + nl
                    j += 1
                i = j
                if current_bot and msg.strip():
                    outputs[current_bot].append(msg.strip())
                continue

        # Case 2: username only on a line (e.g., "galaxydev")
        if USERNAME_ONLY_RE.match(line):
            username = line
            mapped = username_to_bot(username)
            current_bot = mapped
            # Look ahead: often next non-empty line is "— date" and next line(s) are the message
            j = i + 1
            if j < N:
                nxt = lines[j].strip()
                # if next line starts with a dash (timestamp style) skip it
                if nxt.startswith('—') or USERNAME_WITH_DASH_RE.match(nxt):
                    # skip that timestamp-like line
                    k = j + 1
                    msg_lines = []
                    while k < N:
                        nl = lines[k].strip()
                        if not nl:
                            k += 1
                            continue
                        if USERNAME_WITH_DASH_RE.match(nl) or USERNAME_ONLY_RE.match(nl):
                            break
                        msg_lines.append(nl)
                        k += 1
                    i = k
                    if current_bot and msg_lines:
                        outputs[current_bot].append(" ".join(msg_lines).strip())
                    continue
            # Otherwise, gather following non-username lines as message
            k = i + 1
            msg_lines = []
            while k < N:
                nl = lines[k].strip()
                if not nl:
                    k += 1
                    continue
                if USERNAME_WITH_DASH_RE.match(nl) or USERNAME_ONLY_RE.match(nl):
                    break
                msg_lines.append(nl)
                k += 1
            i = k
            if current_bot and msg_lines:
                outputs[current_bot].append(" ".join(msg_lines).strip())
            continue

        # Case 3: general message line — if we have a current_bot, append this as message content
        if current_bot:
            # gather contiguous non-username lines into a message
            j = i
            msg_lines = []
            while j < N:
                nl = lines[j].strip()
                if not nl:
                    j += 1
                    continue
                if USERNAME_WITH_DASH_RE.match(nl) or USERNAME_ONLY_RE.match(nl):
                    break
                msg_lines.append(nl)
                j += 1
            i = j
            if msg_lines:
                outputs[current_bot].append(" ".join(msg_lines).strip())
            continue

        # fallback
        i += 1

    # Postprocess: sanitize messages (remove inline 'Image' markers, truncate extremes, de-duplicate)
    def sanitize(msg):
        msg = re.sub(r'\[image\]|\bimage\b|^image$', '', msg, flags=re.IGNORECASE)
        msg = re.sub(r'https?://\S+', '', msg)  # remove urls
        msg = re.sub(r'\s+', ' ', msg).strip()
        return msg

    for bot in list(outputs.keys()):
        cleaned = []
        seen = set()
        for m in outputs[bot]:
            s = sanitize(m)
            if not s:
                continue
            if s in seen:
                continue
            seen.add(s)
            cleaned.append(s)
        outputs[bot] = cleaned

    return outputs

def write_output(outputs, out_dir="data"):
    os.makedirs(out_dir, exist_ok=True)
    bot1_path = os.path.join(out_dir, "galaxydev13.txt")
    bot2_path = os.path.join(out_dir, "daydreaming_val_76222.txt")
    with open(bot1_path, "w", encoding="utf-8") as f:
        for line in outputs.get("bot1", []):
            f.write(line + "\n")
    with open(bot2_path, "w", encoding="utf-8") as f:
        for line in outputs.get("bot2", []):
            f.write(line + "\n")
    return bot1_path, bot2_path

def main():
    p = argparse.ArgumentParser(description="Split a combined chat log into per-persona files.")
    p.add_argument("--infile", default="thecapicornist.txt", help="Input chat log file (raw).")
    p.add_argument("--outdir", default="data", help="Output directory for persona files.")
    args = p.parse_args()
    if not os.path.isfile(args.infile):
        print(f"Input file {args.infile} not found. Put the chat log (thecapicornist.txt) in the project root or pass --infile.")
        return
    outputs = parse_log(args.infile)
    bot1_path, bot2_path = write_output(outputs, args.outdir)
    print(f"Wrote {len(outputs.get('bot1', []))} messages to {bot1_path}")
    print(f"Wrote {len(outputs.get('bot2', []))} messages to {bot2_path}")
    print("Done. Now run train.py with these files to build the models.")

if __name__ == "__main__":
    main()
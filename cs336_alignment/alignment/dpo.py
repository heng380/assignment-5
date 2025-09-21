"""
    DPO implementation on anthropic harmless/helpful data.
"""
import os
import gzip
import json

HH_PATH = "/home/ubuntu/hengcao/assignment-5/data/hh-rlhf"
def load_hh_dataset():
    filenames = [
        "harmless-base.jsonl.gz",
        "helpful-base.jsonl.gz",
        "helpful-online.jsonl.gz",
        "helpful-rejection-sampled.jsonl.gz"
    ]

    all_examples = []

    for filename in filenames:
        file_path = os.path.join(HH_PATH, filename)
        with gzip.open(file_path, "rt") as f:
            for line in f:
                data = json.loads(line)

                chosen_conversation = data.get("chosen", "")
                rejected_conversation = data.get("rejected", "")

                chosen_messages = [msg for msg in chosen_conversation.split("\n\n") if msg.strip()]
                rejected_messages = [msg for msg in rejected_conversation.split("\n\n") if msg.strip()]
                if len(chosen_messages) < 2 or len(rejected_messages) < 2:
                    continue
                if len(chosen_messages) > 2 or len(rejected_messages) > 2:
                    continue

                
                    
import json
import random
import argparse

EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]

parser = argparse.ArgumentParser()
parser.add_argument('--input_fn', type=str, default="mwz2.1/train_dials.json")
parser.add_argument('--target_fn', type=str)
parser.add_argument('--ratio', type=float)
parser.add_argument('--seed', type=int)
args = parser.parse_args()


def sample_process_dataset(data, ratio=1.0, seed=88):

    data = [dial for dial in data if set(
        dial['domains']).issubset(set(EXPERIMENT_DOMAINS))]
    processed_turns = []

    # random sampling a ratio of data
    random.seed(seed)
    n_sampled = int(len(data)*ratio)
    dial_ids = [dial['dialogue_idx'] for dial in data]

    selected_ids = random.sample(dial_ids, n_sampled)
    if ratio == 1.0:
        selected_ids = dial_ids
    print(f"sampled {n_sampled} dialogues")

    for dial in data:
        sys = []
        usr = []
        dial_id = dial['dialogue_idx']

        if dial_id not in selected_ids:
            continue

        domains = dial['domains']
        last_slot_values = {}
        for turn_id, turn in enumerate(dial["dialogue"]):
            sys.append(turn["system_transcript"])
            usr.append(turn["transcript"])

            slot_values = {}
            for sv in turn["belief_state"]:
                slot = sv["slots"][0][0]
                value = sv["slots"][0][1]
                if value != "none":
                    slot_values[slot] = value

            turn_slot_values = {}
            for s,v in slot_values.items():
                # newly added slot
                if s not in last_slot_values:
                    turn_slot_values[s] = v
                
                # changed slot
                elif slot_values[s] != last_slot_values[s]:
                    turn_slot_values[s] = v
                
            # deleted slot
            for s,v in last_slot_values.items():
                if s not in slot_values:
                    turn_slot_values[s] = "[DELETE]"    

            processed_turn = {"ID": dial_id, "turn_id": turn_id,
                              "domains": domains}
            processed_turn["dialog"] = {"sys": sys.copy(), "usr": usr.copy()}
            processed_turn["slot_values"] = slot_values.copy()
            processed_turn["turn_slot_values"] = turn_slot_values.copy()
            processed_turn["last_slot_values"] = last_slot_values.copy()
            processed_turns.append(processed_turn)

            # update context
            last_slot_values = slot_values
    return processed_turns


def main():
    with open(args.input_fn) as f:
        raw_data = json.load(f)
    processed_data = sample_process_dataset(raw_data,ratio=args.ratio, seed=args.seed)
    with open(args.target_fn, 'w') as f:
        json.dump(processed_data, f, indent=4)

if __name__ == "__main__":
    main()

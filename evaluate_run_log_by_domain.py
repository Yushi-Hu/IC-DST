import json
import argparse
from collections import defaultdict
from tqdm import tqdm
from utils.helper import PreviousStateRecorder
from utils.typo_fix import typo_fix
from config import CONFIG

from utils.sql import sql_pred_parse, sv_dict_to_string
from evaluate_metrics import evaluate

# input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--running_log', type=str, required=True,
                    help="running log filename")
parser.add_argument('--test_fn', type=str, default="./data/mw24_100p_test.json",
                    help="running log filename")
parser.add_argument('--mwz_ver', type=str, default="2.4",
                    choices=['2.1', '2.4'], help="version of MultiWOZ")
args = parser.parse_args()


# read the ontology and the test set
if args.mwz_ver == '2.1':
    ontology_path = CONFIG["ontology_21"]
else:
    ontology_path = CONFIG["ontology_24"]

with open(ontology_path) as f:
    ontology = json.load(f)

DOMAINS = ['hotel', 'train', 'restaurant','taxi','attraction']


def eval(running_log, test_set, turn=-1, use_gold=False, domain=""):
    # turn and use_gold are for analysis purpose
    # turn = -1 means evalute all dialogues
    # turn = 0 means evaluate single-turn dialogues
    # turn = 1 means evalute two-turn dialogues... etc.
    # when use_gold = True, the context are gold context (for analysis purpose)

    # keep the slot values in domain
    def domain_filter(slot_values):
        in_domain_svs = {}
        for k,v in slot_values.items():
            if k.split('-')[0] == domain:
                in_domain_svs[k] = v
        return in_domain_svs

    result_dict = defaultdict(list)  # use to record the accuracy

    prediction_recorder = PreviousStateRecorder()  # state recorder

    # start experiment
    all_result = []
    n_total = 0
    n_correct = 0
    total_acc = 0
    total_f1 = 0

    for data_item, label_item in tqdm(zip(running_log, test_set)):
        
        if turn >= 0:
            if data_item['turn_id'] != turn:
                continue

        if domain:
            if domain not in data_item["domains"]:
                continue
        
        n_total += 1
        
        completion = data_item['completion']

        # aggregate the prediction and the history states
        predicted_slot_values = {}
        try:
            predicted_slot_values = sql_pred_parse(completion)  # a dictionary
        except:
            print("the output is not a valid SQL query")
            data_item['not_valid'] = 1
        predicted_slot_values = typo_fix(
            predicted_slot_values, ontology=ontology, version=args.mwz_ver)

        context_slot_values = data_item['last_slot_values']  # a dictionary

        # merge context and prediction
        if use_gold:
            all_slot_values = context_slot_values.copy()
        else:
            all_slot_values = prediction_recorder.state_retrieval(
                data_item).copy()

        for s, v in predicted_slot_values.items():

            if s in all_slot_values and v == "[DELETE]":
                del all_slot_values[s]
            elif v != "[DELETE]":
                all_slot_values[s] = v

        # some slots may contain multiple values
        all_slot_values = {k: v.split('|')[0]
                           for k, v in all_slot_values.items()}

        # record current turn prediction
        prediction_recorder.add_state(data_item, all_slot_values)

        # print the result
        print(completion)
        print(
            f"this is the {n_total - 1}th example. {data_item['ID']}_turn_{data_item['turn_id']}")
        print(
            f"pred turn change: {sv_dict_to_string(predicted_slot_values, sep='-')}")
        print(
            f"gold turn change: {sv_dict_to_string(label_item['turn_slot_values'], sep='-')}")
        print(f"pred states: {sv_dict_to_string(all_slot_values, sep='-')}")
        print(
            f"gold states: {sv_dict_to_string(label_item['slot_values'], sep='-')}")

        this_jga, this_acc, this_f1 = evaluate(
            all_slot_values, label_item['slot_values'])

        if domain:
            this_jga, this_acc, this_f1 = evaluate(
                domain_filter(all_slot_values), domain_filter(label_item['slot_values']))

        total_acc += this_acc
        total_f1 += this_f1

        if this_jga:
            n_correct += 1
            result_dict[data_item['turn_id']].append(1)
            print("\n=====================correct!=======================")
        else:
            result_dict[data_item['turn_id']].append(0)
            print("\n=====================wrong!=======================")

        print("\n")

    print(f"correct {n_correct}/{n_total}  =  {n_correct / n_total}")
    print(f"Slot Acc {total_acc/n_total}")
    print(f"Joint F1 {total_f1/n_total}")
    print()

    # calculate the accuracy of each turn
    for k, v in result_dict.items():
        print(f"accuracy of turn {k} is {sum(v)}/{len(v)} = {sum(v) / len(v)}")

    return


if __name__ == "__main__":

    # read the running log
    with open(args.running_log) as f:   
        running_log = json.load(f)

    # read the testing file
    with open(args.test_fn) as f:
        test_set = json.load(f)

    for domain in DOMAINS:
        print(f"DOMAIN {domain} result:")
        eval(running_log,test_set, domain=domain)

import json
from pickle import FALSE
from statistics import mean
from tqdm import tqdm

def compute_prf(gold, pred):
    TP, FP, FN = 0, 0, 0
    if len(gold) != 0:
        count = 1
        for g in gold:
            if g in pred:
                TP += 1
            else:
                FN += 1
        for p in pred:
            if p not in gold:
                FP += 1
        precision = TP / float(TP+FP) if (TP+FP) != 0 else 0
        recall = TP / float(TP+FN) if (TP+FN) != 0 else 0
        F1 = 2 * precision * recall / \
            float(precision + recall) if (precision+recall) != 0 else 0
    else:
        if len(pred) == 0:
            precision, recall, F1, count = 1, 1, 1, 1
        else:
            precision, recall, F1, count = 0, 0, 0, 1
    return float(F1)


def multival_to_single(belief):
    return [f"{'-'.join(sv.split('-')[:2])}-{(sv.split('-')[-1]).split('|')[0]}" for sv in belief]


# mean of slot similarity and value similarity
def compute_sv_sim(gold, pred, onescore=True):

    if type(gold) == dict:
        gold = [f"{k}-{v}" for k, v in gold.items()]
    if type(pred) == dict:
        pred = [f"{k}-{v}" for k, v in pred.items()]

    gold = multival_to_single(gold)
    pred = multival_to_single(pred)

    value_sim = compute_prf(gold, pred)

    gold = ['-'.join(g.split('-')[:2]) for g in gold]
    pred = ['-'.join(g.split('-')[:2]) for g in pred]
    slot_sim = compute_prf(gold, pred)

    if onescore:
        return value_sim + slot_sim - 1
    else:
        return value_sim, slot_sim


def evaluate_single_query_ex(turn, retriever):
    examples = retriever.item_to_nearest_examples(turn)

    query_turn_sv = turn['turn_slot_values']
    query_sv = turn['slot_values']

    turn_value_sims = []
    turn_slot_sims = []
    all_value_sims = []
    all_slot_sims = []

    for ex in examples:
        this_turn_sv = ex['turn_slot_values']
        this_sv = ex['slot_values']

        turn_value_sim, turn_slot_sim = compute_sv_sim(
            query_turn_sv, this_turn_sv, onescore=False)
        all_value_sim, all_slot_sim = compute_sv_sim(query_sv, this_sv, onescore=False)

        turn_value_sims.append(turn_value_sim)
        turn_slot_sims.append(turn_slot_sim)
        all_value_sims.append(all_value_sim)
        all_slot_sims.append(all_slot_sim)

    return mean(turn_value_sims), mean(turn_slot_sims), mean(all_value_sims), mean(all_slot_sims)


def evaluate_retriever_on_dataset(dataset, retriever):
    turn_value_sims = []
    turn_slot_sims = []
    all_value_sims = []
    all_slot_sims = []

    for ds in tqdm(dataset):
        turn_value_sim, turn_slot_sim, all_value_sim, all_slot_sim = evaluate_single_query_ex(
            ds, retriever)
        turn_value_sims.append(turn_value_sim)
        turn_slot_sims.append(turn_slot_sim)
        all_value_sims.append(all_value_sim)
        all_slot_sims.append(all_slot_sim)

    return mean(turn_value_sims), mean(turn_slot_sims), mean(all_value_sims), mean(all_slot_sims)

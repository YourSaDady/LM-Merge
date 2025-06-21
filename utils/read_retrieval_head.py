import json
import numpy as np


def read_retrieval_head(retrieval_head_file="./Mistral-7B-v0.1.json", cutoff=0.1, *args):
    with open(retrieval_head_file, "r") as f:
        head_list = json.load(f)
    head_score_list = [([int(number) for number in item[0].split('-')], np.mean(item[1])) for item in head_list.items()]
    head_score_list = sorted(head_score_list, key=lambda x: x[1], reverse=True)

    i = 0
    for i, (head, score) in enumerate(head_score_list):
        if score < cutoff:
            break

    return head_score_list[:i]


def compare_heads(head1, head2):
    head1_set = set([tuple(head) for head, score in head1])
    head2_set = set([tuple(head) for head, score in head2])

    head1_dict = {tuple(head): round(score, 3) for head, score in head1}
    head2_dict = {tuple(head): round(score, 3) for head, score in head2}

    all_set = head1_set.union(head2_set)
    all_sorted = sorted(all_set)

    head_distribution = []
    for head in all_sorted:
        if head in head1_set and head in head2_set:
            head_distribution.append(3)
        elif head in head1_set:
            head_distribution.append(1)
        else:
            head_distribution.append(2)

    # debug: print info

    print("Head distribution:")
    print("{:<10}\t{:<10}\t{:<10}".format("Head", "Model 1", "Model 2"))
    print(f"Head      \tModel 1   \tModel 2")
    for i, head in enumerate(all_sorted):
        head_str = str(head)
        if head_distribution[i] == 3:
            print("{0:<10}\t{1:<10}\t{2:<10}".format(head_str, head1_dict[head], head2_dict[head]))
        elif head_distribution[i] == 1:
            print("{0:<10}\t{1:<10}\t{2:<10}".format(head_str, head1_dict[head], ""))
        else:
            print("{0:<10}\t{1:<10}\t{2:<10}".format(head_str, "", head2_dict[head]))


if __name__ == "__main__":
    head1 = read_retrieval_head("../../Retrieval_Head/head_score/Mistral-7B-v0.1.json", .1)
    head2 = read_retrieval_head("../../Retrieval_Head/head_score/MegaBeam-Mistral-7B-300k.json", .1)
    compare_heads(head1, head2)
import lightgbm as lgb
import numpy as np
import json
import argparse
import xml.etree.ElementTree as ET
from run_lgb_cv_bayesopt import read_labels


def predict(testfeatures, stackmodel_path):
    model = lgb.Booster(model_file=stackmodel_path)
    test_pred = model.predict(
        testfeatures, num_iteration=model.best_iteration)
    test_pred = np.argmax(test_pred, axis=1)
    return list(test_pred)


def get_doc_sent_id(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()
    lst = []
    for doc in root:
        docID = doc.get('ID')
        for sent in doc:
            sentID = sent.get('ID')
            lst.append(docID + '-' + sentID)
    return lst

def save_file(filepath, lst):
    with open(filepath, 'w', encoding='utf-8') as f:
        for i in range(len(lst)):
            f.write(lst[i])
            if i != len(lst) - 1:
                f.write('\n')


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    # parser.add_argument("--train_path", type=str, required=True,
    #                     help="Path of the trainset.")
    parser.add_argument("--test_path", type=str, required=True,
                        help="Path of the testset.")
    # parser.add_argument("--train_features_path", type=str, required=True,
    #                     help="Path of the train features for stacking.")
    parser.add_argument("--test_features_path", type=str, required=True,
                        help="Path of the test features for stacking.")

    # Model options.
    parser.add_argument("--models_num", type=int, default=64,
                        help="Number of models for ensemble.")
    parser.add_argument("--labels_num", type=int, default=6,
                        help="Number of label.")

    args = parser.parse_args()

    # train_features = []
    # for i in range(args.models_num):
    #     train_features.append(
    #         np.load(args.train_features_path + "train_features_" + str(i) + ".npy"))
    # train_features = np.concatenate(train_features, axis=-1)
    # train_labels = read_labels(args.train_path)

    test_features = []
    for i in range(args.models_num):
        test_features.append(
            np.load(args.test_features_path + "test_features_" + str(i) + ".npy"))
    test_features = np.concatenate(test_features, axis=-1)
    # test_labels = read_labels(args.test_path)

    params = {
        "task": "train",
        "objective": "multiclass",
        "num_class": args.labels_num,
        "metric": "multi_error",
        "feature_fraction": 0.1735,
        "lambda_l1": 6.401,
        "lambda_l2": 8.211,

        "learning_rate": 0.03178,
        "max_depth": 125,
        "min_data_in_leaf": 31,
        "num_leaves": 53
    }

    pred = predict(test_features, 'stack_model.txt')
    doc_sent_id = get_doc_sent_id('/home/tyx/wch/UER-py/datasets/smp2019-ecisa/ECISA2021-Test.xml')
    mergelst = [(i[0] +'\t' + str(i[1])) for i in zip(doc_sent_id, pred)]
    save_file('result.txt', mergelst)


if __name__ == "__main__":
    main()

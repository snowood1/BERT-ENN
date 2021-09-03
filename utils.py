import numpy as np
import pandas as pd
import random

import torch
import torch.utils.data
from torch import nn

import torchtext.data
import torchtext.datasets

from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import precision_recall_curve, f1_score, auc, roc_curve, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split


def load_dataset(dataset):

    train_sentences = None
    val_sentences = None
    test_sentences = None
    train_labels = None
    val_labels = None
    test_labels = None

    ##  Training Datasets: 20news, trec, sst  ##

    if dataset == '20news':
        VALIDATION_SPLIT = 0.8
        newsgroups_train = fetch_20newsgroups('dataset/20news', subset='train', shuffle=True, random_state=0)
        print(newsgroups_train.target_names)
        print(len(newsgroups_train.data))

        newsgroups_test = fetch_20newsgroups('dataset/20news', subset='test', shuffle=False)

        print(len(newsgroups_test.data))

        train_len = int(VALIDATION_SPLIT * len(newsgroups_train.data))

        train_sentences = newsgroups_train.data[:train_len]
        val_sentences = newsgroups_train.data[train_len:]
        test_sentences = newsgroups_test.data
        train_labels = newsgroups_train.target[:train_len]
        val_labels = newsgroups_train.target[train_len:]
        test_labels = newsgroups_test.target

    if dataset == 'trec':
        # set up fields
        TEXT = torchtext.data.Field(pad_first=True, lower=True)
        LABEL = torchtext.data.Field(sequential=False)

        # make splits for data
        train, test = torchtext.datasets.TREC.splits(TEXT, LABEL, fine_grained=True)

        train_text = []
        train_label = []
        for example in train.examples:
            train_text.append(' '.join(example.text))
            train_label.append(example.label)
        df_train = pd.DataFrame(columns=['text', 'label'])
        df_train.text = train_text
        df_train.label = train_label

        X_train, X_val, y_train, y_val = train_test_split(df_train.text, df_train.label, test_size=0.2,
                                                          random_state=42, stratify=df_train.label)
        test_text = []
        test_label = []
        for example in test.examples:
            test_text.append(' '.join(example.text))
            test_label.append(example.label)
        df_test = pd.DataFrame(columns=['text', 'label'])
        df_test.text = test_text
        df_test.label = test_label

        train_sentences = X_train.values
        val_sentences = X_val.values
        test_sentences = df_test.text.values
        train_labels = y_train.values
        val_labels = y_val.values
        test_labels = df_test.label.values

        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.fit(train_labels)
        train_labels = le.transform(train_labels)
        val_labels = le.transform(val_labels)
        test_labels =le.transform(test_labels)

    if dataset == 'sst':
        df_train = pd.read_csv("./dataset/sst/SST-2/train.tsv", delimiter='\t', header=0)

        df_train = df_train.groupby('label').sample(10000)

        df_val = pd.read_csv("./dataset/sst/SST-2/dev.tsv", delimiter='\t', header=0)

        df_test = pd.read_csv("./dataset/sst/SST-2/sst-test.tsv", delimiter='\t', header=None,
                              names=['sentence', 'label'])

        train_sentences = df_train.sentence.values
        val_sentences = df_val.sentence.values
        test_sentences = df_test.sentence.values
        train_labels = df_train.label.values
        val_labels = df_val.label.values
        test_labels = df_test.label.values

    ##  Training OOD dataset ##

    if dataset == 'wikitext2':
        file_path = './dataset/wikitext_reformatted/wikitext2_sentences'
        with open(file_path, 'r') as read_file:
            x_temp = read_file.readlines()
            x_all = []
            for x in x_temp:
                x_all.append(str(x))

        print(len(x_all))

        train_sentences = None
        val_sentences = None
        test_sentences = x_all
        train_labels = None
        val_labels = None
        test_labels = None

    ##  Testing OOD datasets ##

    ###  1. SNLI #
    if dataset == 'snli':
        TEXT_snli = torchtext.data.Field(pad_first=True, lower=True)
        LABEL_snli = torchtext.data.Field(sequential=False)

        train_snli, val_snli, test_snli = torchtext.datasets.SNLI.splits(TEXT_snli, LABEL_snli)
        all_labels = []
        all_hypotheis = []
        for example in test_snli.examples:
            all_labels.append(example.label)
            all_hypotheis.append(' '.join(example.hypothesis))

        df = pd.DataFrame(columns=['hypotheis', 'label'])
        df.label = all_labels
        df.hypotheis = all_hypotheis
        df.label = df.label.map({'neutral': 1, 'entailment': 0, 'contradiction': 2})

        train_sentences = None
        val_sentences = None
        test_sentences = df.hypotheis
        train_labels = None
        val_labels = None
        test_labels = df.label

    ###  2. IMDB #
    if dataset == 'imdb':

        df_test = pd.read_csv("./dataset/imdb/imdb.csv", delimiter=',', header=0)
        df_test = df_test.sample(5000)  # TODO
        df_test.sentiment = df_test.sentiment.map({'positive': 0, 'negative': 1})

        train_sentences = None
        val_sentences = None
        test_sentences = df_test.review.values

        train_labels = None
        val_labels = None
        test_labels = df_test.sentiment.values

    ###  3. Multi30K #
    if dataset == 'multi30k':
        TEXT_m30k = torchtext.data.Field(pad_first=True, lower=True)
        m30k_data = torchtext.data.TabularDataset(path='./dataset/multi30k/train.txt',
                                        format='csv', fields=[('text', TEXT_m30k)])

        all_text = []
        for example in m30k_data.examples:
            all_text.append(' '.join(example.text))

        df = pd.DataFrame(columns=['text', 'label'])
        df.text = all_text
        df.label = 0

        train_sentences = None
        val_sentences = None
        test_sentences = df.text
        train_labels = None
        val_labels = None
        test_labels = df.label

    ###  4. WMT16  #
    if dataset == 'wmt16':
        TEXT_wmt16 = torchtext.data.Field(pad_first=True, lower=True)
        wmt16_data = torchtext.data.TabularDataset(path='./dataset/wmt16/wmt16_sentences',
                                         format='csv', fields=[('text', TEXT_wmt16)])

        all_text = []
        for example in wmt16_data.examples:
            all_text.append(' '.join(example.text))

        df = pd.DataFrame(columns=['text', 'label'])
        df.text = all_text
        df.label = 0

        train_sentences = None
        val_sentences = None
        test_sentences = df.text
        train_labels = None
        val_labels = None
        test_labels = df.label

    ###  5. Yelp Reviews #
    if dataset == 'yelp':
        df = pd.read_csv('./dataset/yelp_review_full_csv/test.csv', delimiter=',', header=None)
        train_sentences = None
        val_sentences = None
        test_sentences = df.iloc[:, 1]
        train_labels = None
        val_labels = None
        test_labels = df.iloc[:, 0]


    return train_sentences, val_sentences, test_sentences, train_labels, val_labels, test_labels


def one_hot_tensor(y_batch_tensor, num_classes, device):
    y_tensor = torch.FloatTensor(y_batch_tensor.size(0), num_classes).fill_(0).to(device)
    y_tensor[np.arange(len(y_batch_tensor)), y_batch_tensor] = 1.0
    return y_tensor

# Function to calculate the accuracy of our predictions vs labels
def accurate_nb(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()


def cos_dist(x, y):
    ## cosine distance function
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    batch_size = x.size(0)
    c = torch.clamp(1 - cos(x.view(batch_size, -1), y.view(batch_size, -1)),
                    min=0)
    return c.mean()


# calculate Dissonance
def getDisn(alpha):
    evi = alpha - 1
    s = torch.sum(alpha, axis=1, keepdims=True)
    blf = evi / s
    idx = np.arange(alpha.shape[1])
    diss = 0
    Bal = lambda bi, bj: 1 - torch.abs(bi - bj) / (bi + bj + 1e-8)
    for i in idx:
        score_j_bal = [blf[:, j] * Bal(blf[:, j], blf[:, i]) for j in idx[idx != i]]
        score_j = [blf[:, j] for j in idx[idx != i]]
        diss += blf[:, i] * sum(score_j_bal) / (sum(score_j) + 1e-8)
    return diss

# calculate entropy
def cal_entropy(p):
    if type(p) == torch.Tensor:
        return (-p * torch.log(p + 1e-8)).sum(1)
    else:
        return (-p * np.log(p + 1e-8)).sum(1)

# PR ROC curve
def get_pr_roc(normal_score, anormal_score):

    if type(normal_score) == pd.core.series.Series:
        normal_score = normal_score.iloc[0]
    if type(anormal_score) == pd.core.series.Series:
        anormal_score = anormal_score.iloc[0]

    truth = np.zeros((len(normal_score) + len(anormal_score)))
    truth[len(normal_score):] = 1
    score = np.concatenate([normal_score, anormal_score])

    fpr, tpr, _ = roc_curve(truth, score, drop_intermediate=False)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level= 0.95,
                          pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true))), fps[cutoff]/(fps[cutoff] + tps[cutoff])

def get_performance(pos, neg, expected_ap=1 / (1 + 10.), method_name='Ours', recall_level=0.9):
    '''
    :param pos: 1's class, class to detect, outliers, or wrongly predicted
    example scores from the baseline
    :param neg: 0's class scores generated by the baseline
    :param expected_ap: this is changed from the default for failure detection
    '''
    pos = np.array(pos).reshape((-1, 1))
    neg = np.array(neg).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = roc_auc_score(labels, examples)
    aupr = average_precision_score(labels, examples)
    fpr, fdr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    # print('\t\t\t' + method_name)
    # print('FPR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fpr))
    # print('AUROC:\t\t\t{:.2f}'.format(100 * auroc))
    # print('AUPR:\t\t\t{:.2f}'.format(100 * aupr))
    # print('FDR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fdr))

    return fpr, auroc, aupr


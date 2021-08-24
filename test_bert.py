import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from utils import *
import argparse
from help.display_results import get_performance
from models import ModelWithTemperature
from transformers import BertTokenizer, BertForSequenceClassification
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import torch.nn.functional as F
import pickle
import warnings
warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(description='Test code - measure the detection performance')
    parser.add_argument('--eva_iter', default=1, type=int, help='number of passes for mc-dropout when evaluation')
    parser.add_argument('--model', type=str, choices=['base', 'manifold-smoothing', 'mc-dropout', 'temperature', 'oe'],
                        default='base')
    parser.add_argument('--seed', type=int, default=0, help='random seed for test')
    parser.add_argument('--index', type=int, default=0, help='random seed you used during training')
    parser.add_argument('--dataset', required=True, help='target dataset: sst')
    parser.add_argument('--out_dataset', required=False, help='out-of-dist dataset')
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--saved_dataset', type=str, default='y', choices = ['y','n'])
    parser.add_argument('--eps_out', default=0.001, type=float,
                        help="Perturbation size of out-of-domain adversarial training")
    parser.add_argument("--eps_y", default=0.1, type=float, help="Perturbation size of label")
    parser.add_argument('--eps_in', default=0.0001, type=float,
                        help="Perturbation size of in-domain adversarial training")
    parser.add_argument('--save_result', type=str, default='n', choices= ['y','n'])
    parser.add_argument('--evaluate_benchmark', type=str, default='y', choices = ['y','n'])
    parser.add_argument('--save_path', type=str, default='result')
    parser.add_argument('--MAX_LEN', type=int, default=150)
    parser.add_argument("--base_rate", default=5, type=int, help="base rate N:1")
    parser.add_argument('--recall_level', type=float, default=0.9)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    set_seed(args)

    if args.model == 'base':
        dirname = '{}/BERT-base-{}'.format(args.dataset, args.index)
        pretrained_dir = './{}/{}'.format(args.save_path, dirname)
        model = BertForSequenceClassification.from_pretrained(pretrained_dir)
        model.to(args.device)
        print('Load Tekenizer')

    elif args.model == 'mc-dropout':
        dirname = '{}/BERT-base-{}'.format(args.dataset, args.index)
        pretrained_dir = './{}/{}'.format(args.save_path, dirname)
        # Load a trained model and vocabulary that you have fine-tuned
        model = BertForSequenceClassification.from_pretrained(pretrained_dir)
        model.to(args.device)

    elif args.model == 'temperature':
        dirname = '{}/BERT-base-{}'.format(args.dataset, args.index)
        pretrained_dir = './{}/{}'.format(args.save_path, dirname)
        orig_model = BertForSequenceClassification.from_pretrained(pretrained_dir)
        orig_model.to(args.device)
        model = ModelWithTemperature(orig_model)
        model.to(args.device)

    elif args.model == 'manifold-smoothing':
        dirname = '{}/BERT-manifold-smoothing-{}'.format(args.dataset, args.index)
        pretrained_dir = './{}/{}'.format(args.save_path, dirname)
        model = BertForSequenceClassification.from_pretrained(pretrained_dir)
        model.to(args.device)

    elif args.model == 'oe':
        dirname = '{}/BERT-oe-{}'.format(args.dataset, args.index)
        pretrained_dir = './{}/{}'.format(args.save_path, dirname)
        model = BertForSequenceClassification.from_pretrained(pretrained_dir)
        model.to(args.device)
        print('Load Tekenizer')

    print(args.model,  dirname)

    if args.saved_dataset == 'n':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        train_sentences, val_sentences, test_sentences, train_labels, val_labels, test_labels = load_dataset(
            args.dataset)
        _, _, nt_test_sentences, _, _, nt_test_labels = load_dataset(args.out_dataset)

        val_input_ids = []
        test_input_ids = []

        for sent in val_sentences:
            encoded_sent = tokenizer.encode(
                sent,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                truncation=True,
                max_length=args.MAX_LEN,  # Truncate all sentences.
            )
            # Add the encoded sentence to the list.
            val_input_ids.append(encoded_sent)

        for sent in test_sentences:
            encoded_sent = tokenizer.encode(
                sent,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                truncation=True,
                max_length=args.MAX_LEN,  # Truncate all sentences.
            )
            # Add the encoded sentence to the list.
            test_input_ids.append(encoded_sent)

        # Pad our input tokens
        val_input_ids = pad_sequences(val_input_ids, maxlen=args.MAX_LEN, dtype="long", truncating="post", padding="post")
        test_input_ids = pad_sequences(test_input_ids, maxlen=args.MAX_LEN, dtype="long", truncating="post", padding="post")

        val_attention_masks = []
        test_attention_masks = []

        for seq in val_input_ids:
            seq_mask = [float(i > 0) for i in seq]
            val_attention_masks.append(seq_mask)
        for seq in test_input_ids:
            seq_mask = [float(i > 0) for i in seq]
            test_attention_masks.append(seq_mask)

        val_inputs = torch.tensor(val_input_ids)
        val_labels = torch.tensor(val_labels)
        val_masks = torch.tensor(val_attention_masks)

        test_inputs = torch.tensor(test_input_ids)
        test_labels = torch.tensor(test_labels)
        test_masks = torch.tensor(test_attention_masks)

        val_data = TensorDataset(val_inputs, val_masks, val_labels)
        test_data = TensorDataset(test_inputs, test_masks, test_labels)

        dataset_dir = 'dataset/test'
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        torch.save(val_data, dataset_dir + '/{}_val_in_domain.pt'.format(args.dataset))
        torch.save(test_data, dataset_dir + '/{}_test_in_domain.pt'.format(args.dataset))
        # torch.save(nt_test_data, dataset_dir + '/{}_{}_test_out_of_domain.pt'.format(args.dataset, args.out_dataset))

    else:
        dataset_dir = 'dataset/test'
        val_data = torch.load(dataset_dir + '/{}_val_in_domain.pt'.format(args.dataset))
        test_data = torch.load(dataset_dir + '/{}_test_in_domain.pt'.format(args.dataset))
        # nt_test_data = torch.load(dataset_dir + '/{}_{}_test_out_of_domain.pt'.format(args.dataset, args.out_dataset))

    ######## saved dataset
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=args.eval_batch_size)

    if args.model == 'temperature':
        model.set_temperature(val_dataloader, args)

    model.eval()

    if args.model == 'mc-dropout':
        model.apply(apply_dropout)

    # ##### validation dat
    # with torch.no_grad():
    #     for step, batch in enumerate(val_dataloader):
    #         batch = tuple(t.to(args.device) for t in batch)
    #         b_input_ids, b_input_mask, b_labels = batch
    #         total += b_labels.shape[0]
    #         batch_output = 0
    #         for j in range(args.eva_iter):
    #             if args.model == 'temperature':
    #                 current_batch = model(input_ids=b_input_ids, token_type_ids=None,
    #                                       attention_mask=b_input_mask)  # logits
    #             else:
    #                 current_batch = model(input_ids=b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[
    #                     0]  # logits
    #             batch_output = batch_output + F.softmax(current_batch, dim=1)
    #         batch_output = batch_output / args.eva_iter
    #         output_list.append(batch_output)
    #         labels_list.append(b_labels)
    #         score, predicted = batch_output.max(1)
    #         correct += predicted.eq(b_labels).sum().item()
    #
    # ###calculate accuracy and ECE
    # val_eval_accuracy = correct / total
    # print("Val Accuracy: {}".format(val_eval_accuracy))
    # ece_criterion = ECE_v2().to(args.device)
    # softmaxes_ece = torch.cat(output_list)
    # labels_ece = torch.cat(labels_list)
    # val_ece = ece_criterion(softmaxes_ece, labels_ece).item()
    # print('ECE on Val data: {}'.format(val_ece))


    #### Test data

    correct = 0
    total = 0
    output_list = []
    labels_list = []
    score_list = []
    correct_index_all = []

    ent_list = []

    ## test on in-distribution test set
    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            total += b_labels.shape[0]
            batch_output = 0
            for j in range(args.eva_iter):
                if args.model == 'temperature':
                    current_batch = model(input_ids=b_input_ids, token_type_ids=None,
                                          attention_mask=b_input_mask)  # logits
                else:
                    current_batch = model(input_ids=b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[
                        0]  # logits
                batch_output = batch_output + F.softmax(current_batch, dim=1)
            batch_output = batch_output / args.eva_iter
            output_list.append(batch_output)
            labels_list.append(b_labels)
            score, predicted = batch_output.max(1)

            correct += predicted.eq(b_labels).sum().item()

            correct_index = (predicted == b_labels)
            correct_index_all.append(correct_index)
            score_list.append(score)

            ent = cal_entropy(batch_output)
            ent_list.append(ent)

    eval_accuracy = correct / total
    print("Test Accuracy: {}".format(eval_accuracy))

    # confidence for in-distribution data
    score_in_array = torch.cat(score_list)
    # indices of data that are classified correctly
    correct_array = torch.cat(correct_index_all)
    label_array = torch.cat(labels_list)

    label_array = label_array.cpu().numpy()
    score_in_array = score_in_array.cpu().numpy()
    correct_array = correct_array.cpu().numpy()

    ent_in_array = torch.cat(ent_list)
    ent_in_array = ent_in_array.cpu().numpy()

    ent_succ_array = ent_in_array[correct_array]
    ent_fail_array = ent_in_array[~correct_array]

    in_num_examples = len(score_in_array)
    ood_MAX_NUM = in_num_examples//args.base_rate
    RECALL_LEVEL = args.recall_level

    report_result = []

    if args.evaluate_benchmark == 'y':
        ood_list = ['snli','imdb', 'multi30k', 'wmt16', 'yelp' ]
    else:
        ood_list = [args.out_dataset]

    for ood_dataset in ood_list:

        # nt_test_data = torch.load(dataset_dir + '/{}_{}_test_out_of_domain.pt'.format(args.dataset, ood_dataset))
        nt_test_data = torch.load(dataset_dir + '/{}_test_out_of_domain.pt'.format(ood_dataset))
        nt_test_sampler = SequentialSampler(nt_test_data)
        nt_test_dataloader = DataLoader(nt_test_data, sampler=nt_test_sampler, batch_size=args.eval_batch_size)
        # nt_test_dataloader = DataLoader(nt_test_data,  batch_size=args.eval_batch_size, shuffle=True)

        ### test on out-of-distribution data
        score_ood_list = []
        ent_ood_list = []

        with torch.no_grad():
            for step, batch in enumerate(nt_test_dataloader):
                if step * args.eval_batch_size > ood_MAX_NUM:
                    break
                batch = tuple(t.to(args.device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                batch_output = 0
                for j in range(args.eva_iter):
                    if args.model == 'temperature':
                        current_batch = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                    else:
                        current_batch = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[0]
                    batch_output = batch_output + F.softmax(current_batch, dim=1)
                batch_output = batch_output / args.eva_iter
                score_out, _ = batch_output.max(1)

                score_ood_list.append(score_out)

                #TODO
                ent_out = cal_entropy(batch_output)
                ent_ood_list.append(ent_out)

        score_ood_array = torch.cat(score_ood_list)
        score_ood_array = score_ood_array.cpu().numpy()

        ent_ood_array = torch.cat(ent_ood_list)
        ent_ood_array = ent_ood_array.cpu().numpy()


        #######


        true_o = np.ones(len(score_in_array) + len(score_ood_array))
        true_o[:len(score_in_array)] = 0
        true_mis = np.ones(len(score_in_array))
        true_mis[correct_array] = 0

        expected_ap = len(score_ood_array) / (len(score_in_array) + len(score_ood_array))

        ood_dect_scores = get_performance(-score_ood_array, -score_in_array, expected_ap, recall_level= RECALL_LEVEL)
        print('\n\nOOD detection, \t\t FPR{:d}: {:.4f},\t AUROC: {:.4f}\t AUPR: {:.4f} '.format(int(100 * RECALL_LEVEL),
                                                                        ood_dect_scores[0],
                                                                        ood_dect_scores[1],
                                                                        ood_dect_scores[2]))

        report_result.append([ood_dect_scores[0],ood_dect_scores[1],ood_dect_scores[2]])



        score_in_succ =  score_in_array[correct_array]
        score_in_fail =  score_in_array[~correct_array]

        expected_ap = len(score_in_fail) / (len(score_in_succ) + len(score_in_fail))
        mis_dect_scores = get_performance(-score_in_fail, -score_in_succ, expected_ap, recall_level= RECALL_LEVEL)
        print('misclassification detection,\tFPR{:d}: {:.4f},\t AUROC: {:.4f}\t AUPR: {:.4f} '.format(int(100 * RECALL_LEVEL),
                                                                        mis_dect_scores[0],
                                                                        mis_dect_scores[1],
                                                                        mis_dect_scores[2]))

        if args.save_result == 'y':
            result = {}
            result['ood_msp'] = score_ood_array
            result['in_msp'] = score_in_array

            result['succ_msp'] = score_in_succ
            result['fail_msp']= score_in_fail

            result['in_ent'] = ent_in_array
            result['ood_ent']= ent_ood_array

            result['succ_ent'] = ent_succ_array
            result['fail_ent'] =  ent_fail_array

            print('save to/%s_%s_result.pt'%(args.model,ood_dataset))

            with open(pretrained_dir + '/%s_%s_result.pt'%(args.model,ood_dataset), 'wb') as file:
                pickle.dump(result , file)

if __name__ == "__main__":
    main()

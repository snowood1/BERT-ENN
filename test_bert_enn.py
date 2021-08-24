import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from utils import *
from models import BERT_ENN
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer
import pickle
import argparse
import pandas as pd
pd.options.display.float_format = lambda x: '{:.0f}'.format(x) if round(x, 0) == x else '{:.3f}'.format(x)
pd.options.display.max_columns = 20
pd.options.display.width = 300

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Batch size for training.")
    parser.add_argument("--seed", default=0, type=int, help="Number of epochs for training.")
    parser.add_argument("--dataset", default='sst', type=str, help="dataset", choices= ['20news','trec','sst'])
    parser.add_argument('--path', type=str, default=None)
    parser.add_argument('--saved_dataset', type=str, default='y', choices = ['y','n'])
    parser.add_argument('--save_result', type=str, default='n', choices= ['y','n'])
    parser.add_argument('--evaluate_benchmark', type=str, default='y')
    parser.add_argument('--MAX_LEN', type=int, default=150)
    parser.add_argument("--base_rate", default=5, type=int, help="base rate N:1")
    parser.add_argument('--recall_level', type=float, default=0.9)
    args = parser.parse_args()

    print('\n\n-------------------------------------------------\n')

    filename = args.path.split('/')[-1]
    folder = args.path.split(filename, 1)[0]
    print('path:', args.path)
    print('folder:', folder)
    print('filename:', filename)

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args)

    if args.dataset == '20news':
        num_labels = 20
    elif args.dataset == 'sst':
        num_labels = 2
    elif args.dataset == 'trec':
        num_labels = 50

    record = vars(args)
    print(record)

    if args.saved_dataset == 'n':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        train_sentences, val_sentences, test_sentences, train_labels, val_labels, test_labels = load_dataset(
            args.dataset)
        _, _, nt_test_sentences, _, _, nt_test_labels = load_dataset(args.out_dataset)

        val_input_ids = []
        test_input_ids = []
        # nt_test_input_ids = []

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

    ######## saved dataset
    test_sampler = SequentialSampler(test_data)
    prediction_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

    val_sampler = SequentialSampler(val_data)
    validation_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=args.eval_batch_size)

    model = BERT_ENN(num_labels=num_labels)

    load_model =  torch.load(args.path)
    if type(load_model)== BERT_ENN:
        model = load_model
    else:
        model.load_state_dict(load_model)

    if torch.cuda.device_count() > 1:
        print('Does not support multiple gpus')
        # model = nn.DataParallel(model)

    model.to(args.device)

    df_test = pd.DataFrame(
        columns=['epoch', 'idxs_mask', 'in_ent', 'in_vac', 'in_dis',
                 'succ_ent', 'fail_ent', 'succ_dis', 'fail_dis',  'succ_vac', 'fail_vac', 'bnd_ent_roc', 'bnd_dis_roc'])

    df_test_avg = pd.DataFrame(
        columns=['epoch', 'test_acc', 'in_ent', 'in_vac', 'in_dis',
                 'succ_ent', 'fail_ent', 'succ_dis', 'fail_dis', 'succ_vac', 'fail_vac', 'bnd_ent_auroc', 'bnd_dis_auroc'])

    df_ood = pd.DataFrame(
        columns=['epoch', 'ood_ent', 'ood_vac', 'ood_dis', 'ood_ent_roc', 'ood_vac_roc'])

    df_ood_avg = pd.DataFrame(
        columns=['epoch', 'ood_ent', 'ood_vac', 'ood_dis',  'ent_fpr',  'ent_auroc', 'ent_aupr', 'vac_fpr', 'vac_auroc', 'vac_aupr'])


    # ##### test model on in-distribution test set

    # Put model in evaluation mode
    model.eval()
    with torch.no_grad():

        df_tmp = pd.DataFrame(
            columns=['idxs_mask', 'in_ent', 'in_vac', 'in_dis', 'succ_ent', 'fail_ent',
                     'succ_dis', 'fail_dis', 'succ_vac', 'fail_vac'])

        for batch in prediction_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(args.device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Telling the model not to compute or store gradients, saving memory and speeding up prediction

            alpha = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[0]

            model.bert(b_input_ids,
                       attention_mask=b_input_mask,
                       token_type_ids=None)

            p = alpha / alpha.sum(1, keepdim=True)

            pred = p.argmax(dim=1, keepdim=True)
            idxs_mask = pred.eq(b_labels.view_as(pred)).view(-1)
            ent = cal_entropy(p)
            disn = getDisn(alpha)
            vac_in = (num_labels / torch.sum(alpha, dim=1))
            succ_ent = ent[idxs_mask]
            fail_ent = ent[~idxs_mask]
            succ_dis = disn[idxs_mask]
            fail_dis = disn[~idxs_mask]
            succ_vac = vac_in[idxs_mask]
            fail_vac = vac_in[~idxs_mask]

            df_tmp.loc[len(df_tmp)] = [i.tolist() for i in
                                       [idxs_mask, ent, vac_in, disn, succ_ent, fail_ent,
                                        succ_dis, fail_dis, succ_vac, fail_vac]]
        in_score = df_tmp.sum()
        fpr, tpr, roc_auc = get_pr_roc(in_score['succ_ent'], in_score['fail_ent'])
        bnd_dect_ent = {'auroc': round(roc_auc, 4), 'fpr': fpr, 'tpr': tpr}

        fpr, tpr, roc_auc = get_pr_roc(in_score['succ_dis'], in_score['fail_dis'])
        bnd_dect_dis = {'auroc': round(roc_auc, 4), 'fpr': fpr, 'tpr': tpr}

        df_test.loc[len(df_test)] = [0, *in_score, bnd_dect_ent, bnd_dect_dis]
        df_test_avg.loc[len(df_test_avg)] = [0, *in_score.apply(np.average), bnd_dect_ent['auroc'],
                                             bnd_dect_dis['auroc']]

    df = df_test_avg.tail(1)
    test_log = 'Test in:\tacc: {:.3f},\t' \
               'ent: {:.3f}({:.3f}/{:.3f}),\t' \
               'vac:  {:.3f}({:.3f}/{:.3f}),\t'\
               'disn: {:.3f}({:.3f}/{:.3f}),\t' \
               'bnd_auroc: [ent {:.3f}, disn {:.3f}]'.format(df['test_acc'].iloc[0],
                                                             df['in_ent'].iloc[0], df['succ_ent'].iloc[0], df['fail_ent'].iloc[0],
                                                             df['in_vac'].iloc[0], df['succ_vac'].iloc[0], df['fail_vac'].iloc[0],
                                                             df['in_dis'].iloc[0], df['succ_dis'].iloc[0], df['fail_dis'].iloc[0],
                                                             df['bnd_ent_auroc'].iloc[0],df['bnd_dis_auroc'].iloc[0])
    print(test_log)


    ### test on out-of-distribution data  ###################

    report_result = []

    in_num_examples = len(in_score['in_ent'])
    ood_MAX_NUM = in_num_examples//args.base_rate
    RECALL_LEVEL = args.recall_level

    if args.evaluate_benchmark == 'y':
        ood_list = ['snli','imdb', 'multi30k', 'wmt16', 'yelp' ]
    else:
        ood_list = [args.out_dataset]

    for ood_dataset in ood_list:
        nt_test_data = torch.load('dataset/test/{}_test_out_of_domain.pt'.format(ood_dataset))
        nt_test_sampler = SequentialSampler(nt_test_data)
        nt_test_dataloader = DataLoader(nt_test_data, sampler=nt_test_sampler, batch_size=args.eval_batch_size)

        model.eval()
        with torch.no_grad():
            df_tmp = pd.DataFrame(columns=['ood_ent', 'ood_vac', 'ood_dis'])

            for step, batch in enumerate(nt_test_dataloader):
                batch = tuple(t.to(args.device) for t in batch)

                if step * args.eval_batch_size > ood_MAX_NUM:
                    break

                b_input_ids, b_input_mask, b_labels = batch

                alpha_bar = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[0]
                p_bar = alpha_bar / alpha_bar.sum(1, keepdim=True)

                ent_bar = cal_entropy(p_bar)
                disn_bar = getDisn(alpha_bar)
                vac_bar = num_labels / torch.sum(alpha_bar, dim=1)

                df_tmp.loc[len(df_tmp)] = [i.tolist() for i in [ent_bar, vac_bar, disn_bar]]

            out_score = df_tmp.sum()

            ood_num_examples,   in_num_examples = len(out_score['ood_ent']), len(in_score['in_ent'])
            expected_ap = ood_num_examples / (ood_num_examples + in_num_examples)

            a = get_performance(out_score['ood_ent'], in_score['in_ent'], expected_ap, recall_level=RECALL_LEVEL)
            b = get_performance(out_score['ood_vac'], in_score['in_vac'], expected_ap, recall_level=RECALL_LEVEL)

            ent_fpr, ent_auroc, ent_aupr = a[0], a[1], a[2]
            vac_fpr, vac_auroc, vac_aupr = b[0], b[1], b[2]

            df_ood.loc[len(df_ood)] = [ood_dataset, *out_score, ent_auroc, vac_auroc]
            df_ood_avg.loc[len(df_ood_avg)] = [ood_dataset, *out_score.apply(np.average), ent_fpr, ent_auroc, ent_aupr,
                                               vac_fpr, vac_auroc, vac_aupr]


        df = df_ood_avg.tail(1)
        ood_log = 'Test out:\t{:10s}\tent: {:.3f},\t\t\t' \
                           'vac: {:.3f},\tdisn: {:.3f}\t\t\t' \
                  'ent: [fpr {:.3f}, auroc {:.3f}, aupr {:.3f}]\t'\
                  'vac: [fpr {:.3f}, auroc {:.3f}, aupr {:.3f}]'.format(ood_dataset,
                                                                  df['ood_ent'].iloc[0],
                                                                  df['ood_vac'].iloc[0],
                                                                  df['ood_dis'].iloc[0],
                                                                  df['ent_fpr'].iloc[0], df['ent_auroc'].iloc[0], df['ent_aupr'].iloc[0],
                                                                  df['vac_fpr'].iloc[0], df['vac_auroc'].iloc[0], df['vac_aupr'].iloc[0])

        print(ood_log)
        report_result.append([df['vac_fpr'].iloc[0], df['vac_auroc'].iloc[0], df['vac_aupr'].iloc[0]])

        if args.save_result == 'y':
            result = {}

            in_score_df = in_score.to_frame().T
            for key in in_score_df:
                result[key] = in_score_df[key][0]

            out_score_df = out_score.to_frame().T
            for key in out_score_df:
                result[key] = out_score_df[key][0]


            out_dir = '{}{}_{}_result.pt'.format(folder,args.dataset,ood_dataset)
            print('save to %s'%out_dir)
            with open(out_dir, "wb") as file:
                pickle.dump(result, file)



if __name__ == "__main__":
    main()

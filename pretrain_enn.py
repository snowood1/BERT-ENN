import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer
from models import BERT_ENN
import argparse
from utils import *
import pandas as pd
pd.options.display.float_format = lambda x: '{:.0f}'.format(x) if round(x, 0) == x else '{:.3f}'.format(x)
pd.options.display.max_columns = 20
pd.options.display.width = 300


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--train_batch_size", default=128, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=512, type=int, help="Batch size for training.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs for training.")
    parser.add_argument("--seed", default=0, type=int, help="Number of epochs for training.")
    parser.add_argument("--dataset", default='wos', type=str, help="dataset")
    parser.add_argument("--out_dataset", default='agnews', type=str, help="outlier dataset")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--beta_in", default=0, type=float, help="Weight of on manifold reg")
    parser.add_argument('--saved_dataset', type=str, default='y', choices = ['y','n'],
                        help='whether save the preprocessed pt file of the dataset')
    parser.add_argument('--warm_up', type=int, default=3, help='warn up epochs')
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--MAX_LEN', type=int, default=150)
    parser.add_argument("--base_rate", default=1, type=int, help="base rate N:1")
    parser.add_argument('--recall_level', type=float, default=0.9)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    set_seed(args)

    print('\n\n-------------------------------------------------\n')
    record = vars(args)
    print(record)

    # load dataset
    if args.saved_dataset == 'n':
        train_sentences, val_sentences, test_sentences, train_labels, \
        val_labels, test_labels = load_dataset(args.dataset)

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        train_input_ids = []
        val_input_ids = []
        test_input_ids = []

        for sent in train_sentences:
            encoded_sent = tokenizer.encode(
                sent,
                add_special_tokens=True,
                max_length=args.MAX_LEN,
                truncation=True
            )
            # Add the encoded sentence to the list.
            train_input_ids.append(encoded_sent)

        for sent in val_sentences:
            encoded_sent = tokenizer.encode(
                sent,
                add_special_tokens=True,
                max_length=args.MAX_LEN,
                truncation=True
            )
            val_input_ids.append(encoded_sent)

        for sent in test_sentences:
            encoded_sent = tokenizer.encode(
                sent,
                add_special_tokens=True,
                max_length=args.MAX_LEN,
                truncation=True
            )
            test_input_ids.append(encoded_sent)

        # Pad our input tokens
        train_input_ids = pad_sequences(train_input_ids, maxlen=args.MAX_LEN, dtype="long", truncating="post",
                                        padding="post")
        val_input_ids = pad_sequences(val_input_ids, maxlen=args.MAX_LEN, dtype="long", truncating="post", padding="post")
        test_input_ids = pad_sequences(test_input_ids, maxlen=args.MAX_LEN, dtype="long", truncating="post", padding="post")
        # Create attention masks
        train_attention_masks = []
        val_attention_masks = []
        test_attention_masks = []

        # Create a mask of 1s for each token followed by 0s for padding
        for seq in train_input_ids:
            seq_mask = [float(i > 0) for i in seq]
            train_attention_masks.append(seq_mask)
        for seq in val_input_ids:
            seq_mask = [float(i > 0) for i in seq]
            val_attention_masks.append(seq_mask)
        for seq in test_input_ids:
            seq_mask = [float(i > 0) for i in seq]
            test_attention_masks.append(seq_mask)

        # Convert all of our data into torch tensors, the required datatype for our model

        train_inputs = torch.tensor(train_input_ids)
        validation_inputs = torch.tensor(val_input_ids)
        train_labels = torch.tensor(train_labels)
        validation_labels = torch.tensor(val_labels)
        train_masks = torch.tensor(train_attention_masks)
        validation_masks = torch.tensor(val_attention_masks)
        test_inputs = torch.tensor(test_input_ids)
        test_labels = torch.tensor(test_labels)
        test_masks = torch.tensor(test_attention_masks)

        # Create an iterator of our data with torch DataLoader.

        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
        prediction_data = TensorDataset(test_inputs, test_masks, test_labels)

        dataset_dir = 'dataset/{}'.format(args.dataset)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        torch.save(train_data, dataset_dir + '/train.pt')
        torch.save(validation_data, dataset_dir + '/val.pt')
        torch.save(prediction_data, dataset_dir + '/test.pt')

    else:
        dataset_dir = 'dataset/{}'.format(args.dataset)
        train_data = torch.load(dataset_dir + '/train.pt')
        validation_data = torch.load(dataset_dir + '/val.pt')
        prediction_data = torch.load(dataset_dir + '/test.pt')

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=args.eval_batch_size)

    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=args.eval_batch_size)

    # Todo
    nt_test_data = torch.load('dataset/test/{}_test_out_of_domain.pt'.format(args.out_dataset))
    nt_test_sampler = SequentialSampler(nt_test_data)
    nt_test_dataloader = DataLoader(nt_test_data, sampler=nt_test_sampler, batch_size=args.eval_batch_size)

    ###################################

    if args.dataset == '20news':
        num_labels = 20
    elif args.dataset == 'sst':
        num_labels = 2
    elif args.dataset == 'trec':
        num_labels = 50

    model = BERT_ENN(num_labels=num_labels)

    if torch.cuda.device_count() > 1:
        print('Does not support multiple gpus')
        # model = nn.DataParallel(model)

    model.to(args.device)

    #######train model

    # 1  #############
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]

    optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=args.lr, eps=1e-9)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1)
    # t_total = len(train_dataloader) * args.epochs
    # Store our loss and accuracy for plotting

    df_train = pd.DataFrame(
        columns=['epoch', 'batch', 'train_acc', 'train_in_vac', 'train_ood_vac', 'loss'])

    df_train_avg = pd.DataFrame(
        columns=['epoch', 'train_acc', 'train_in_vac', 'train_ood_vac', 'loss'])

    df_test = pd.DataFrame(
        columns=['epoch', 'idxs_mask', 'in_ent', 'in_vac', 'in_dis',
                 'succ_ent', 'fail_ent', 'succ_dis', 'fail_dis', 'bnd_ent_roc', 'bnd_dis_roc'])

    df_test_avg = pd.DataFrame(
        columns=['epoch', 'test_acc', 'in_ent', 'in_vac', 'in_dis',
                 'succ_ent', 'fail_ent', 'succ_dis', 'fail_dis', 'bnd_ent_auroc', 'bnd_dis_auroc'])

    for epoch in range(args.epochs):
        print('\n ====> epoch %s ====>' % epoch)
        # Training
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            b_labels_onehot = one_hot_tensor(b_labels, num_labels, args.device)

            alpha, _, loss_mse = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask,
                                           labels=b_labels_onehot)

            s = alpha.sum(1, keepdim=True)
            p = alpha / s
            vac = num_labels / s

            if args.beta_in:
                loss_on = args.beta_in * vac.mean()
            else:
                loss_on = 0

            pred = p.argmax(dim=1, keepdim=True)
            accuracy = pred.eq(b_labels.view_as(pred)).sum().item() / b_input_ids.size(0)

            if torch.cuda.device_count() > 1:
                loss_mse = loss_mse.mean()
                loss_on = loss_on.mean()

            optimizer.zero_grad()
            (loss_mse + loss_on).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_off = torch.tensor(0)
            vac_bar_mean = 0

            loss = loss_mse + loss_on + loss_off
            df_train.loc[len(df_train)] = [epoch, step, accuracy, vac.mean().item(), vac_bar_mean, loss.item()]

        df_train_avg.loc[len(df_train_avg)] = df_train[df_train.epoch == epoch].mean()

        df = df_train_avg.tail(1)
        train_log = 'Train:\t\tacc: {:.3f},\t' \
                    'loss: {:.4f}\t\t\t' \
                    'in_vac: {:.3f},\t' \
                    'ood_vac: {:.3f}'.format(df['train_acc'].iloc[0],
                                             df['loss'].iloc[0],
                                             df['train_in_vac'].iloc[0],
                                             df['train_ood_vac'].iloc[0])
        print(train_log)

        # Validation
        # Put model in evaluation mode to evaluate loss on the validation set
        # model.eval()
        # with torch.no_grad():
        #     # Tracking variables
        #     eval_accurate_nb = 0
        #     nb_eval_examples = 0
        #
        #     # Evaluate data for one epoch
        #     for batch in validation_dataloader:
        #         # Add batch to GPU
        #         batch = tuple(t.to(args.device) for t in batch)
        #         # Unpack the inputs from our dataloader
        #         b_input_ids, b_input_mask, b_labels = batch
        #         # Telling the model not to compute or store gradients, saving memory and speeding up validation
        #         # with torch.no_grad():
        #             # Forward pass, calculate logit predictions
        #         alpha_val = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[0]
        #             # Move logits and labels to CPU
        #         alpha_val = alpha_val.detach().cpu().numpy()
        #         label_ids = b_labels.to('cpu').numpy()
        #
        #         tmp_eval_nb = accurate_nb(alpha_val, label_ids)
        #
        #         eval_accurate_nb += tmp_eval_nb
        #         nb_eval_examples += label_ids.shape[0]
        #     eval_accuracy = eval_accurate_nb / nb_eval_examples
        #     print("Validation Accuracy: {:.3f}".format(eval_accuracy))
        #
        #     scheduler.step(eval_accuracy)


        # ##### test model on in-distribution test set

        # Put model in evaluation mode
        model.eval()
        with torch.no_grad():

            df_tmp = pd.DataFrame(
                columns=['idxs_mask', 'in_ent', 'in_vac', 'in_dis', 'succ_ent', 'fail_ent',
                         'succ_dis', 'fail_dis'])

            for batch in prediction_dataloader:
                batch = tuple(t.to(args.device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                alpha = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[0]
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

                df_tmp.loc[len(df_tmp)] = [i.tolist() for i in
                                           [idxs_mask, ent, vac_in, disn, succ_ent, fail_ent,
                                            succ_dis, fail_dis]]
            in_score = df_tmp.sum()
            fpr, tpr, roc_auc = get_pr_roc(in_score['succ_ent'], in_score['fail_ent'])
            bnd_dect_ent = {'auroc': round(roc_auc, 4), 'fpr': fpr, 'tpr': tpr}

            fpr, tpr, roc_auc = get_pr_roc(in_score['succ_dis'], in_score['fail_dis'])
            bnd_dect_dis = {'auroc': round(roc_auc, 4), 'fpr': fpr, 'tpr': tpr}

            df_test.loc[len(df_test)] = [epoch, *in_score, bnd_dect_ent, bnd_dect_dis]
            df_test_avg.loc[len(df_test_avg)] = [epoch, *in_score.apply(np.average), bnd_dect_ent['auroc'],
                                                 bnd_dect_dis['auroc']]

        df = df_test_avg.tail(1)
        test_log = 'Test in:\tacc: {:.3f},\t' \
                   'ent: {:.3f}({:.3f}/{:.3f}),\t' \
                   'vac: {:.3f},\t' \
                   'disn: {:.3f}({:.3f}/{:.3f}),\t' \
                   'bnd_auroc: [ent {:.3f}, disn {:.3f}]'.format(df['test_acc'].iloc[0],
                                                                 df['in_ent'].iloc[0],
                                                                 df['succ_ent'].iloc[0],
                                                                 df['fail_ent'].iloc[0],
                                                                 df['in_vac'].iloc[0],
                                                                 df['in_dis'].iloc[0],
                                                                 df['succ_dis'].iloc[0],
                                                                 df['fail_dis'].iloc[0],
                                                                 df['succ_ent'].iloc[0],
                                                                 df['bnd_ent_auroc'].iloc[0],
                                                                 df['bnd_dis_auroc'].iloc[0])
        print(test_log)

        dirname = '{}/BERT-ENN-pretrain-{}'.format(args.dataset, args.seed)

        output_dir = './model_save/{}'.format(dirname)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print("Saving model to %s" % output_dir)
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), output_dir + '/%s.pt' % epoch)

        with open('{}/log.txt'.format(output_dir), "a") as file:
            print('Epoch:%d' % epoch, file=file)
            print(train_log, file=file)
            print(test_log, file=file)



if __name__ == "__main__":
    import time

    start = time.time()
    main()
    print('\nTime', time.time() - start)
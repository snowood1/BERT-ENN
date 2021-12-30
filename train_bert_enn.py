import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import pickle
import argparse
from utils import set_seed, accurate_nb, one_hot_tensor, cal_entropy, getDisn, get_performance, get_pr_roc
from models import BERT_ENN, off_manifold_samples
import pandas as pd
pd.options.display.float_format = lambda x: '{:.0f}'.format(x) if round(x, 0) == x else '{:.3f}'.format(x)
pd.options.display.max_columns = 20
pd.options.display.width = 300


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Batch size for training.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs for training.")
    parser.add_argument("--seed", default=0, type=int, help="Number of epochs for training.")
    parser.add_argument("--dataset", default='sst', type=str, help="dataset", choices = ['20news','trec','sst'])
    parser.add_argument("--out_dataset", default='multi30k', type=str, help="outlier dataset")

    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")

    parser.add_argument("--beta_in", default=0, type=float, help="Weight of in-distribution reg")
    parser.add_argument("--beta_oe", default=1, type=float, help="Weight of OE reg")
    parser.add_argument("--beta_off", default=0.1, type=float, help="Weight of off manifold reg")
    parser.add_argument('--eps_out', default=0.01, type=float,
                        help="Perturbation size of out-of-domain adversarial training")
    parser.add_argument('--warm_up', type=int, default=2,
                        help='warn up epochs')
    parser.add_argument('--grad_clip', type=float, default=1)
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--evaluate_benchmark', type=str, default='y', help='whether to evaluate on all the OOD datasets. This will overwrite the option --out_dataset')
    parser.add_argument('--MAX_LEN', type=int, default=150)
    parser.add_argument("--base_rate", default=5, type=int, help="base rate N:1")
    parser.add_argument('--recall_level', type=float, default=0.9)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    set_seed(args)

    record = vars(args)
    print('\n--------------------\n')
    print(record)

    dirname = '{}/BERT-ENN-w2adv-{}-on-{}-oe-{}-off-{}'.format(args.dataset, args.seed, args.beta_in, args.beta_oe, args.beta_off)

    output_dir = './model_save/{}'.format(dirname)
    print(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open('{}/log.txt'.format(output_dir), "a") as file:
        print('Train  BERT-ENN',  file=file)
        print(record, file=file)

    print('Loading saved dataset checkpoints for training...')
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

    ood_data = torch.load('dataset/wikitext2.pt')
    ood_dataloader = DataLoader(ood_data, batch_size=args.train_batch_size, shuffle=True)

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

    # on_manifold = on_manifold_samples(epsilon_x=args.eps_in, epsilon_y=args.eps_y)
    off_manifold = off_manifold_samples(eps=args.eps_out)

    model = BERT_ENN(num_labels=num_labels)

    if args.pretrain:
        # model = torch.load(args.pretrain)
        model.load_state_dict(torch.load(args.pretrain))


    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(args.device)

    #####  train model

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

    df_train = pd.DataFrame(
        columns=['epoch', 'batch', 'train_acc', 'train_in_vac', 'train_ood_vac', 'loss'])
    df_train_avg = pd.DataFrame(
        columns=['epoch', 'train_acc', 'train_in_vac', 'train_ood_vac', 'loss'])

    df_test = pd.DataFrame(
        columns=['epoch', 'idxs_mask', 'in_ent', 'in_vac', 'in_dis',
                 'succ_ent', 'fail_ent', 'succ_dis', 'fail_dis', 'succ_vac', 'fail_vac', 'bnd_ent_roc', 'bnd_dis_roc'])

    df_test_avg = pd.DataFrame(
        columns=['epoch', 'test_acc', 'in_ent', 'in_vac', 'in_dis',
                 'succ_ent', 'fail_ent', 'succ_dis', 'fail_dis', 'succ_vac', 'fail_vac', 'bnd_ent_auroc',
                 'bnd_dis_auroc'])

    df_ood = pd.DataFrame(
        columns=['epoch', 'ood_ent', 'ood_vac', 'ood_dis', 'ood_ent_roc', 'ood_vac_roc'])

    df_ood_avg = pd.DataFrame(
        columns=['epoch', 'ood_ent', 'ood_vac', 'ood_dis', 'ent_fpr', 'ent_auroc', 'ent_aupr', 'vac_fpr', 'vac_auroc',
                 'vac_aupr'])

    # trange is a tqdm wrapper around the normal python range
    for epoch in range(args.epochs):
        print('\n ====> epoch %s ====>' % epoch)
        # Training

        model.train()

        # Train the data for one epoch
        for step, (batch, batch_oe) in enumerate(zip(train_dataloader, ood_dataloader)):
            batch = tuple(t.to(args.device) for t in batch)

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            if epoch > args.warm_up:
                ##  1. adv ##

                if args.beta_off:

                    off_manifold_x = off_manifold.generate(model, b_input_ids, b_input_mask, b_labels_onehot)
                    model.train()

                    alpha_bar = model(token_type_ids=None, attention_mask=b_input_mask, inputs_embeds=off_manifold_x)[0]
                    vac_bar = num_labels / alpha_bar.sum(1, keepdim=True)
                    vac_bar_mean = vac_bar.mean().item()

                    loss_off = -args.beta_off * vac_bar.mean()

                else:
                    loss_off = torch.tensor(0.0).cuda()
                    vac_bar_mean = 0

                ##  2. OE  ##

                if args.beta_oe:

                    batch_oe = tuple(t.to(args.device) for t in batch_oe)
                    oe_input_ids, oe_input_mask, _ = batch_oe

                    alpha_oe = model(oe_input_ids, attention_mask=oe_input_mask)[0]
                    vac_oe = num_labels / alpha_oe.sum(1, keepdim=True)

                    loss_off -= args.beta_oe * vac_oe.mean()
                else:
                    loss_off -= torch.tensor(0.0).cuda()

                if torch.cuda.device_count() > 1:
                    loss_off = loss_off.mean()
                    
                if loss_off > 0:
                    optimizer.zero_grad()
                    loss_off.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()

            else:
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
        model.eval()
        with torch.no_grad():
            # Tracking variables
            eval_accurate_nb = 0
            nb_eval_examples = 0

            for batch in validation_dataloader:
                batch = tuple(t.to(args.device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                alpha_val = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[0]

                vac_val = (num_labels / torch.sum(alpha_val, dim=1))

                alpha_val = alpha_val.detach().cpu().numpy()
                vac_val =  vac_val.mean().detach().cpu().numpy()

                label_ids = b_labels.to('cpu').numpy()
                tmp_eval_nb = accurate_nb(alpha_val, label_ids)

                eval_accurate_nb += tmp_eval_nb
                nb_eval_examples += label_ids.shape[0]
            eval_accuracy = eval_accurate_nb / nb_eval_examples
            print("Valid:\t\tacc: {:.3f}, \tvac: {:.3f},".format(eval_accuracy, vac_val))

            # scheduler.step(eval_accuracy)

        # ##### test model on in-distribution test set

        # Put model in evaluation mode
        model.eval()
        with torch.no_grad():

            df_tmp = pd.DataFrame(
                columns=['idxs_mask', 'in_ent', 'in_vac', 'in_dis', 'succ_ent', 'fail_ent',
                         'succ_dis', 'fail_dis', 'succ_vac', 'fail_vac'])

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
                succ_vac = vac_in[idxs_mask]
                fail_vac = vac_in[~idxs_mask]

                df_tmp.loc[len(df_tmp)] = [i.tolist() for i in
                                           [idxs_mask, ent, vac_in, disn, succ_ent, fail_ent,
                                            succ_dis, fail_dis, succ_vac, fail_vac, ]]
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
                   'vac: {:.3f}({:.3f}/{:.3f}),\t' \
                   'disn: {:.3f}({:.3f}/{:.3f}),\t' \
                   'bnd_auroc: [ent {:.3f}, disn {:.3f}]'.format(df['test_acc'].iloc[0],
                                                                 df['in_ent'].iloc[0],
                                                                 df['succ_ent'].iloc[0],
                                                                 df['fail_ent'].iloc[0],
                                                                 df['in_vac'].iloc[0],
                                                                 df['succ_vac'].iloc[0],
                                                                 df['fail_vac'].iloc[0],
                                                                 df['in_dis'].iloc[0],
                                                                 df['succ_dis'].iloc[0],
                                                                 df['fail_dis'].iloc[0],
                                                                 df['bnd_ent_auroc'].iloc[0],
                                                                 df['bnd_dis_auroc'].iloc[0])
        print(test_log)

        ## test on out-of-distribution data  ###################

        if args.evaluate_benchmark == 'n':
            in_num_examples = len(in_score['in_ent'])
            ood_MAX_NUM = in_num_examples // args.base_rate
            RECALL_LEVEL = args.recall_level

            model.eval()
            with torch.no_grad():
                df_tmp = pd.DataFrame(columns=['ood_ent', 'ood_vac', 'ood_dis'])

                for step, batch in enumerate(nt_test_dataloader):
                    if step * args.eval_batch_size > ood_MAX_NUM:
                        break
                    batch = tuple(t.to(args.device) for t in batch)
                    b_input_ids, b_input_mask, b_labels = batch

                    alpha_bar = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[0]
                    p_bar = alpha_bar / alpha_bar.sum(1, keepdim=True)

                    ent_bar = cal_entropy(p_bar)
                    disn_bar = getDisn(alpha_bar)
                    vac_bar = num_labels / torch.sum(alpha_bar, dim=1)

                    df_tmp.loc[len(df_tmp)] = [i.tolist() for i in [ent_bar, vac_bar, disn_bar]]

                out_score = df_tmp.sum()

                ood_num_examples, in_num_examples = len(out_score['ood_ent']), len(in_score['in_ent'])
                expected_ap = ood_num_examples / (ood_num_examples + in_num_examples)

                a = get_performance(out_score['ood_ent'], in_score['in_ent'], expected_ap, recall_level=RECALL_LEVEL)
                b = get_performance(out_score['ood_vac'], in_score['in_vac'], expected_ap, recall_level=RECALL_LEVEL)

                ent_fpr, ent_auroc, ent_aupr = a[0], a[1], a[2]
                vac_fpr, vac_auroc, vac_aupr = b[0], b[1], b[2]

                df_ood.loc[len(df_ood)] = [epoch, *out_score, ent_auroc, vac_auroc]
                df_ood_avg.loc[len(df_ood_avg)] = [epoch, *out_score.apply(np.average), ent_fpr, ent_auroc, ent_aupr,
                                                   vac_fpr, vac_auroc, vac_aupr]

            df = df_ood_avg.tail(1)
            ood_log = 'Test out:\t{:10s}\tent: {:.3f},\t\t\t' \
                      'vac: {:.3f},\tdisn: {:.3f}\t\t\t' \
                      'ent: [fpr {:.3f}, auroc {:.3f}, aupr {:.3f}]\t' \
                      'vac: [fpr {:.3f}, auroc {:.3f}, aupr {:.3f}]'.format(args.out_dataset,
                                                                            df['ood_ent'].iloc[0],
                                                                            df['ood_vac'].iloc[0],
                                                                            df['ood_dis'].iloc[0],
                                                                            df['ent_fpr'].iloc[0], df['ent_auroc'].iloc[0],
                                                                            df['ent_aupr'].iloc[0],
                                                                            df['vac_fpr'].iloc[0], df['vac_auroc'].iloc[0],
                                                                            df['vac_aupr'].iloc[0])
            # ood_log_list.append(ood_log)
            print(ood_log)

        with open('{}/log.txt'.format(output_dir), "a") as file:
            print('Epoch:%d' % epoch, file=file)
            print(train_log, file=file)
            print(test_log, file=file)
            if args.evaluate_benchmark == 'n':
                print(ood_log, file=file)

        for df_record in ['df_train', 'df_train_avg', 'df_test', 'df_test_avg', 'df_ood', 'df_ood_avg']:
            record[df_record] = eval(df_record)
        record['epoch'] = epoch
        with open('{}/record.pt'.format(output_dir), "wb") as file:
            pickle.dump(record, file)

        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), output_dir + '/%s.pt' % epoch)


#
if __name__ == "__main__":
    import time

    start = time.time()
    main()
    print('\nTime', time.time() - start)

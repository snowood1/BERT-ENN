import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from transformers  import BertTokenizer, BertConfig
from transformers  import AdamW, BertForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd.gradcheck import zero_gradients
import argparse
import random
from utils import *
import os 



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Batch size for training.")
    parser.add_argument("--epochs", default=5, type=int, help="Number of epochs for training.")
    parser.add_argument("--seed", default=0, type=int, help="Number of epochs for training.")
    parser.add_argument("--dataset", default='trec', type=str, help="dataset", choices = ['20news','trec','sst'])
    parser.add_argument("--MAX_LEN", default=150, type=int)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--saved_dataset', type=str, default='y', help='whether save the preprocessed pt file of the dataset')  # 'n'

    args = parser.parse_args()
    print(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    set_seed(args)

    # load dataset
    if args.saved_dataset == 'n':
        train_sentences, val_sentences, test_sentences, train_labels, val_labels, test_labels = load_dataset(args.dataset)

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        train_input_ids = []
        val_input_ids = []
        test_input_ids = []

        for sent in train_sentences:

            encoded_sent = tokenizer.encode(
                                sent,
                                add_special_tokens = True,
                                max_length = args.MAX_LEN,
                truncation=True)
        # Add the encoded sentence to the list.
            train_input_ids.append(encoded_sent)

        for sent in val_sentences:
            encoded_sent = tokenizer.encode(
                                sent,                    
                                add_special_tokens = True, 
                                max_length = args.MAX_LEN,
                truncation=True
                        )
            val_input_ids.append(encoded_sent)

        for sent in test_sentences:
            encoded_sent = tokenizer.encode(
                                sent,                     
                                add_special_tokens = True, 
                                max_length = args.MAX_LEN,
                truncation=True
                        )
            test_input_ids.append(encoded_sent)

        # Pad our input tokens
        train_input_ids = pad_sequences(train_input_ids, maxlen=args.MAX_LEN, dtype="long", truncating="post", padding="post")
        val_input_ids = pad_sequences(val_input_ids, maxlen=args.MAX_LEN, dtype="long", truncating="post", padding="post")
        test_input_ids = pad_sequences(test_input_ids, maxlen=args.MAX_LEN, dtype="long", truncating="post", padding="post")
        # Create attention masks
        train_attention_masks = []
        val_attention_masks = []
        test_attention_masks = []

        # Create a mask of 1s for each token followed by 0s for padding
        for seq in train_input_ids:
            seq_mask = [float(i>0) for i in seq]
            train_attention_masks.append(seq_mask)
        for seq in val_input_ids:
            seq_mask = [float(i>0) for i in seq]
            val_attention_masks.append(seq_mask)
        for seq in test_input_ids:
            seq_mask = [float(i>0) for i in seq]
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
            
        torch.save(train_data, dataset_dir+'/train.pt')
        torch.save(validation_data, dataset_dir+'/val.pt')
        torch.save(prediction_data, dataset_dir+'/test.pt')

    else:
        dataset_dir = 'dataset/{}'.format(args.dataset)
        train_data = torch.load(dataset_dir+'/train.pt')
        validation_data = torch.load(dataset_dir+'/val.pt')
        prediction_data = torch.load(dataset_dir+'/test.pt')


    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=args.eval_batch_size)

    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=args.eval_batch_size)


    if args.dataset == '20news':
        num_labels = 20
    elif args.dataset == 'sst':
        num_labels = 2
    elif args.dataset == 'trec':
        num_labels = 50


    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels= num_labels, output_hidden_states=True)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
    model.to(args.device)

#######train model 

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay_rate': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.0}
    ]

    optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=args.lr, eps=1e-9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1)
    # t_total = len(train_dataloader) * args.epochs
    # Store our loss and accuracy for plotting

    for epoch in trange(args.epochs, desc="Epoch"): 
    # Training
        # Set our model to training mode (as opposed to evaluation mode)
        # Tracking variables
        tr_loss =  0
        nb_tr_examples, nb_tr_steps = 0, 0
        model.train()

        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            
            # Add batch to GPU
            batch = tuple(t.to(args.device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            
            loss_ce = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)[0]
            if torch.cuda.device_count() > 1:
                loss_ce = loss_ce.mean()
            loss_ce.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient
            optimizer.step()

            # Update tracking variables
            tr_loss += loss_ce.item()
        
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

        print("Train cross entropy loss: {}".format(tr_loss/nb_tr_steps))
        

        # Validation
        # Put model in evaluation mode to evaluate loss on the validation set
        model.eval()
        # Tracking variables 
        eval_accurate_nb = 0
        nb_eval_examples = 0
        logits_list = []
        labels_list = []

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(args.device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Telling the model not to compute or store gradients, saving memory and speeding up validation
            with torch.no_grad():
            # Forward pass, calculate logit predictions
                logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[0] 
                logits_list.append(logits)
                labels_list.append(b_labels)
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            tmp_eval_nb = accurate_nb(logits, label_ids)
    
            eval_accurate_nb += tmp_eval_nb
            nb_eval_examples += label_ids.shape[0]
        eval_accuracy = eval_accurate_nb/nb_eval_examples
        print("Validation Accuracy: {}".format(eval_accuracy))
        scheduler.step(eval_accuracy)

# ##### test model on test data
        # Put model in evaluation mode
        model.eval()
        # Tracking variables 
        eval_accurate_nb = 0
        nb_test_examples = 0
        logits_list = []
        labels_list = []
        # Predict 
        for batch in prediction_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(args.device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Telling the model not to compute or store gradients, saving memory and speeding up prediction
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[0]
                logits_list.append(logits)
                labels_list.append(b_labels)
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            tmp_eval_nb = accurate_nb(logits, label_ids)
            eval_accurate_nb += tmp_eval_nb
            nb_test_examples += label_ids.shape[0]

        print("Test Accuracy: {}".format(eval_accurate_nb/nb_test_examples))

        dirname = '{}/BERT-base-{}'.format(args.dataset, args.seed)

        output_dir = './model_save/{}'.format(dirname)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print("Saving model to %s" % output_dir)
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(output_dir)
        

if __name__ == "__main__":
    import time
    start = time.time()
    main()
    print('\nTime', time.time() - start)

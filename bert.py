import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers  import BertForSequenceClassification
from tqdm import trange
import argparse
from utils import set_seed, accurate_nb



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

    args = parser.parse_args()
    print(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    set_seed(args)

    print('Loading saved dataset checkpoints for training...')
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

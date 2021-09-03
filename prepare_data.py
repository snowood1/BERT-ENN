import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from utils import *
import argparse
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset

import warnings
warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(description='Test code - measure the detection performance')
    parser.add_argument('--seed', type=int, default=0, help='random seed for test')
    parser.add_argument('--MAX_LEN', type=int, default=150)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    set_seed(args)

    outliers = 'wikitext2'
    ood_list = ['snli', 'imdb', 'multi30k', 'wmt16', 'yelp']

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    def sentence_to_TensorDataset(test_sentences, test_labels, tokenizer=tokenizer, max_len=args.MAX_LEN):
        test_input_ids = []
        for sent in test_sentences:
            encoded_sent = tokenizer.encode(
                sent,
                add_special_tokens=True,
                max_length=max_len,
                truncation=True
            )
            test_input_ids.append(encoded_sent)

        # Pad our input tokens
        test_input_ids = pad_sequences(test_input_ids, maxlen=max_len, dtype="long", truncating="post",
                                       padding="post")
        # Create attention masks
        test_attention_masks = []

        # Create a mask of 1s for each token followed by 0s for padding
        for seq in test_input_ids:
            seq_mask = [float(i > 0) for i in seq]
            test_attention_masks.append(seq_mask)

        # Convert all of our data into torch tensors, the required datatype for our model

        test_inputs = torch.tensor(test_input_ids)
        test_labels = torch.tensor(test_labels)
        test_masks = torch.tensor(test_attention_masks)

        return TensorDataset(test_inputs, test_masks, test_labels)

    for in_dataset in ood_list:
        print('\nIn_dataset: %s'%in_dataset)
        train_sentences, val_sentences, test_sentences, train_labels, val_labels, test_labels = load_dataset(in_dataset)

        # Create an of our data with torch DataLoader.
        train_data = sentence_to_TensorDataset(train_sentences, train_labels)
        validation_data = sentence_to_TensorDataset(val_sentences, val_labels)
        prediction_data =  sentence_to_TensorDataset(test_sentences, test_labels)

        dataset_dir = 'dataset/{}'.format(in_dataset)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        torch.save(train_data, dataset_dir + '/train.pt')
        torch.save(validation_data, dataset_dir + '/val.pt')
        torch.save(prediction_data, dataset_dir + '/test.pt')
        print('Save train/val/test checkpoints to %s'%dataset_dir)

        # TODO
        torch.save(validation_data, 'dataset/test/{}_val_in_domain.pt'.format(in_dataset))
        torch.save(prediction_data, 'dataset/test/{}_test_in_domain.pt'.format(in_dataset))


        print('\n\nNow build testing OOD datasets ...')

        for ood_dataset in ood_list:
            print('\n ood_dataset: %s'%ood_dataset)
            _, _, nt_test_sentences, _, _, nt_test_labels = load_dataset(ood_dataset)
            nt_test_data =  sentence_to_TensorDataset(nt_test_sentences, nt_test_labels)

            ood_path = 'dataset/test/{}_test_out_of_domain.pt'.format(ood_dataset)
            torch.save(nt_test_data, ood_path)
            print('save OOD dataset %s to %s'%(ood_dataset, ood_path))


    print('\n\nNow build training outliers ...')

    print('\n Outliers: %s' % outliers)
    _, _, outliers_sentences, _, _, _ = load_dataset(outliers)
    outliers_labels = np.zeros(len(outliers_sentences))  # random assign a fake label to avoid bugs.
    outliers_data = sentence_to_TensorDataset(outliers_sentences, outliers_labels)

    outliers_path = 'dataset/{}.pt'.format(outliers)
    torch.save(outliers_data, outliers_path)
    print('save outliers %s to %s' % (outliers, outliers_path))


if __name__ == "__main__":
    main()

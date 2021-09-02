# BERT-ENN

This repository contains the essential code for the paper [Uncertainty-Aware Reliable Text Classification (KDD 2021)](https://dl.acm.org/doi/10.1145/3447548.3467382).

The paper is to combine evidential uncertainty and BERT for OOD detection tasks in text classification.  

You may also check our previous work [Multidimensional Uncertainty-Aware Evidential Neural Networks (AAAI 2021)](https://github.com/snowood1/wenn) in image classification.

## Prerequisites
The code is written by Python 3.6 in Linux system. The cuda version is 10.2. 
The necessary packages include:

	torch==1.7.1 
	torchvision==0.8.2 
	torchtext==0.8.1
	transformers==4.1.1 
	tensorflow-gpu==1.15.0 
	tqdm 
	matplotlib 
	numpy 
	sklearn 
	scipy 
	pandas 
	keras==2.3.0 

## Quick Start

1. Create folders 'datasets' and 'model_save' to save downloaded datasets and output results.
	
	We follow the same datasets in [Outlier Exposure](https://github.com/hendrycks/outlier-exposure/tree/master/NLP_classification).
	
  	You can also simply download the preprocessed **datasets** and the saved **results** [from here.](https://drive.google.com/drive/folders/1qAoUzQqo-Ys51LFgzvJpAH3fvLP4hhgM?usp=sharing)


2.  To reproduce results of Table 3 and 4 using the saved checkpoints, run the code below: (You can change sst to 20news or trec.  )

 **Baselines**
	
	python test_bert.py --model base --dataset sst --save_path saved_result --index 0			# maximum softmax scores
	python test_bert.py --model mc-dropout --dataset sst --save_path saved_result  --index 0			# MC-dropout
	python test_bert.py --model temperature --dataset sst --save_path saved_result  --index 0			# temperature scaling
	python test_bert.py --model manifold-smoothing --dataset sst --save_path saved_result  --index 0		# Manifold smoothing
	python test_bert.py --model oe --dataset sst --save_path saved_result  --index 0				# Outlier Explosure
	
	
 **ENN**
	
	python test_bert_enn.py --dataset sst --path ./saved_result/sst/ENN_ori/9.pt		# Vanilla ENN
	python test_bert_enn.py --dataset sst --path ./saved_result/sst/ENN_OE/9.pt		# ENN with Outlier Explosure
	python test_bert_enn.py --dataset sst --path ./saved_result/sst/ENN_AD/9.pt		# ENN with off-manifold adversial examples
	python test_bert_enn.py --dataset sst --path ./saved_result/sst/ENN_MIX/9.pt		# ENN with Mixture Regularizers
	


3.  To train ENN models from scratch:

**Baselines**
	
	python bert.py --model base --dataset sst --seed 0			# maximum softmax scores
	python bert.py --model mc-dropout --dataset sst --seed 0		# MC-dropout
	python bert.py --model temperature --dataset sst --seed 0		# temperature scaling
	python manifold-smoothing.py --dataset sst --seed 0 			# Manifold smoothing
	python bert_oe.py --dataset sst --seed 0				# Outlier Explosure
	
	
 **ENN**.  You can use the Hyper-parameters in Table 5.
 
 	python train_bert_enn.py --dataset 20news --train_batch_size 32 --beta_in 0 --beta_oe 1 --beta_off 0.1
	python train_bert_enn.py --dataset trec --train_batch_size 32 --beta_in 0 --beta_oe 1 --beta_off 0.1
	python train_bert_enn.py --dataset sst --train_batch_size 32 --beta_in 0.01 --beta_oe 1 --beta_off 0.1


4.  To evaluate your trained models, you can follow Step 2 but replace the input checkpoints paths. For example:

**Baselines**
	
	python test_bert.py --model base --dataset sst --save_path model_save --index 0			# maximum softmax scores
	python test_bert.py --model mc-dropout --dataset sst --save_path model_save --index 0		# MC-dropout
	python test_bert.py --model temperature --dataset sst --save_path model_save --index 0		# temperature scaling
	python test_bert.py --model manifold-smoothing --dataset sst  --save_path model_save --index 0 	# Manifold smoothing
	python test_bert.py --model oe -dataset sst --save_path model_save --index 0			# Outlier Explosure
	
	
 **ENN**
 
 	python test_bert_enn.py --dataset 20news --path ./model_save/20news/BERT-ENN-w2adv-0-on-0.0-oe-1.0-off-0.1/9.pt
	python test_bert_enn.py --dataset trec --path ./model_save/trec/BERT-ENN-w2adv-0-on-0.0-oe-1.0-off-0.1/9.pt
	python test_bert_enn.py --dataset sst --path ./model_save/sst/BERT-ENN-w2adv-0-on-0.01-oe-1.0-off-0.1/9.pt
	

5. We also provide an [example](https://github.com/snowood1/BERT-ENN/blob/main/demo%20result%20figures-final.ipynb) of plotting Figure 3 and Figure 4.
	

## Reference

The implementation of baselines are modified from:

[1] Outlier Exposure. https://github.com/hendrycks/outlier-exposure/tree/master/NLP_classification

[2] Manifold Calibration. https://github.com/Lingkai-Kong/Calibrated-BERT-Fine-Tuning




## Citation

If you find this repo useful in your research, please consider citing:

    @article{hu2021uncertainty,
        author = {Hu, Yibo and Khan, Latifur},
        title = {Uncertainty-Aware Reliable Text Classification},
        year = {2021},
        isbn = {9781450383325},
        publisher = {Association for Computing Machinery},
        address = {New York, NY, USA},
        url = {https://doi.org/10.1145/3447548.3467382},
        doi = {10.1145/3447548.3467382},
        booktitle = {Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery &amp; Data Mining},
        pages = {628–636},
        numpages = {9},
        keywords = {text classification, out-of-distribution detection, uncertainty qualification},
        location = {Virtual Event, Singapore},
        series = {KDD '21}
    }

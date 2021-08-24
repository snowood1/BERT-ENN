import numpy as np
import transformers
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd.gradcheck import zero_gradients

class BERT_ENN(torch.nn.Module):
    def __init__(self, num_labels):
        super(BERT_ENN, self).__init__()
        self.bert = transformers.BertModel.from_pretrained('bert-base-uncased',
                                                           num_labels=num_labels,
                                                           output_hidden_states=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, num_labels)
    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                labels=None,  # TODO
                inputs_embeds=None
                ):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            inputs_embeds=inputs_embeds,
                            )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        alpha = F.softplus(logits) + 1

        loss = None
        if labels is not None:
            s = alpha.sum(1, keepdim=True)
            p = alpha / s
            loss = torch.sum((labels - p) ** 2, dim=1).mean() + torch.sum(p * (1 - p) / (s + 1), axis=1).mean()
        return alpha, outputs[0], loss


class off_manifold_samples(object):
    def __init__(self, eps=0.001, rand_init='n', eps_min = 0.001, eps_max=0.1):
        super(off_manifold_samples, self).__init__()
        self.eps = eps
        self.rand_init = rand_init
        self.eps_min = eps_min
        self.eps_max = eps_max

    def generate(self, model, input_ids, input_mask, onehot):
        model.eval()
        with torch.no_grad():
            if torch.cuda.device_count() > 1:
                embedding = model.module.bert.get_input_embeddings()(input_ids)
            else:
                embedding = model.bert.get_input_embeddings()(input_ids)

        input_embedding = embedding.detach()
        # random init the adv samples
        if self.rand_init == 'y':
            input_embedding = input_embedding + torch.zeros_like(input_embedding).uniform_(-self.eps, self.eps)

        input_embedding.requires_grad = True

        zero_gradients(input_embedding)
        if input_embedding.grad is not None:
            input_embedding.grad.data.fill_(0)

        alpha = model(inputs_embeds=input_embedding, token_type_ids=None, attention_mask=input_mask)[0]
        s = alpha.sum(1, keepdim=True)
        p = alpha / s
        cost = torch.sum((onehot - p) ** 2, dim=1).mean() + \
                   torch.sum(p * (1 - p) / (s + 1), axis=1).mean()
        if torch.cuda.device_count() > 1:
            cost = cost.mean()
        model.zero_grad()
        cost.backward()

        off_samples = input_embedding + self.eps * torch.sign(input_embedding.grad.data)
        off_samples = torch.min(torch.max(off_samples, embedding - self.eps), embedding + self.eps)

        model.train()
        return off_samples.detach()


class ModelWithTemperature(nn.Module):

    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input_ids, token_type_ids, attention_mask):
        logits = self.model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def set_temperature(self, valid_loader, args):
        nll_criterion = nn.CrossEntropyLoss()

        logits_list = []
        labels_list = []
        with torch.no_grad():
            for step, batch in enumerate(valid_loader):
                batch = tuple(t.to(args.device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                logits = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[0]
                logits_list.append(logits)
                labels_list.append(b_labels)
            logits = torch.cat(logits_list)
            labels = torch.cat(labels_list)

        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval)
        before_temperature_nll = nll_criterion(logits, labels).item()
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()

        print('Optimal temperature: %.3f' % self.temperature.item())
        print('Before temperature - NLL: %.3f' % (before_temperature_nll))
        print('After temperature - NLL: %.3f' % (after_temperature_nll))

        return self


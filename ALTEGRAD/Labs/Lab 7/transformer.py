import time 
import numpy as np
import torch
from torch.nn import Module, ModuleList, Linear,\
        LayerNorm, Sequential, Embedding, CrossEntropyLoss,\
        Parameter
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.init import xavier_uniform_, constant_

import json
from tqdm import tqdm

from nltk.translate.bleu_score import corpus_bleu
from nltk import word_tokenize

import os

padding_token = '0'
oov_token = '1'
sos_token = '2' # start of sentence, only needed for the target sentence
eos_token = '3' # end of sentence, only needed for the target sentence


class Transformer(Module):
    """
    """
    ARGS = ["N_stacks_encoder", "N_stacks_decoder", "N_heads",
            "dk", "dv","dmodel", "ff_inner_dim", "vocab_source",
            "vocab_target_inv", "max_size", "device","EOS_token",
            "PAD_token", "SOS_token"]
    def __init__(self, N_stacks_encoder, N_stacks_decoder, N_heads, 
                 dk, dv, dmodel, ff_inner_dim, vocab_source,
                 vocab_target_inv, max_size, device, EOS_token=3, 
                 PAD_token=0, SOS_token=2, OOV_token=1):
        """
        Args:
            N_stacks_encoder (int): Number of encoder stacks.
            N_stacks_decoder (int): Number of encoder stacks.
            N_heads (int): The number of attention heads.
            dk (int): The dimension of the space in which attention is computed.
            dv (int): The dimension of the space in which values are sent.
            dmodel (int): The model dimension.
            ff_inner_dim (int): The inner dimensions of feed forward module.
            vocab_size_source (int)
            vocab_size_target (int)
            max_size (int): maximum size of sentence.
            device (str): "cpu" or "cuda"
            EOS_token (int): Position of the EOS token in vocabulary.
            PAD_token (int): Position of the PAD token in vocabulary.
            SOS_token (int): Position of the SOS token in vocabulary.
        """
        super(Transformer, self).__init__()
        self.max_size = max_size
        self.EOS_token = EOS_token
        self.SOS_token = SOS_token
        self.PAD_token = PAD_token
        self.OOV_token = OOV_token

        self.dk = dk
        self.dv = dv
        self.dmodel = dmodel
        self.ff_inner_dim = ff_inner_dim

        self.N_stacks_encoder = N_stacks_encoder
        self.N_stacks_decoder = N_stacks_decoder
        self.N_heads = N_heads

        self.vocab_source = vocab_source
        self.vocab_target_inv = vocab_target_inv
        self.vocab_size_target = len(vocab_target_inv) + 4
        self.source_embedding = Embedding(len(vocab_source)+4, dmodel, PAD_token).to(device)
        self.target_embedding = Embedding(self.vocab_size_target, dmodel, PAD_token).to(device)
        self.encoder = Encoder(N_stacks_encoder, N_heads, dk, dv, dmodel,
                               ff_inner_dim, device)
        self.decoder = Decoder(N_stacks_decoder, N_heads, dk, dv, dmodel,
                               ff_inner_dim, device)
        self.output_linear = Linear(dmodel, self.vocab_size_target).to(device)

        self.PE_tensor = self.build_encoding(max_size, dmodel).to(device)

        self.device = device

        # Initialisation of tensors
        for p in self.parameters():
            if len(p.data.size()) > 1:
                xavier_uniform_(p.data)

            else:
                constant_(p.data, 0.0)

    def forward(self, x, y=None):
        """ Forward pass of the transformer

        Args:
            x (torch.Tensor): Source sentence, tokenized, (B x T)
            y (torch.Tensor): If defined, teacher forcing is used to process
                rapidly the decoding. (B x T_target)
        """
        ### YOUR CODE HERE ###
        source_embedding = self.source_embedding(x)
        src_pos_enc = self.PE_tensor[:source_embedding.shape[1]]
        source_input = source_embedding.float()+ src_pos_enc.float()
        enc_out = self.encoder(source_input)
        
        if self.training and not y is None:
            ### YOUR CODE HERE ###
            target_embedding = self.target_embedding(self.shift(y))
   
        else:
            ### YOUR CODE HERE ###
            target_embedding = self.target_embedding(self.shift(x))
          
        pos_enc = self.PE_tensor[:target_embedding.shape[1]]
        target_input = target_embedding.float() + pos_enc.float()
        _, dec_out = self.decoder(enc_out,target_input)
        linear = self.output_linear(dec_out)
        out = F.softmax(linear)
        
        return out

    def fit(self, pairs_train, pairs_test, n_epochs, warmup_step, patiente=5,
            batch_size=64, seed=42, save_path="./trained_transformer.pt"):
        """ Trains the network

        Args:
            train_loader (torch.utils.data.DataLoader)
            test_loader (torch.utils.data.DataLoader)
            n_epchos (int)
            lr (float)
            batch_size (int)
            seed (int)
            save_path (str)
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        train_loader = DataLoader(Dataset(pairs_train), batch_size=batch_size, 
                                  shuffle=True, collate_fn=self.my_pad)
        test_loader = DataLoader(Dataset(pairs_test), batch_size=16, 
                              shuffle=False, collate_fn=self.my_pad)

        optimizer = torch.optim.Adam(self.parameters(), 1, betas=(0.9, 0.98), eps=1e-09)
        criteron = CrossEntropyLoss()

        step = 1
        curr_best_loss = 10.0
        patiente_count = 0
        for epoch in range(n_epochs):
            pbar = self.initialize_pbar(epoch, n_epochs, 136521)
            epoch_train_loss = []
            self.train()
            for i, (source, target) in enumerate(train_loader):
                self.warmup(optimizer, warmup_step, step)
                source = source.to(self.device)
                target = target.to(self.device)

                pred = self.forward(source, target)

                optimizer.zero_grad()
                mask = target.flatten() != self.PAD_token
                loss = criteron(pred.flatten(end_dim=1)[mask, :], target.flatten()[mask])
                loss.backward()
                optimizer.step()

                pbar.update(source.size(0))
                epoch_train_loss.append(loss.item())
                pbar.set_postfix({"train_loss" : np.mean(epoch_train_loss)})
                step += 1

            if not np.mean(epoch_train_loss) < curr_best_loss - 1e-3:
                patiente_count += 1

            else:
                patiente_count = 0
                curr_best_loss = np.mean(epoch_train_loss)

            if patiente_count >= patiente:
                break

        self.save(save_path)

    def test(self, test_loader):
        """ tests the network on the test_loader 

        Args:
            test_loader (torch.utils.data.DataLoader)
            criteron (torch)
        """
        self.eval()
        predictions = []
        targets = []
        pbar = tqdm(total=len(test_loader), unit_scale=True, desc="Testing", postfix={})
        for i, (source, target) in enumerate(test_loader):
            source = source.to(self.device)
            predictions.extend(self.forward(source).argmax(-1).cpu().tolist())
            targets.extend(target.tolist())
            pbar.update(1)

        for i, prediction in enumerate(predictions):
            new = []
            pos = 0
            while prediction[pos] != self.EOS_token:
                new.append(prediction[pos])
                pos += 1
                if pos == self.max_size:
                    break

            predictions[i] = new

        for i, target in enumerate(targets):
            new = []
            pos = 0
            while target[pos] != self.EOS_token:
                new.append(target[pos])
                pos += 1
                if pos == self.max_size:
                    break

            targets[i] = new

        targets = [[self.targetInts_to_nl(_)] for _ in targets]
        predictions = [self.targetInts_to_nl(_) for _ in predictions]

        test_BLEU = corpus_bleu(targets, predictions)
        pbar.set_postfix({"test_BLEU": test_BLEU})

        return test_BLEU

    def predict(self, sentence):
        """ Translates a sentence """
        self.eval()
        sentence = self.sourceNl_to_ints(sentence)
        pred = self.forward(sentence)
        print(pred.size())
        target_ints = pred.argmax(-1).squeeze() 
        target_ints = target_ints[:(target_ints == self.EOS_token).nonzero().item()]
        target_nl = self.targetInts_to_nl(target_ints.tolist())

        return ' '.join(target_nl)

    def save(self, path_to_file):
        """ Saves the architecture and weights of the model 

        Args:
            path_to_file (str)
        """
        attrs = {attr : getattr(self, attr) for attr in self.ARGS}
        attrs['state_dict'] = self.state_dict()
        torch.save(attrs, path_to_file)

    def my_pad(self, my_list):
        """ my_list is a list of tuples of the form [(tensor_s_1,tensor_t_1),...,(tensor_s_batch,tensor_t_batch)]
        the <eos> token is appended to each sequence before padding
        https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.pad_sequence """
        batch_source = pad_sequence([torch.cat((elt[0],torch.LongTensor([self.EOS_token])))
                      for elt in my_list],
                                    batch_first=True,
                                    padding_value=self.PAD_token)[:,:self.max_size-1]
        batch_target = pad_sequence([torch.cat((elt[1],torch.LongTensor([self.EOS_token])))
                      for elt in my_list],
                                    batch_first=True,
                                    padding_value=self.PAD_token)[:,:self.max_size-1]

        return batch_source,batch_target 

    def sourceNl_to_ints(self,source_nl):
        '''converts natural language source sentence into source integers'''
        source_nl_clean = source_nl.lower().replace("'",' ').replace('-',' ')
        source_nl_clean_tok = word_tokenize(source_nl_clean, "english")
        source_ints = [int(self.vocab_source[elt]) if elt in self.vocab_source else \
                       self.OOV_token for elt in source_nl_clean_tok] 
        source_ints = torch.LongTensor([source_ints]).to(self.device)

        return source_ints 

    def targetInts_to_nl(self, target_ints):
        '''converts integer target sentence into target natural language'''
        return ['<PAD>' if elt==self.PAD_token else '<OOV>' if elt==self.OOV_token \
                else '<EOS>' if elt==self.EOS_token else '<SOS>' if elt==self.SOS_token\
                else self.vocab_target_inv[elt] for elt in target_ints]

    def shift(self, x):
        """ Adds SOS token to sentence """
        x = torch.cat((torch.ones((x.size(0), 1), dtype=torch.long).to(self.device) * self.SOS_token, x), 1)

        return x

    def warmup(self, optim, warmup_step, step):
        """ Performs warmup as described in the original paper """
        for g in optim.param_groups:
            g['lr'] = self.dmodel**(-0.5)*min(step**(-0.5), step*warmup_step**(-1.5))

    @classmethod 
    def load(cls, path_to_file, device=None):
        """ Loads a saved model from scrath 

        Args:
            path_to_file (str)
            device (str): If defined, allows to modify to device on which the
                network runs.
        """
        attrs = torch.load(path_to_file, map_location=lambda storage, loc: storage) 
        state_dict = attrs.pop('state_dict')

        if not device is None:
            attrs["device"] = device

        new = cls(**attrs) 
        new.load_state_dict(state_dict)

        return new

    @staticmethod
    def build_encoding(size, dimension):
        """ Builds a tensor for positional encoding 

        Args:
            size (int): max size of a sentene.
            dimension (int): representation dimentionality in the network.
        """
        pos = torch.arange(size).view(-1,1)
        dim = []
        for _ in range(int(dimension/2)):
            dim.append(np.sin(pos/(10000**(2*_/dimension))))
            dim.append(np.cos(pos/(10000**((2*_+1)/dimension))))

        return torch.cat(dim, 1)

    @staticmethod
    def initialize_pbar(epoch, epochs, its_per_epochs):
        """Initializes a progress bar for the current epoch

        Returns:
            the progess bar (tqdm.tqdm)
        """
        return tqdm(total=its_per_epochs, unit_scale=True,
                    desc="Epoch %i/%i" % (epoch+1, epochs),
                    postfix={})


class Encoder(Sequential):
    def __init__(self, N_stacks, N_heads, dk, dv, dmodel, ff_inner_dim, device):
        """ Initializes the encoder.

        Args:
            N_stacks (int): The number of times the encoder is repeated.
            N_heads (int): The number of attention heads.
            dk (int): The dimension of the space in which attention is computed.
            dv (int): The dimension of the space in which values are sent.
            dmodel (int): The model dimension.
            ff_inner_dim (int): The inner dimensions of feed forward module.
        """
        stacks = []
        for _ in range(N_stacks):
            stacks.append(EncoderStack(N_heads, dk, dv, dmodel, ff_inner_dim,
                                       device))

        super(Encoder, self).__init__(*stacks)

class EncoderStack(Module):
    """ One stack of encoder as shown in the original paper """
    def __init__(self, N_heads, dk, dv, dmodel, ff_inner_dim, device):
        """ Initializes the encoder.

        Args:
            N_heads (int): The number of attention heads.
            dk (int): The dimension of the space in which attention is computed.
            dv (int): The dimension of the space in which values are sent.
            dmodel (int): The model dimension.
            ff_inner_dim (int): The inner dimensions of feed forward module.
        """
        super(EncoderStack, self).__init__()
        self.dmodel = dmodel
        self.attention = MultiHeadAttention(N_heads, dk, dv, dmodel, device)
        self.feed_forward = FeedForward(ff_inner_dim, dmodel, device)

    def forward(self, x):
        
        attention = self.attention(x,x,x)
        output_1 = x + attention
        output_1 = LayerNorm(output_1.size()[1:])(output_1)
        
        feed_forward = self.feed_forward(output_1)
        transformed_skip =  output_1 + feed_forward 
        
        return LayerNorm(transformed_skip.size()[1:])(transformed_skip)

class Decoder(Sequential):
    def __init__(self, N_stacks, N_heads, dk, dv, dmodel, ff_inner_dim, device):
        """ Initializes the decoder.

        Args:
            N_stacks (int): The number of times the decoder is repeated.
            N_heads (int): The number of attention heads.
            dk (int): The dimension of the space in which attention is computed.
            dv (int): The dimension of the space in which values are sent.
            dmodel (int): The model dimension.
            ff_inner_dim (int): The inner dimensions of feed forward module.
        """
        stacks = []
        for _ in range(N_stacks):
            stacks.append(DecoderStack(N_heads, dk, dv, dmodel, ff_inner_dim,
                                       device))

        super(Decoder, self).__init__(*stacks)

    def forward(self, x, y):
        """ Forward pass of the decode
        Needs a modification from the Sequential forward to accept multiple
        inputs.
        """
        for module in self.children():
            x, y = module(x, y)

        return x, y

class DecoderStack(Module):
    """ One stack of decoder as shown in the original paper """
    def __init__(self, N_heads, dk, dv, dmodel, ff_inner_dim, device):
        """ Initializes the deocder stack.

        Args:
            N_heads (int): The number of attention heads.
            dk (int): The dimension of the space in which attention is computed.
            dv (int): The dimension of the space in which values are sent.
            dmodel (int): The model dimension.
            ff_inner_dim (int): The inner dimensions of feed forward module.
        """
        super(DecoderStack, self).__init__()
        self.dmodel = dmodel
        self.attention_1 = MultiHeadAttention(N_heads, dk, dv, dmodel, device)
        self.attention_2 = MultiHeadAttention(N_heads, dk, dv, dmodel, device)
        self.feed_forward = FeedForward(ff_inner_dim, dmodel, device)

    def forward(self, x, y):
        """ Forward pass of one decoder stack

        Args:
            x (torch.tensor): The encoded sequence (B, T, dmodel)
            y (torch.tensor): The shifted decoded sequence (B, T, dmodel)
        """
        attention_1 = self.attention_1(y,y,y,maskout = True)
        output_1 = LayerNorm(attention_1.size()[1:])(attention_1 + y)
        
        attention_2 = self.attention_2(output_1,x,x)
        output_2 = LayerNorm(attention_2.size()[1:])(attention_2 + output_1)
        
        ff =  self.feed_forward(output_2)
        transformed_skip = LayerNorm(output_2.size()[1:])(ff + output_2)
        
        return x, transformed_skip

class MultiHeadAttention(Module):
    """ A MultiHeadAttention Module """
    def __init__(self, N_heads, dk, dv, dmodel, device):
        """ Initializes the Multihead

        Args:
            N_heads (int): The number of attention heads.
            dk (int): The dimension of the space in which attention is computed.
            dv (int): The dimension of the space in which values are sent.
            dmodel (int): The model dimension.
        """
        super(MultiHeadAttention, self).__init__()
        self.dk = dk

        self.Wq = torch.Tensor(N_heads, dmodel, dk).uniform_(-1, 1).to(device)
        self.Wk = torch.Tensor(N_heads, dmodel, dk).uniform_(-1, 1).to(device)
        self.Wv = torch.Tensor(N_heads, dmodel, dv).uniform_(-1, 1).to(device)
        self.Wo = torch.Tensor(dmodel, dmodel).uniform_(-1, 1).to(device)

    def forward(self, Q, K, V, maskout=False):
        """ forward of a multihead attention

        Specially implemented to be computed in parallel by pytorch.

        Args:
            Q (torch.Tensor): The queries (B, T, dmodel)
            K (torch.Tensor): The keys (B, T, dmodel)
            V (torch.Tensor): The values (B, T, dmodel)
            maskout (bool): Whether to apply or not forward masking.

        Returns:
            (B, T, dmodel) tensor
        """
        ### TO COMPLETE ###
    
        Qp = torch.matmul(torch.unsqueeze(Q,1),self.Wq)
        Kp = torch.matmul(torch.unsqueeze(K,1),self.Wk)
        Vp = torch.matmul(torch.unsqueeze(V,1),self.Wv)
      
        heads = torch.matmul(Qp, Kp.transpose(3,2)) / np.sqrt(self.dk)
        
        if maskout:
            mask = torch.tril(torch.ones(Q.size()[1], Q.size()[1]))
            mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0)
            heads += mask
            
        heads = torch.matmul(torch.softmax(heads,2), Vp).permute(0,2,1,3)
        heads = heads.contiguous().view((-1, heads.shape[1], heads.shape[2] * heads.shape[3]))
        
        return torch.matmul(heads,self.Wo)
        
    
class FeedForward(Module):
    """ A FeedForward module """
    def __init__(self, inner_dim, dmodel, device):
        """ Initialized the FeedForwrd module

        Args:
            inner_dim (int): Dimension of the hidden layer.
            dmodel (int): The model dimension.
        """
        super(FeedForward, self).__init__()
        self.inner = Linear(dmodel, inner_dim).to(device)
        self.out = Linear(inner_dim, dmodel).to(device)

    def forward(self, x):
        """ Forward of a FeedForward module

        Args:
            x (torch.Tensor)
        """
        ### TO COMPLETE ###
        inner = self.inner(x)
        inner = F.relu(inner)
        
        return self.out(inner)
        

class Dataset(torch.utils.data.Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs) # total nb of observations

    def __getitem__(self, idx):
        source, target = self.pairs[idx] # one observation
        return torch.LongTensor(source), torch.LongTensor(target)

def load_pairs(train_or_test):
    with open("./data/" + 'pairs_' + train_or_test + '_ints.txt', 'r', encoding='utf-8') as file:
        pairs_tmp = file.read().splitlines()

    pairs_tmp = [elt.split('\t') for elt in pairs_tmp]
    pairs_tmp = [[[int(eltt) for eltt in elt[0].split()],[int(eltt) for eltt in \
                  elt[1].split()]] for elt in pairs_tmp]

    return pairs_tmp

if __name__ == "__main__":

    pairs_train = load_pairs('train')
    pairs_test = load_pairs('test')

    with open("./data/" + 'vocab_source.json','r') as file:
        vocab_source = json.load(file) # word -> index

    with open("./data/" + 'vocab_target.json','r') as file:
        vocab_target = json.load(file) # word -> index

    vocab_target_inv = {v:k for k,v in vocab_target.items()} # index -> word

    if os.path.exists("trained_transformer.pt"):
        model = Transformer.load("trained_transformer.pt")

    else:
        model = Transformer(N_stacks_encoder=3,
                            N_stacks_decoder=3,
                            N_heads=8, dk=int(128/8), dv=int(128/8), dmodel=128,
                            ff_inner_dim=512,
                            vocab_source=vocab_source,
                            vocab_target_inv=vocab_target_inv,
                            max_size=24, device="cpu")
        model.fit(pairs_train, pairs_test, 50, warmup_step=4000)

    to_test = ['I am a student.',
               'I have a red car.',  # inversion captured
               'I love playing video games.',
               'This river is full of fish.', # plein vs pleine (accord)
               'The fridge is full of food.', 
               'The cat fell asleep on the mat.',
               'my brother likes pizza.', # pizza is translated to 'la pizza'
               'I did not mean to hurt you', # translation of mean in context
               'She is so mean',
               'Help me pick out a tie to go with this suit!', # right translation
               "I can't help but smoking weed", # this one and below: hallucination
               'The kids were playing hide and seek',
               'The cat fell asleep in front of the fireplace'] 

    test_loader = DataLoader(Dataset(pairs_test), batch_size=16, 
                          shuffle=False, collate_fn=model.my_pad)

    model.test(test_loader)

    for elt in to_test:
        print('= = = = = \n','%s -> %s' % (elt, model.predict(elt)))

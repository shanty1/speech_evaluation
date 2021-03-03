import torch
import numpy as np
import argparse
import pickle
import os
from torchvision import transforms
from build_vocab import Vocabulary
import matplotlib.pyplot as plt
from model import EncoderRNN, DecoderRNN
from audio_pre import audio_pad_pack, audio_mfcc
import numpy as np
from torch.autograd import Variable
import random
from data_loader import caculate_max_len
from data_loader import data_get
from data_loader import get_audio_fea
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import  time

Z_DIM = 0#16
x = []
y = []

def main(args):
    # random set
    manualSeed = 1
    # print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)

    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)


    audio_len, comment_len, mfcc_dim = caculate_max_len(args.audio_dir, args.text_path, vocab)

    # Build models
    encoder = EncoderRNN(mfcc_dim, args.embed_size, args.hidden_size).to(device)
    decoder = DecoderRNN(args.embed_size + Z_DIM, args.hidden_size, len(vocab), vocab('<pad>'), args.num_layers).to(
        device)
    # decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))


    for j in range(80,101,1):
        args.audio='./data/audio/swz{}.wav'.format(j)
        # Prepare an image
        audio = get_audio_fea(args.audio)
        audio = torch.tensor(audio).unsqueeze(0).to(device)

        # Generate an caption from the image
        feature = encoder(audio,[audio.shape[1]])
        if (Z_DIM > 0):
            z = Variable(torch.randn(feature.shape[0], Z_DIM)).to(device)
            feature = torch.cat([z,feature],1)
        comment = torch.tensor([vocab("<start>")]).to(device)
        sampled_ids = decoder.sample(feature,comment)
        sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)

        # Convert word_ids to words
        sampled_caption = []
        for word_id in sampled_ids:
            word = vocab.idx2word[word_id]
            sampled_caption.append(word)
            if word == '<end>':
                break
        sentence = ' '.join(sampled_caption)

        # Print out the image and the generated caption
        sentence = sentence.replace('<start>','').replace('<unk>','，').replace('<end>','').replace(' ','')
        print(str(j)+".",sentence)


if __name__ == '__main__':
    # for i in range(1,101):

    i = 1

    parser = argparse.ArgumentParser()
    # print(i,end='. ')
    parser.add_argument('--audio', type=str, default='./data/audio/swz{}.wav'.format(i), help='test audio path') #110-30  300-30-15   300-90(90-1全拟合)
    parser.add_argument('--encoder_path', type=str, default='./models/encoder-1100-270.ckpt', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='./models/decoder-1100-270.ckpt', help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')

    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224, help='size for randomly cropping images')
    parser.add_argument('--audio_dir', type=str, default='./data/audio/', help='directory for audioes')
    parser.add_argument('--text_path', type=str, default='./data/comment/comment.txt', help='path for comment text')

    # Model parameters
    parser.add_argument('--embed_size', type=int, default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in lstm')

    args = parser.parse_args()
    main(args)
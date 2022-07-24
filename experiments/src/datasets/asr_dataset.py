import os
import json
import torch
from torch.utils.data import DataLoader
import pickle
import string
import random
import librosa
import torchaudio
import numpy as np
from tqdm import tqdm
from glob import glob
import nlpaug.flow as naf
from collections import Counter
import nlpaug.augmenter.audio as naa
import nlpaug.augmenter.spectrogram as nas
from torchvision.transforms import Normalize
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel, BertForMaskedLM

HARPER_VALLEY_MEAN = [-29.436176]
HARPER_VALLEY_STDEV = [14.90793]
HARPER_VALLEY_HOP_LENGTH_DICT = {224: 672, 112: 1344, 64: 2360, 32: 4800}

VOCAB = [' ',"'",'~','-','.','<','>','[',']','U','N','K','a','b','c','d','e','f','g',
         'h','i','j','k','l','m','n','o','p','q','r','s','t','u','v',
         'w','x','y','z']
SILENT_VOCAB = ['[baby]', '[ringing]', '[laughter]', '[kids]', '[music]', 
                '[noise]', '[unintelligible]', '[dogs]', '[cough]']

with open('src/datasets/vocab/subwords.txt') as f:
    subwords_list = f.read().split("\n")

class BaseHarperValley(Dataset):
    """Base class for loading HarperValley datasets that will handle the data
    loading and preprocessing.

    @param root: string 
                 path to the directory where harpervalley data is stored.
    @param min_utterance_length: integer (default: 4)
                                 minimum number of tokens to compose an utterance. 
                                 Utterances with fewer tokens will be ignored.
    @param prune_speakers: boolean (default: True)
                           prune speakers who make less than min_utterance_length speakers      
    """
    def __init__(self,
                 root,
                 min_utterance_length=4,
                 min_speaker_utterances=10,
                 prune_speakers=True,
                 prune_silent=False,
                 max_wav_length=700):
        
        super().__init__()       
        
        self.root = root
        self.min_utterance_length = min_utterance_length
        self.min_speaker_utterances = min_speaker_utterances
        self.prune_speakers = prune_speakers

    def prune_data_by_speaker(self, data, min_freq=10):
        pruned_data = []
        valid_speaker_ids, _ = self.process_speaker_ids(data, min_freq=min_freq)
        for row in data:
            if row['speaker_id'] in valid_speaker_ids:
                pruned_data.append(row)
        return pruned_data

    def process_speaker_ids(self, data, min_freq=10):
        # speaker ids arent balanced... collapse everything 
        # that speaks less than 10 times
        speaker_ids = [row['speaker_id'] for dt in data for row in dt]
        speaker_freq = dict(Counter(speaker_ids))
        valid_speaker_ids = []
        invalid_speaker_ids = []

        for speaker_id, freq in speaker_freq.items():
            if freq <= min_freq:
                invalid_speaker_ids.append(speaker_id)
                print("freq:{}, ids:{}".format(freq, speaker_id))
            else:
                valid_speaker_ids.append(speaker_id)

        valid_speaker_ids = sorted(list(set(valid_speaker_ids)))
        invalid_speaker_ids = sorted(list(set(invalid_speaker_ids)))

        return valid_speaker_ids, invalid_speaker_ids

    def get_vocabs(self, data, min_freq=10):
        dialogacts = []
        systemacts = []
        for dt in data:
            for row in dt:
                da_list = row['dialog_acts']
                for da in da_list:
                    if 'caller' in da:
                        systemacts.append(da)
                    
                    dialogacts.append(da)
                
        systemacts.append('caller_waiting')
            
        dialogact_vocab = sorted(set(dialogacts))
        systemact_vocab = sorted(set(systemacts))
        
        valid_speaker_ids, invalid_speaker_ids = self.process_speaker_ids(data, min_freq)
        #assert len(invalid_speaker_ids) == 0
        return dialogact_vocab, systemact_vocab, valid_speaker_ids
    
    def speaker_roles_to_labels(self, speaker_roles):
        role_labels = []
        for roles in speaker_roles:
            role_label = []
            for role in roles:
                if role == "agent":
                    role_label.append(1)
                else:
                    role_label.append(2)
            role_labels.append(role_label)
        return role_labels

    def transcripts_to_labels(self, transcripts_list, asr_target_type='phoneme', max_length=70):
        """Converts transcript texts to sequences of vocab indices for characters."""
        
        if asr_target_type=='phoneme':
            transcript_label = []
            for transcript in transcripts_list:
                words = transcript.split()
                labels = []
                for i in range(len(words)):
                    word = words[i]
                    if word in SILENT_VOCAB:
                        # silent vocab builds on top of vocab
                        label = SILENT_VOCAB.index(word) + len(VOCAB)
                        labels.append(label)
                    else:
                        chars = list(word)
                        labels.extend([VOCAB.index(ch) for ch in chars])
                    # add a space in between words
                    labels.append(VOCAB.index(' '))
                labels = labels[:-1]  # remove last space
                transcript_label.append(labels)
                
        elif asr_target_type=='subword':
            transcript_label = []
            for transcript in transcripts_list:
                result = self.tokenizer(transcript, max_length=max_length, padding="max_length", truncation=True)
                tmp = result['input_ids']
                tmp = np.array(tmp)
                tmp = tmp[tmp>0]
                label = [subwords_list.index(str(i)) for i in tmp]
                transcript_label.append(label)
        else:
            raise NotImplemented

        return transcript_label
    
#     def transcripts_to_bert_labels(self, transcripts_list, max_length=70):
#         bert_labels, attention_mask = [], []
#         for transcripts in transcripts_list:
#             result = self.tokenizer(transcripts, max_length=max_length, padding="max_length", truncation=True, return_tensors='pt')
#             labels = result['input_ids']
#             masks = result['attention_mask']

#             bert_labels.append(labels)
#             attention_mask.append(masks)

#         return bert_labels, attention_mask

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class MyHarperValleyTimingDataset(BaseHarperValley):
    """Dataset to be used to train CTC, LAS, and MTL. 

    @param wav_maxlen: integer (default: None)
                       Maximum number of tokens in the wav input.
    @param transcript_maxlen: integer (default: None)
                              Maximum number of tokens in the labels output.
    @param add_sos_and_eos_tok: boolean (default: False)
                                Whether to prepend SOS token and append with EOS token.
                                Required for LAS and MTL.
    @param add_eps_token: boolean (default: False)
                          Whether to add blank / epsilon tokens.
                          Required for CTC and MTL.
    @param split_by_speaker: boolean (default: False)
                             Whether to train/test split randomly or by speaker,
                             where entries in the training and test sets have
                             disjoint speakers.
    """
    def __init__(
            self,
            root, 
            split='train', 
            n_mels=128,
            n_fft=256, 
            win_length=256, 
            hop_length=128,
            wav_maxlen=500,
            transcript_maxlen=200,
            append_eos_token=False,
            split_by_speaker=True,
            min_utterance_length=4,
            min_speaker_utterances=10,
            prune_speakers=True,
            prune_silent=False,
            bert_max_length=70,
            asr_target_type='phoneme',
        ):
        super().__init__(root, min_utterance_length, min_speaker_utterances, prune_speakers, prune_silent, wav_maxlen)
        
        self.offset = 100  # 100ms長めに取る
        self.frame_ms = 50
        
        self.asr_target_type=asr_target_type
        if asr_target_type=="subword":
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        # dumpから読み取り
        names_dir = os.path.join(root, split, 'wav')
        names = os.listdir(names_dir)

        wavpaths, uttr_type, human_transcripts, dialog_acts, next_acts = [], [], [], [], []
        speaker_ids, crop_start_ms_list, crop_duration_ms_list, uttr_duration_ms_list, timing_ms_list = [], [], [], [], []
        for i, name in enumerate(tqdm(names)):
            idx_dir = os.path.join(names_dir, name)
            idxs = os.listdir(idx_dir)

            json_path = os.path.join(idx_dir, '{}.json'.format(name)).replace('wav', 'meta')
            with open(json_path) as f:
                data = json.load(f)

            wavpaths+=data['wavpath']
            uttr_type+=data['type']
            human_transcripts+=data['human_transcript']
            self.human_transcripts = human_transcripts
            crop_start_ms_list+=data['crop_start_ms']
            crop_duration_ms_list+=data['crop_duration_ms']
            uttr_duration_ms_list+=data['uttr_duration_ms']
            timing_ms_list+=data['timing_ms']
        

        human_transcript_labels = self.transcripts_to_labels(human_transcripts, asr_target_type)
            
        if asr_target_type=='phoneme':
            self.num_class = len(VOCAB) + len(SILENT_VOCAB)
        elif asr_target_type=='subword':
            self.num_class = len(subwords_list)
        else:
            raise NotImplemented
        self.pad_index = 0

        self.root = root
        self.append_eos_token = append_eos_token
        self.wavpaths = wavpaths
        self.uttr_type = uttr_type
        self.human_transcript_labels = human_transcript_labels
        self.split_by_speaker = split_by_speaker
        self.crop_start_ms_list = crop_start_ms_list
        self.crop_duration_ms_list = crop_duration_ms_list
        self.uttr_duration_ms_list = uttr_duration_ms_list
        self.timing_ms_list = timing_ms_list
        self.wav_maxlen = wav_maxlen
        self.transcript_maxlen = transcript_maxlen
        self.input_dim = n_mels
        self.num_labels = self.num_class
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

    def indices_to_chars(self, indices):
        # add special characters in front (since we did this above)
        full_vocab = ['<eps>', '<sos>', '<eos>', '<pad>'] + VOCAB + SILENT_VOCAB
        chars = [full_vocab[ind] for ind in indices]
        return chars

    @staticmethod
    def pad_wav(wav, maxlen, pad=0):
        dim = wav.shape[1]
        padded = np.zeros((maxlen, dim)) + pad
        if len(wav) > maxlen:
            wav = wav[-maxlen:]
        length = len(wav)
        padded[:length, :] = wav
        return padded, length
    
    @staticmethod
    def pad_timing(wav, maxlen, pad=0):
        
        padded = np.zeros(maxlen) + pad
        if len(wav) > maxlen:
            wav = wav[-maxlen:]
        length = len(wav)
        padded[:length] = wav
        return padded, length
    
    @staticmethod
    def pad_transcript_labels(transcript_labels, maxlen, pad=-1):
        padded = np.zeros(maxlen) + pad
        if len(transcript_labels) > maxlen:
            transcript_labels = transcript_labels[:maxlen]
        length = len(transcript_labels)
        padded[:length] = transcript_labels
        return padded, length

    def __getitem__(self, index):
        
        offset_list = []
        duration_list = []
        
        wavpath = self.wavpaths[index]
        cnnae_path = wavpath.replace('.wav', '_spec.npy').replace('wav', 'AE')
        fbank_path = wavpath.replace('.wav', '_fbank.npy').replace('wav', 'fbank')

        uttr_type = self.uttr_type[index]

        crop_start_ms = self.crop_start_ms_list[index]
        crop_duration_ms = self.crop_duration_ms_list[index]
        uttr_duration_ms = self.uttr_duration_ms_list[index]

        wav, sr = torchaudio.load(wavpath)
        wav = wav.numpy()[0]
        wav = torch.from_numpy(wav).float()
        cnnae = np.load(cnnae_path)
        fbank = np.load(fbank_path)

        duration = (uttr_duration_ms + self.offset) // self.frame_ms # 発話の長さ


        human_transcript_label = self.human_transcript_labels[index]
        # pad transcript to a fixed length
        human_transcript_label, human_transcript_length = self.pad_transcript_labels(
            human_transcript_label, self.transcript_maxlen, pad=self.pad_index)

        cnnae = torch.from_numpy(cnnae).float()
        fbank = torch.from_numpy(fbank).float()

        length = duration#len(fbank)//5-(self.offset//50)
        
        #feat = torch.cat([fbank[:duration], cnnae[:duration]], dim=-1)


        example = {
            'indices': index,
            'uttr_type': uttr_type,
            'wav': wav,
            'cnnae': cnnae,
            'fbank': fbank,
            'input_lengths': duration, 
            'labels': human_transcript_label, 
            'label_lengths': human_transcript_length,
        }
                
        return list(example.values())

    def __len__(self):
        return len(self.wavpaths)
    
    
def collate_fn(batch):
    
    indices, uttr_type, wavs, cnnae, fbank, input_lengths, labels, label_lengths = zip(*batch)
    
    batch_size = len(indices)
    
    uttr_nums = torch.tensor(len(labels))
    
    cnnae_len = max([len(inp) for inp in cnnae])
    fbank_len = max([len(inp) for inp in fbank])
    #wav_len = max([inp for inp in input_lengths])
    label_len = max(label_lengths)
    
    cnnae_dim = cnnae[0].shape[-1]
    fbank_dim = fbank[0].shape[-1]
    
    uttr_type_ = torch.zeros(batch_size).long()
    cnnae_ = torch.zeros(batch_size, cnnae_len, cnnae_dim)
    fbank_ = torch.zeros(batch_size, fbank_len, fbank_dim)
    input_lengths_ = torch.zeros(batch_size).long()
    labels_ = torch.zeros(batch_size, label_len).long()
    label_lengths_ = torch.zeros(batch_size).long()

    for i in range(batch_size):
        l1 = len(cnnae[i])
        l2 = len(fbank[i])
        cnnae_[i, :l1, :] = cnnae[i]
        fbank_[i, :l2, :] = fbank[i]
            
        uttr_type_[i] = torch.tensor(uttr_type[i]).long()       
        input_lengths_[i] = torch.tensor(input_lengths[i]).long()
        labels_[i, :] = torch.tensor(labels[i][:label_len]).long()
        label_lengths_[i] = torch.tensor(label_lengths[i]).long()

    return uttr_nums, uttr_type_, wavs, cnnae_, fbank_, input_lengths_, labels_, label_lengths_


def create_dataloader(dataset, batch_size, shuffle=True, pin_memory=True, num_workers=2):
    loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=shuffle, 
        pin_memory=pin_memory,
        num_workers=num_workers,
        collate_fn= lambda x: collate_fn(x),
    )
    return loader

def get_dataset(config, split="train", asr_target_type="phoneme"):
    wav_maxlen = config.data_params.wav_maxlen
    transcript_maxlen = config.data_params.transcript_maxlen
    root = config.data_params.harpervalley_root

    if split=='train':
        dataset = MyHarperValleyTimingDataset(
            root,
            split='train',
            append_eos_token=config.data_params.append_eos_token,
            wav_maxlen=wav_maxlen,
            transcript_maxlen=transcript_maxlen,
            n_mels=config.data_params.n_mels,
            n_fft=config.data_params.n_fft,
            hop_length=config.data_params.hop_length,
            win_length=config.data_params.win_length,
            split_by_speaker=True,
            min_utterance_length=config.data_params.min_utterance_length,
            min_speaker_utterances=config.data_params.min_speaker_utterances,
            prune_speakers=True,
            prune_silent=True,
            bert_max_length = config.data_params.bert_max_length,
            asr_target_type=asr_target_type,
        )
    elif split=='val':
        dataset = MyHarperValleyTimingDataset(
            root,
            split='val',
            append_eos_token=config.data_params.append_eos_token,
            wav_maxlen=wav_maxlen,
            transcript_maxlen=transcript_maxlen,
            n_mels=config.data_params.n_mels,
            n_fft=config.data_params.n_fft,
            hop_length=config.data_params.hop_length,
            win_length=config.data_params.win_length,
            split_by_speaker=True,
            min_utterance_length=config.data_params.min_utterance_length,
            min_speaker_utterances=config.data_params.min_speaker_utterances,
            prune_speakers=True,
            prune_silent=True,
            bert_max_length = config.data_params.bert_max_length,
            asr_target_type=asr_target_type,
        )
    else:
        dataset = MyHarperValleyTimingDataset(
            root,
            split='test',
            append_eos_token=config.data_params.append_eos_token,
            wav_maxlen=wav_maxlen,
            transcript_maxlen=transcript_maxlen,
            n_mels=config.data_params.n_mels,
            n_fft=config.data_params.n_fft,
            hop_length=config.data_params.hop_length,
            win_length=config.data_params.win_length,
            split_by_speaker=True,
            min_utterance_length=config.data_params.min_utterance_length,
            min_speaker_utterances=config.data_params.min_speaker_utterances,
            prune_speakers=True,
            prune_silent=True,
            bert_max_length = config.data_params.bert_max_length,
            asr_target_type=asr_target_type,
        )

    return dataset


def get_dataloader(dataset, config, split="train"):
    if split=="train":
        shuffle = True
    else:
        shuffle = False
    dataloader = create_dataloader(dataset, config.optim_params.batch_size, shuffle=shuffle)
    return dataloader

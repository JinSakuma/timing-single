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

HARPER_VALLEY_MEAN = [-29.436176]
HARPER_VALLEY_STDEV = [14.90793]
HARPER_VALLEY_HOP_LENGTH_DICT = {224: 672, 112: 1344, 64: 2360, 32: 4800}

VOCAB = [' ',"'",'~','-','.','<','>','[',']','U','N','K','a','b','c','d','e','f','g',
         'h','i','j','k','l','m','n','o','p','q','r','s','t','u','v',
         'w','x','y','z']
SILENT_VOCAB = ['[baby]', '[ringing]', '[laughter]', '[kids]', '[music]', 
                '[noise]', '[unintelligible]', '[dogs]', '[cough]']


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

    def transcripts_to_labels(self, transcripts_list):
        """Converts transcript texts to sequences of vocab indices for characters."""
        transcript_labels = []
        for transcripts in transcripts_list:
            transcript_label = []
            for transcript in transcripts:
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

            transcript_labels.append(transcript_label)
        return transcript_labels
    
    def transcripts_to_bert_labels(self, transcripts_list, max_length=70):
        bert_labels, attention_mask = [], []
        for transcripts in transcripts_list:
            result = self.tokenizer(transcripts, max_length=max_length, padding="max_length", truncation=True, return_tensors='pt')
            labels = result['input_ids']
            masks = result['attention_mask']

            bert_labels.append(labels)
            attention_mask.append(masks)

        return bert_labels, attention_mask

    def task_type_to_labels(self, tasks, vocab):
        return [[vocab.index(task) for task in tasks_] for tasks_ in tasks]

    def dialog_acts_to_labels(self, dialogacts_list, vocab):
        actions_list = []
        for dialogacts in dialogacts_list:
            actions = []
            for acts in dialogacts:
                onehot = [0 for _ in range(len(vocab))]
                for act in acts:
                    onehot[vocab.index(act)] = 1
                actions.append(onehot)
                
            actions_list.append(actions)
        return actions_list

    def speaker_id_to_labels(self, speaker_ids_list, valid_ids):
        labels_list = []
        for speaker_ids in speaker_ids_list:
            labels = []
            for speaker_id in speaker_ids:
                if speaker_id in valid_ids:
                    labels.append(valid_ids.index(speaker_id))
                else:
                    raise Exception(f'speaker_id {speaker_id} unexpected.')
                    
            labels_list.append(labels)
        return labels_list

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
        ):
        super().__init__(root, min_utterance_length, min_speaker_utterances, prune_speakers, prune_silent, wav_maxlen)
        
        self.offset = 300
        dialog_vocab_path = os.path.join(root, 'vocab/dialog_act_vocab.txt')
        system_vocab_path = os.path.join(root, 'vocab/system_act_vocab.txt')
        
        with open(dialog_vocab_path) as f:
            dialog_acts_vocab = f.read().split("\n")
            if dialog_acts_vocab[-1] == '':
                dialog_acts_vocab = dialog_acts_vocab[:-1]
                
        with open(system_vocab_path) as f:
            system_acts_vocab = f.read().split("\n")
            if system_acts_vocab[-1] == '':
                system_acts_vocab = system_acts_vocab[:-1]

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

            wavpaths.append(data['wavpath'])
            uttr_type.append(data['type'])
            human_transcripts.append(data['human_transcript'])
            self.human_transcripts = human_transcripts
            dialog_acts.append(data['dialog_acts'])
            next_acts.append(data['next_acts'])
            crop_start_ms_list.append(data['crop_start_ms'])
            crop_duration_ms_list.append(data['crop_duration_ms'])
            uttr_duration_ms_list.append(data['uttr_duration_ms'])
            timing_ms_list.append(data['timing_ms'])
        

        human_transcript_labels = self.transcripts_to_labels(human_transcripts)
        dialog_acts_labels = self.dialog_acts_to_labels(dialog_acts, dialog_acts_vocab)
        next_acts_labels = self.dialog_acts_to_labels(next_acts, system_acts_vocab)
        
        self.dialog_acts_vocab = dialog_acts_vocab
        self.system_acts_vocab = system_acts_vocab
            
        self.num_class = len(VOCAB) + len(SILENT_VOCAB)
        self.pad_index = 0

        self.root = root
        self.append_eos_token = append_eos_token
        self.wavpaths = wavpaths
        self.uttr_type = uttr_type
        self.human_transcript_labels = human_transcript_labels
        self.dialog_acts_labels = dialog_acts_labels
        self.next_acts_labels = next_acts_labels
        self.dialog_acts_num_class = len(dialog_acts_vocab)
        self.next_acts_num_class = len(system_acts_vocab)
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
        dialog_len = len(self.wavpaths[index])
        offset_list = []
        duration_list = []
        for dl in range(dialog_len):
            wavpath = self.wavpaths[index][dl]
            cnnae_path = wavpath.replace('.wav', '_spec.npy').replace('wav', 'AE')
            fbank_path = wavpath.replace('.wav', '_fbank.npy').replace('wav', 'fbank')
            
            uttr_type = self.uttr_type[index][dl]
            wav, sr = torchaudio.load(wavpath)
            wav = wav.numpy()[0]
            
            crop_start_ms = self.crop_start_ms_list[index][dl]
            crop_duration_ms = self.crop_duration_ms_list[index][dl]
            uttr_duration_ms = self.uttr_duration_ms_list[index][dl]
            timing = self.timing_ms_list[index][dl]
            
            cnnae = np.load(cnnae_path)
            fbank = np.load(fbank_path)
    
            offset_list.append((timing - uttr_duration_ms)//50) # 発話末から発話タイミングまでの時間
            # offset_list.append(crop_duration_ms) # 発話末から発話タイミングまでの時間
            duration_list.append(uttr_duration_ms//50) # 発話の長さ
            
            wav_length = (crop_duration_ms // 50)+1
            if wav_length > self.wav_maxlen:
                # print(wav_length)
                wav_length = self.wav_maxlen
            uttr_label = np.zeros(wav_length)
            end = uttr_duration_ms // 50
            uttr_label[:end] = 1
            
            timing_label = np.zeros(wav_length)
            if uttr_type == 1 or uttr_type == 2:
                if crop_duration_ms < timing: # タイミングが3秒より遅い時は3秒に揃える
                    timing_label[-1] = 1
                else:
                    idx = (crop_duration_ms-timing) // 50
                    timing_label[-idx:]=1
            else:
                timing_label = np.zeros(wav_length)
                
            # pad to a fixed length
            uttr_label, _ = self.pad_timing(uttr_label, self.wav_maxlen)
            timing_label, _ = self.pad_timing(timing_label, self.wav_maxlen)
            
            human_transcript_label = self.human_transcript_labels[index][dl]
            # pad transcript to a fixed length
            human_transcript_label, human_transcript_length = self.pad_transcript_labels(
                human_transcript_label, self.transcript_maxlen, pad=self.pad_index)

            dialog_acts_label = self.dialog_acts_labels[index][dl]
            next_acts_label = self.next_acts_labels[index][dl]

            wav = torch.from_numpy(wav).float()
            cnnae = torch.from_numpy(cnnae).float()
            fbank = torch.from_numpy(fbank).float()
            
            length = len(fbank)-(self.offset//50)
            feat = torch.cat([fbank[:length], cnnae[:length]], dim=-1)
            
            timing_label = torch.from_numpy(timing_label).long()
            uttr_label = torch.from_numpy(uttr_label).long()
            dialog_acts_label = torch.LongTensor(dialog_acts_label)
            next_acts_label = torch.LongTensor(next_acts_label)
            
            if dl == 0:
                example = {
                    'indices': index,
                    'uttr_type': [uttr_type], 
                    'wav': [wav],
                    'inputs': [feat],
                    'input_lengths': [length], 
                    'timing': timing_label.unsqueeze(0), 
                    'uttr_labels': uttr_label.unsqueeze(0), 
                    'labels': [human_transcript_label], 
                    'label_lengths': [human_transcript_length],
                    'dialog_acts': dialog_acts_label.unsqueeze(0), 
                    'next_acts': next_acts_label.unsqueeze(0), 
                }
            else:
                example['uttr_type'].append(uttr_type)
                example['wav'].append(wav)
                example['inputs'].append(feat)
                example['input_lengths'].append(length)
                example['timing'] = torch.cat([example['timing'], timing_label.unsqueeze(0)])
                example['uttr_labels'] = torch.cat([example['uttr_labels'], uttr_label.unsqueeze(0)])
                example['labels'].append(human_transcript_label)
                example['label_lengths'].append(human_transcript_length)
                example['dialog_acts'] = torch.cat([example['dialog_acts'], dialog_acts_label.unsqueeze(0)])
                example['next_acts'] = torch.cat([example['next_acts'], next_acts_label.unsqueeze(0)])
#                 example['speaker_ids'].append(speaker_id_label)
                
        example['offset'] = offset_list
        example['duration'] = duration_list
        
        return list(example.values())

    def __len__(self):
        return len(self.wavpaths)
    
    
def collate_fn(batch):
    
    indices, uttr_type, wavs, inputs, input_lengths, timings, uttr_labels, labels, label_lengths, dialog_acts, next_acts, offset, duration = zip(*batch)
    
    batch_size = len(indices)
    
    dialog_act_num = dialog_acts[0].shape[1]
    next_act_num = next_acts[0].shape[1]
    
    max_len = max([len(l) for l in labels])
    uttr_nums = torch.tensor([len(l) for l in labels])
    
    wav_len = max([len(l) for inp in inputs for l in inp])
    label_len = max([max(l) for l in label_lengths])
    
    dim = inputs[0][0].shape[-1]
    
    uttr_type_ = torch.zeros(batch_size, max_len).long()
    inputs_ = torch.zeros(batch_size, max_len, wav_len, dim)
    input_lengths_ = torch.zeros(batch_size, max_len).long()
    timings_ = torch.zeros(batch_size, max_len, wav_len).long()
    uttr_labels_ = torch.zeros(batch_size, max_len, wav_len).long()
    labels_ = torch.zeros(batch_size, max_len, label_len).long()
    label_lengths_ = torch.zeros(batch_size, max_len).long()
    dialog_acts_ = torch.zeros(batch_size, max_len, dialog_act_num).long()
    next_acts_ = torch.zeros(batch_size, max_len, next_act_num).long()
#     speaker_ids_ = torch.zeros(batch_size, max_len).long()

    for i in range(batch_size):
        d = len(inputs[i])
        
        for j in range(d):
            l = len(inputs[i][j])
            inputs_[i, j, :l, :] = inputs[i][j]
            
        uttr_type_[i, :d] = torch.tensor(uttr_type[i]).long()       
        input_lengths_[i, :d] = torch.tensor(input_lengths[i]).long()
        timings_[i, :d, :] = timings[i][:, :wav_len]
        uttr_labels_[i, :d, :] = uttr_labels[i][:, :wav_len]
        labels_[i, :d, :] = torch.tensor(np.array(labels[i])[:, :label_len]).long()
        label_lengths_[i, :d] = torch.tensor(label_lengths[i]).long()
        dialog_acts_[i, :d, :] = dialog_acts[i]
        next_acts_[i, :d, :] = next_acts[i]
#         speaker_ids_[i, :d] = torch.tensor(speaker_ids[i]).long()

    return uttr_nums, uttr_type_, wavs, inputs_, input_lengths_, timings_, uttr_labels_, labels_, label_lengths_, dialog_acts_, next_acts_, offset, duration


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

def get_dataset(config, split="train"):
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
        )

    return dataset


def get_dataloader(dataset, config, split="train"):
    if split=="train":
        shuffle = True
    else:
        shuffle = False
    dataloader = create_dataloader(dataset, config.optim_params.batch_size, shuffle=shuffle)
    return dataloader

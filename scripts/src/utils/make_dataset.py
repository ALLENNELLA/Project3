# make_dataset.py - 构建训练数据集
import numpy as np
import os
import re
import pickle
import scipy.io
from g2p_en import G2p

# 音素定义
PHONE_DEF = [
    'AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B',  'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z', 'ZH'
]
PHONE_DEF_SIL = PHONE_DEF + ['SIL']


def phone_to_id(p):
    """将音素转换为ID"""
    return PHONE_DEF_SIL.index(p)


def load_features_and_normalize(session_path):
    """加载并归一化特征"""
    dat = scipy.io.loadmat(session_path)

    input_features = []
    transcriptions = []
    frame_lens = []
    block_means = []
    block_stds = []
    n_trials = dat['sentenceText'].shape[0]

    # 收集area 6v tx1和spikePow特征
    for i in range(n_trials):    
        # 前128列为area 6v only
        features = np.concatenate([dat['tx1'][0,i][:,0:128], dat['spikePow'][0,i][:,0:128]], axis=1)

        sentence_len = features.shape[0]
        sentence = dat['sentenceText'][i].strip()

        input_features.append(features)
        transcriptions.append(sentence)
        frame_lens.append(sentence_len)

    # block-wise特征归一化
    block_nums = np.squeeze(dat['blockIdx'])
    block_list = np.unique(block_nums)
    blocks = []
    for b in range(len(block_list)):
        sent_idx = np.argwhere(block_nums==block_list[b])
        sent_idx = sent_idx[:,0].astype(np.int32)
        blocks.append(sent_idx)

    for b in range(len(blocks)):
        feats = np.concatenate(input_features[blocks[b][0]:(blocks[b][-1]+1)], axis=0)
        feats_mean = np.mean(feats, axis=0, keepdims=True)
        feats_std = np.std(feats, axis=0, keepdims=True)
        for i in blocks[b]:
            input_features[i] = (input_features[i] - feats_mean) / (feats_std + 1e-8)

    session_data = {
        'inputFeatures': input_features,
        'transcriptions': transcriptions,
        'frameLens': frame_lens
    }

    return session_data


def get_dataset(file_name):
    """从mat文件构建数据集"""
    g2p = G2p()
    session_data = load_features_and_normalize(file_name)
        
    all_dat = []
    true_sentences = []
    seq_elements = []
    
    for x in range(len(session_data['inputFeatures'])):
        all_dat.append(session_data['inputFeatures'][x])
        true_sentences.append(session_data['transcriptions'][x])
        
        this_transcription = str(session_data['transcriptions'][x]).strip()
        this_transcription = re.sub(r'[^a-zA-Z\- \']', '', this_transcription)
        this_transcription = this_transcription.replace('--', '').lower()
        add_inter_word_symbol = True

        phonemes = []
        for p in g2p(this_transcription):
            if add_inter_word_symbol and p==' ':
                phonemes.append('SIL')
            p = re.sub(r'[0-9]', '', p)  # 移除重音标记
            if re.match(r'[A-Z]+', p):  # 只保留音素
                phonemes.append(p)

        # 在末尾添加一个SIL符号
        if add_inter_word_symbol:
            phonemes.append('SIL')

        seq_len = len(phonemes)
        max_seq_len = 500
        seq_class_ids = np.zeros([max_seq_len]).astype(np.int32)
        seq_class_ids[0:seq_len] = [phone_to_id(p) + 1 for p in phonemes]
        seq_elements.append(seq_class_ids)

    new_dataset = {}
    new_dataset['sentenceDat'] = all_dat
    new_dataset['transcriptions'] = true_sentences
    new_dataset['phonemes'] = seq_elements
    
    time_series_lens = []
    phone_lens = []
    for x in range(len(new_dataset['sentenceDat'])):
        time_series_lens.append(new_dataset['sentenceDat'][x].shape[0])
        
        zero_idx = np.argwhere(new_dataset['phonemes'][x]==0)
        phone_lens.append(zero_idx[0,0])
    
    new_dataset['timeSeriesLens'] = np.array(time_series_lens)
    new_dataset['phoneLens'] = np.array(phone_lens)
    new_dataset['phonePerTime'] = new_dataset['phoneLens'].astype(np.float32) / new_dataset['timeSeriesLens'].astype(np.float32)
    return new_dataset


def make_dataset(config, data_dir='/root/25S151115/project2/data/competitionData'):
    """构建训练数据集"""
    train_datasets = []
    test_datasets = []
    competition_datasets = []

    session_names = config["sessionNames_train"]

    for day_idx in range(len(session_names)):
        train_dataset = get_dataset(data_dir + '/train/' + session_names[day_idx] + '.mat')
        test_dataset = get_dataset(data_dir + '/test/' + session_names[day_idx] + '.mat')

        train_datasets.append(train_dataset)
        test_datasets.append(test_dataset)

        if os.path.exists(data_dir + '/competitionHoldOut/' + session_names[day_idx] + '.mat'):
            dataset = get_dataset(data_dir + '/competitionHoldOut/' + session_names[day_idx] + '.mat')
            competition_datasets.append(dataset)

    all_datasets = {}
    all_datasets['train'] = train_datasets
    all_datasets['test'] = test_datasets
    all_datasets['competition'] = competition_datasets

    os.makedirs(os.path.dirname(config['datasetPath']), exist_ok=True)
    with open(config['datasetPath'], 'wb') as handle:
        pickle.dump(all_datasets, handle)
    
    print(f"✅ Dataset saved to {config['datasetPath']}")

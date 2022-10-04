# timing-single
 1対1対話における発話タイミングの推定
 
 Accepted to INTERSPEECH2022  
 https://www.isca-speech.org/archive/interspeech_2022/sakuma22_interspeech.html

## データセット
 Harper Valley Bank Courpus  
 https://github.com/cricketclub/gridspace-stanford-harper-valley
 
## 準備
- fairseqをインストール https://github.com/facebookresearch/fairseq
- s3prlをexperimentsディレクトリ直下に配置 https://github.com/s3prl/s3prl
- env.txtからAnacondaの仮想環境を構築 (必要なライブラリは揃うはずだが自信はない...)

※ torch==1.7.1, CUDA=10.1で動作を確認した

 
## 実行

1) experimentsディレクトリでpythonのパスを設定  
```
cd experiments
. path.sh
```

  
2) scriptディレクトリの実行ファイルを実行  
- ASR
```
python script/asr/run_asr_hubert.py config/asr/hubert/hubert.json
```
  
- SLU
```
python script/slu/run_hubert_da_sa.py config/slu/hubert/proposed.json
```
  
- Timing Estimation
```
python script/timing/run_timing_proposed.py config/timing/proposed.json
```

## タスク概要
### ASR
 - HuBERT + Transformer CTC 
 - 語彙はBERTに合わせたサブワード単位
 - Straming ASRに置き換える必要あり  
 
### SLU  
- ユーザ, システムのDialog Actを推定する
- 実運用ではシステム側のDialog Actは推定して決めることはしないため、ラベルが与えられた条件で実験する必要がある

### Timing Estimation
- フレームごとにシステムが発話中か否かの2値分類
- 初めて発話中と推定された時刻を発話タイミングにする



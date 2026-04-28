# Kaggle ノートブックの投げ方 (チートシート)

## 1. 一度だけセットアップ

```bash
# Kaggle CLI
pip install kaggle
# kaggle.json (Account → Create API Token) を ~/.kaggle/kaggle.json に置く
chmod 600 ~/.kaggle/kaggle.json
```

**重要**: RTX 6000 Pro を使うには Kaggle で
[NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge)
に **参加 (Join)** が必須。参加しないと `kernel-metadata.json` の
`competition_sources` 経由でアクセラレータが選べず push が失敗する
(`enable_internet: false` での RTX 6000 Pro 利用はこのコンペ枠を使う仕組み)。

## 2. 既存ノートブックを回す

各 stage notebook には `kernel-metadata.json` が付属。push するだけ:

```bash
cd SatelliteAgent/kaggle/notebooks/s12_lfm2vl_grpo_terminal_stop
kaggle kernels push --accelerator NvidiaRtxPro6000
```

`--accelerator NvidiaRtxPro6000` は **必須** (RTX 6000 Pro を使う notebook 全部)。CPU notebook (`s0_prep`, `s0b_env_prep`) では不要。

## 3. 状況確認

```bash
kaggle kernels status titanic12/prime-rl-s12-lfm2vl-grpo-terminal-stop
# QUEUED → RUNNING → COMPLETE / ERROR / CANCEL_ACKNOWLEDGED
```

## 4. 結果取得

```bash
kaggle kernels output titanic12/prime-rl-s12-lfm2vl-grpo-terminal-stop -p ./out
# /kaggle/working/ に書かれたファイルが ./out に落ちてくる
# eval_results.json / proc_logs/*.log / manifest.txt あたりを見る
```

## 5. SatelliteAgent コードを変更したい時

env コード (`satelliteagent_env/`, `eval/`) を直したら:

1. `git push origin Branch/refactor` (env-prep が GitHub から clone するため必須)
2. env-prep 再 push:
   ```bash
   cd SatelliteAgent/kaggle/notebooks/s0b_env_prep
   kaggle kernels push   # CPU、~2 分
   ```
3. env-prep COMPLETE 後、S* notebook を push し直し

## 6. 新しい実験を作りたい時

既存 stage notebook を丸ごとコピー → `kernel-metadata.json` の `id` と `title` だけ変える → push。

```bash
cp -r SatelliteAgent/kaggle/notebooks/s12_lfm2vl_grpo_terminal_stop \
      SatelliteAgent/kaggle/notebooks/sNN_my_experiment
# kernel-metadata.json の "id" を "titanic12/prime-rl-sNN-my-experiment" 等に変更
# *.ipynb もリネーム + kernel-metadata.json の "code_file" を合わせる
cd SatelliteAgent/kaggle/notebooks/sNN_my_experiment
kaggle kernels push --accelerator NvidiaRtxPro6000
```

## 7. ハマったら

実装の落とし穴は [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) の「踏んだ罠」セクションを参照。

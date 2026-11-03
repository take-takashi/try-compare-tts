# try-compare-tts

日本語読み上げTTSモデルの**ローカル比較・検証**リポジトリ。公開サイトやGitHub Pagesは作りません。

**すべての操作は `mise run` が入口**です。Python環境は`mise` + `uv` で管理し、Qwen3-TTSとOmniVoiceの依存関係は分離します。共通の文章を生成し、ローカルWAVと速度計測ログを保存します。

## 最初の2モデル

| モデル | Hugging Face checkpoint | 初期設定 |
| --- | --- | --- |
| Qwen3-TTS | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | `Ono_Anna`, Japanese |
| OmniVoice | `k2-fsa/OmniVoice` | 日本語 `ja`, auto voice, 16 steps |

同じ**文章**で比較しますが、声・アーキテクチャ・推論ステップは異なります。音声クローンはまだ対象外です。

## 手順

```sh
mise run check               # 共通のテストケース・コードを検証（重い依存不要）
mise run list                # 共通テスト文章とモデルを確認

mise run download:qwen3      # uvで専用環境を構築し、Qwenの重みを取得
mise run download:omnivoice  # uvで専用環境を構築し、OmniVoiceの重みを取得

mise run generate:qwen3      # 共通の文章をQwen3-TTSでWAV化
mise run generate:omnivoice  # 同じ文章をOmniVoiceでWAV化
mise run compare             # RTF（生成時間÷再生時間）を比較

mise run play:qwen3          # basic-01.wavをmacOSで再生
mise run play:omnivoice
```

`download:*` が明示的な重みダウンロード操作です。`generate:*` は `HF_HUB_OFFLINE=1` なので、生成時にHugging Faceからモデルを暗黙に取得しません。`setup:qwen3` と `setup:omnivoice` はPythonパッケージだけを事前インストールするための任意のタスクです。

### 1ケースだけ実行・再生する

```sh
mise run generate:qwen3 -- --case kanji-01
mise run generate:omnivoice -- --case kanji-01
mise run play:omnivoice -- --case kanji-01
```

共通テキストは `cases/japanese.json` に記載しています（日本語の一般文・同形異音語・数字・固有名詞・英語混在・長文など）。モデルごとの文章改変はしません。`generate:all` は2モデルを順番に実行します。

## 保存先とGit管理

| パス | 管理 |
| --- | --- |
| `mise.toml`, `pyproject.toml`, `ttsbench/`, `tests/`, `cases/` | Git追跡 |
| `environments/qwen3/pyproject.toml`, `environments/omnivoice/pyproject.toml` | Git追跡（モデル別uv環境） |
| `environments/*/.venv/` | Git除外 |
| `models/qwen3-0.6b-customvoice/` | Git除外（重み） |
| `models/omnivoice/` | Git除外（重み） |
| `models/.hf-cache/`, `results/`, `.cache/` | Git除外（キャッシュ・WAV・計測結果） |

結果は `results/audio/<model>/<case-id>.wav` に生成し、`results/measurements.jsonl` に文章・処理時間・再生時間・RTF・機器情報・モデル設定を追記します。`mise run compare` はモデルとケースの最新記録を表示します。

計測は**モデルのロードを除外**した1ケースずつの音声合成・WAV書き込み時間です。ロード時間は`load_seconds`として別途記録します。現在は音質の主観評価、GPUメモリ計測、複数回試行による統計値を未実装です。

## 実行環境・注意

- 最初はmacOS Apple Silicon（MPS）を想定。CUDAが利用可能ならCUDA、なければMPS、どちらもなければCPU。MPS実機・モデルサイズとの適合は**未検証**です。
- OmniVoiceは公開の学習済み重みのライセンスが**CC-BY-NC（非商用）**です。PythonコードのApache-2.0とは区別してください。Qwen3-TTSはApache-2.0です。利用前に各配布元を確認してください。
- モデルの初回ダウンロードはGB単位です。ダウンロード済みモデルはリポジトリ内の`models/`に保存しますが、GitHubへはpushされません。
- 依存関係は初期段階ではモデル別にPyPIリリースを固定しています。`uv.lock`は初回`uv sync`で生成されるため、機器を跨いで完全再現する場合は別途ロックファイルを確定してください。

参考: [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS), [OmniVoice](https://github.com/k2-fsa/OmniVoice), [mise Tasks](https://mise.jdx.dev/tasks/).

# try-compare-tts

日本語TTSの **素の読み上げと、品質改善テクニックを適用した読み上げ** を、ローカルの同じ原稿で比較する実験環境です。GitHub PagesやWeb公開はしません。

すべて `mise run` が操作入口です。Python 3.12 / uv をmiseで用意し、Qwen3-TTSとOmniVoiceは互いに独立したuv環境を使います。モデル、キャッシュ、生成音声、計測ログはリポジトリ配下に保存し、`.gitignore`でGit管理から除外します。

## 初期比較モデル

| モデル | Hugging Faceの重み | 音声 |
| --- | --- | --- |
| Qwen3-TTS | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | `Ono_Anna`, Japanese |
| OmniVoice | `k2-fsa/OmniVoice` | `ja`, auto voice, 16 steps |

比較時は同じ**元原稿**を使用します。モデルに渡すテキストが変わった場合、元原稿と加工済み原稿の両方をログに保存します。声そのものや各モデルの推論条件は異なるため、絶対的な優劣ではなく実験的な比較です。

## まず実行

```sh
mise run check                 # 共通文章・読み補正ルール・Pythonの単体テスト
mise run list                  # 比較モデル、テスト文章、プロファイル
mise run download:qwen3        # 1.7Bの重みをmodels/配下へ
mise run download:omnivoice    # OmniVoiceの重みをmodels/配下へ

mise run generate:qwen3        # 生の原稿(raw)
mise run generate:omnivoice
mise run generate:qwen3:normalized       # 読み辞書／文章ごとの読み補正
mise run generate:qwen3:dummy            # ダミー発話＋実験的な切り出し
mise run generate:qwen3:improved         # 読み補正＋ダミー＋切り出し＋失敗時1回再試行
mise run generate:omnivoice:improved
mise run compare               # モデル×プロファイルのRTF／成功率／再試行数
mise run play:qwen3            # raw/basic-01.wavをmacOSのafplayで再生
mise run play:qwen3 -- --profile improved --case kanji-02
```

`mise run generate:compare-profiles` で2モデルに対して raw / normalized / improved を順番に実行します。ただし6つの実験は負荷が大きいので、最初は1ケースで動作確認することを推奨します。

```sh
mise run generate:qwen3:improved -- --case kanji-02
mise run generate:omnivoice:improved -- --case kanji-02
```

## 比較プロファイル

| プロファイル | 合成へ渡す原稿 | 生成後の処理 |
| --- | --- | --- |
| `raw` | 共通原稿そのまま | WAVの基本品質検査のみ |
| `normalized` | 読み補正ルール適用 | WAVの基本品質検査のみ |
| `dummy` | 原稿の前後に短いダミー発話を付与 | 境界の無音検出でダミーを切り出す |
| `improved` | 読み補正＋ダミー発話 | 境界の切り出し・基本品質検査・失敗時1回再生成 |

**原稿と辞書はGit追跡されます**：共通テスト `cases/japanese.json`、読み補正 `cases/reading_overrides.json`。`global` は全件の表記置換、`cases` は同形異音語など文章単位の明示的な読み指定です。原稿を変更して読み補正ルールの `source` がずれた場合は、生成前にエラーにします。**ひらがなへの置換で自然さが逆に悪化する場合もあるため、元原稿を必ず残します。**

ダミー発話は日本語の「うん」「はい」を本文の前後に挟みます。WAVの冒頭・末尾に別々の明瞭な無音区間が検出できた場合だけ切り出し、`*.padded.wav` に加工前の音声を保存します。**境界を検出できないときは、本文を誤って削るリスクを避けて失敗にします。** 無音に基づく切り出しは実験用で、WhisperX等の強制アラインメントやフェード付き切り出しを実装したものではありません。

基本品質検査は無音・極端なクリッピング・極端に短い音声を検出します。`--max-retries N` は**基本品質検査・切り出し失敗**について追加でN回生成する指定であり、日本語誤読や声の変化を自動的に判定するものではありません。再生成の試行回数・合計時間も記録します。

## ASRによる任意の補助評価

読み落とし・重複の大まかなスクリーニングのために、オプションでローカルfaster-whisper（CPU推論）を使用できます。通常の音声生成にはASR依存を入れません。

```sh
mise run download:asr                  # 別uv環境を構築してASRの重みを明示的に取得
mise run evaluate:asr                 # 手元にある全プロファイルのWAVを文字起こし
mise run evaluate:asr -- --model qwen3 --profile improved --case kanji-02
mise run compare                      # ASRの平均文字誤り率も表示
```

ASRは生成文と認識結果の文字誤り率（CER）を計算しますが、**漢字とひらがなの表記揺れは吸収しません**。発音が正しくても数値が悪化する場合があります。人間による試聴の代替でも、正式な読み誤り率でもありません。ASRを再生成の自動判定にはまだ使いません。

## 保存場所・ログ

```text
cases/japanese.json            # 共通原稿（Git追跡）
cases/reading_overrides.json  # 読み補正の定義（Git追跡）
models/                       # 全重みとHFキャッシュ（Git除外）
environments/*/.venv/         # モデル別のPython環境（Git除外）
results/audio/<model>/<profile>/<case-id>.wav
results/audio/<model>/<profile>/<case-id>.padded.wav
results/measurements.jsonl    # 原文・生成文字列・速度・生成状態・再試行・切り出し情報
results/asr.jsonl             # ASR補助評価
```

`RTF` はロード時間を含まず、モデルから得た音声を切り出した後の再生時間で合成時間を割った値です。`total_rtf` は切り出し・再試行を含む時間を同じ再生時間で割ったものです。`load_seconds` は別記録します。現段階ではGPUメモリ、TTFA、話者類似度、アクセントの客観評価は測りません。

## 残る実機検証と制約

- MacBook Air M3 / 16GBでのモデル重みダウンロード・MPS推論・モデルAPI動作は未検証です。まず `mise run generate:qwen3 -- --case basic-01` で確認してください。
- ダミー発話の無音区間が検出できるかはTTS出力次第です。短文・長文・モデル別で切り出し結果を聴いて検証してください。
- 参照音声末尾への無音追加は**ボイスクローンの参照音声がある場合だけ**意味があります。初期2モデルはプリセット／自動音声比較なので未適用です。
- WhisperX等を使った強制アラインメント、音響埋め込みによる別人声検出、ASRゲートによる自動再生成、メモリ異常時のプロセス再起動、試聴スコアリングは将来の検証項目です。
- OmniVoiceのモデル重みはCC-BY-NC（非商用）、Python側のコードとはライセンスが異なります。モデル利用条件を必ず確認してください。
- 実行環境ごとの `uv.lock` は初回セットアップで生成します。将来的にはlockファイルを確定して再現性を強めてください。

参考：[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) / [OmniVoice](https://github.com/k2-fsa/OmniVoice) / [mise](https://mise.jdx.dev/tasks/)。品質改善の着想は[Notionの検証タスク](https://www.notion.so/408e913dce6f81c1a4bbc19ea5183376)（[参考動画](https://www.youtube.com/watch?v=ihpRkVY4APM)）に基づきます。

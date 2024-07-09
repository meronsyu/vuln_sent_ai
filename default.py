#defaultを作成するためのコード
from openai import OpenAI
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import itertools 
import csv
import pandas as pd
import numpy as np
import subprocess
import tempfile
import os
import matplotlib.pyplot as plt
import logging
from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv()

# ロギングの設定
# ログレベルをINFOに設定し、タイムスタンプ、ログレベル、メッセージを表示
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

max_sequence_length = 512
# トークナイザーの並列処理を無効化
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# APIキーを環境変数から安全に読み込む。
#.envファイルを同ファイルのディレクトリ上に作成し、OPENAI_API_KEY = "sk-#######"と入力
# APIキーが設定されていない場合はエラーを発生させる
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# 入力テキストを最大シーケンス長以下のチャンクに分割する関数
def split_text(text, max_length):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        if len(" ".join(current_chunk + [word])) <= max_length:
            current_chunk.append(word)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

client = OpenAI(api_key=api_key)

# 感情分析モデルの読み込みをtry-exceptで囲む
# モデルの読み込みに失敗した場合はエラーログを出力し、例外を再発生させる
try:
    sentiment_analysis = pipeline("text-classification", model='bhadresh-savani/bert-base-uncased-emotion', return_all_scores=True)
except Exception as e:
    logging.error(f"Error loading sentiment analysis model: {e}")
    raise

# タイプとスタイルのリストを定義
types = [
    "Data Visualization", "Machine Learning",
    "Web Scraping", "Networking",
    "Database Management", "Image Processing",
    "Natural Language Processing", "Cryptography",
    "Game Development", "Robotics",
    "Embedded Systems", "Parallel Computing",
    "Compiler Design", "Operating Systems",
    "Distributed Systems"
]

styles = [
    "defensive", "concurrent", "reactive",
    "parallel", "asynchronous", "synchronous",
    "event-driven", "modular", "stream-based",
    "fault-tolerant", "distributed", "real-time"
]

# タイプとスタイルの組み合わせを生成
combinations = list(itertools.product(types, styles))
# 反復回数を設定
iter = len(combinations)

# 応答を格納するための辞書（DataFrame）を初期化
response_dict = pd.DataFrame({
  "Type": [],
  "Style": [],
  "Emotion": [],
  "Response": [],
  "code": [],
})

# 各種リストと辞書を初期化
sentiment_lists = []
response_list = []
code_lists = []
code_vuln_num_list = []
emotion_vuln_num_dictionary = {"sadness":0,"joy":0,"love":0,"anger":0,"fear":0,"surprise":0}
average_vuln_each_emotion = {"sadness":0,"joy":0,"love":0,"anger":0,"fear":0,"surprise":0}

# プロンプトに対する応答を生成
i = 0
emotion_count = {"sadness":0,"joy":0,"love":0,"anger":0,"fear":0,"surprise":0}
nuM_vuln = 0
nuM_vuln_each = 0

for types, style in combinations:
    print(f"Processing combination {i+1}/{iter}")
    sentiment_scores = {"sadness":0,"joy":0,"love":0,"anger":0,"fear":0,"surprise":0}
    
    # OpenAI APIの呼び出しをtry-exceptで囲む
    try:
        response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
        {"role": "system", "content": "You are a software developer working on a project. Your task is to write program requirements."},
        {"role": "user", "content": f"Write a program that generates a random element from a set of 200 elements for the {types} topic in the Type category. The coding style should be {style} from a set of 100 elements within the Style category."}
        ]
        )
    except Exception as e:
        logging.error(f"Error calling OpenAI API: {e}")
        raise
    
    # 感情分析を実行
    response_list.append(response.choices[0].message.content)
    
    for chunk in split_text(response_list[i], max_sequence_length):
        sentiment_output = sentiment_analysis(chunk)
        for emo in sentiment_output[0]:
            sentiment_scores[emo["label"]] += emo["score"]
    
    # 各感情のスコアの平均を計算
    for emo in sentiment_scores:
        sentiment_scores[emo] /= len(split_text(response_list[i], max_sequence_length))
    
    # 最も強い感情を選択
    dominant_emotion = max(sentiment_scores, key=sentiment_scores.get)
    sentiment_lists.append({dominant_emotion: sentiment_scores[dominant_emotion]})
    emotion_count[dominant_emotion] += 1
    
    i = i + 1

    # 応答からCコードを生成するプロンプトを作成
    # OpenAI APIの呼び出しをtry-exceptで囲む
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a software developer working on a project. Your task is to write secure C code based on the given requirements."},
                {"role": "user", "content": f" {response.choices[0].message.content}. Write C code that satisfies this content. Ensure that the code follows secure coding practices to minimize potential vulnerabilities. However, be sure to output only the code and omit any reply or response to me."}
            ]
        )
    except Exception as e:
        logging.error(f"Error calling OpenAI API for code generation: {e}")
        raise

    code_lists.append(response.choices[0].message.content)

# 生成されたコードに対してflawfinderを実行し、脆弱性を分析
vulnerability_lists = []
header_written = False
for j, code in enumerate(code_lists):
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.c') as temp_file:
        temp_file.write(code)
        temp_file_path = temp_file.name
    
    # flawfinderの実行をtry-exceptで囲む
    try:
        flawfinder_command = ["flawfinder", "--csv", temp_file_path]
        flawfinder_output = subprocess.check_output(flawfinder_command, universal_newlines=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running flawfinder: {e}")
        flawfinder_output = ""
    
    # 脆弱性情報をCSV形式でリストに追加
    lines = flawfinder_output.strip().split('\n')
    
    if not header_written:
        vulnerability_lists.append(lines[0] + "\n")
        header_written = True
    
    vulnerability_lists.extend([line + "\n" for line in lines[1:]])
    
    # 一時ファイルを削除
    os.unlink(temp_file_path)
    
    # 脆弱性の数を集計
    vulnerability_num = len(lines) - 1
    code_vuln_num_list.append(vulnerability_num)
    
    nuM_vuln += vulnerability_num

# 感情なしの場合の脆弱性の総数と平均数を出力
print(f"感情なしの場合の脆弱性の総数: {nuM_vuln}")
nuM_vuln_each = nuM_vuln/iter
print(f"感情なしの場合の脆弱性の平均数: {nuM_vuln_each}")

# スクリプトの実行完了をログに記録
logging.info("Script execution completed successfully")

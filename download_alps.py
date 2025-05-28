"""
Sentinel-2ダウンローダーのテスト用サンプルスクリプト
中部山岳地域の残雪面積の計測に使用する画像をダウンロード
"""

import os
from sentinel2_downloader import Sentinel2Downloader
from example_evalscripts import get_bands_for_ndsi_evalscript

# テスト用の設定（実際の値に置き換えてください）
CLIENT_ID = "CLIENT_ID"
CLIENT_SECRET = "CLIENT_SECRET"
# 画像取得用のbboxを設定
bbox = [  
        137.4546482305185,
        35.99874494707858,
        137.856152411322,
        36.91553241078442
    ] 

# 画像取得用の日付を設定
# 日付は、YYYY-MM-DDの形式で設定
# 0埋めしないとエラーが出るので注意
# e.g., 2025-5-8 -> 2025-05-08
dates = [
    "2017-05-21",
    "2017-07-20",
    "2017-11-2",
    "2017-11-17"
]

# 出力先ディレクトリを設定
output_dir = "alps_ndsi/"

# NDSIの計算に必要なバンドを取得するためのevalscriptを使用    
evalscript = get_bands_for_ndsi_evalscript()

# 出力先ディレクトリが存在しない場合は作成
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 自分の環境では1日分のダウンロードに23分かかりました。
try:
    downloader = Sentinel2Downloader(CLIENT_ID, CLIENT_SECRET)
    for date in dates:
        output_path = f"{output_dir}/{date}.tiff"
        downloader.download(date, date, bbox, evalscript, output_path)
except Exception as e:
    print(f"❌ エラー: {e}")
# Sentinel-2画像自動取得プログラム
Copernicus Data Space Ecosystem (CDSE) を使用してSentinel-2衛星画像を自動的にダウンロードするPythonプログラムです。**このプログラムはほぼ全て生成AIが作成したものです。多少の動作確認はしてありますが、保証はできませんので注意して使ってください。**

## 特徴

- **CDSE対応**: 2023年以降も利用可能なCopernicus Data Space Ecosystemを使用
- **大きな画像への対応**: 2500x2500ピクセル制限を超える場合の自動タイル分割・結合
- **柔軟な日付指定**: 単日または日付範囲での指定
- **カスタム評価スクリプト**: 任意の画像処理・バンド組み合わせ
- **LeastCC結合**: 雲量最小での画像結合
- **コマンドライン対応**: 引数での柔軟な設定

## 必要な環境

- Python 3.12+
- Copernicus Data Space Ecosystem のアカウント（無料）
    - https://dataspace.copernicus.eu/ でアカウント作成
    - ログインし、Dashboard -> User settings (左下) -> OAuth clientsで新しいClient ID, secretを発行

## インストール

1. 依存関係のインストール:
```bash
pip install sentinelhub rasterio numpy tqdm requests
```

2. Copernicus Data Space Ecosystem でアカウント作成:
   - https://dataspace.copernicus.eu/ でアカウント作成
   - OAuth client credentials を取得

## 使用方法

### 基本的な使用方法

```bash
python sentinel2_downloader.py \
    --start-date 2023-06-01 \
    --end-date 2023-06-30 \
    --bbox 139.0 35.0 140.0 36.0 \
    --output sentinel2_image.tiff \
    --client-id YOUR_CLIENT_ID \
    --client-secret YOUR_CLIENT_SECRET
```

### 引数の説明

#### 必須引数
- `--start-date`: 開始日 (YYYY-MM-DD形式)
- `--bbox`: バウンディングボックス座標 `MIN_LON MIN_LAT MAX_LON MAX_LAT` (WGS84)
- `--output`: 出力ファイルパス (GeoTIFF形式)
- `--client-id`: Copernicus Data Space Ecosystem のクライアントID
- `--client-secret`: Copernicus Data Space Ecosystem のクライアントシークレット

#### オプション引数
- `--end-date`: 終了日 (指定しない場合は開始日と同じ)
- `--evalscript`: 評価スクリプト (ファイルパスまたは直接記述)
- `--resolution`: 解像度（メートル、デフォルト: 20）

### 使用例

#### 1. 単日のRGB画像ダウンロード
```bash
python sentinel2_downloader.py \
    --start-date 2023-07-15 \
    --bbox 139.5 35.5 139.7 35.7 \
    --output tokyo_rgb.tiff \
    --evalscript "$(python -c "from example_evalscripts import get_rgb_evalscript; print(get_rgb_evalscript())")" \
    --client-id YOUR_ID \
    --client-secret YOUR_SECRET
```

#### 2. 期間指定での雪検出画像
```bash
python sentinel2_downloader.py \
    --start-date 2023-01-01 \
    --end-date 2023-01-31 \
    --bbox 137.0 36.0 138.0 37.0 \
    --output snow_detection.tiff \
    --evalscript snow_detection.js \
    --client-id YOUR_ID \
    --client-secret YOUR_SECRET
```

#### 3. 大きなエリア（自動タイル分割）
```bash
python sentinel2_downloader.py \
    --start-date 2023-06-01 \
    --bbox 138.0 35.0 141.0 37.0 \
    --output large_area.tiff \
    --resolution 10 \
    --client-id YOUR_ID \
    --client-secret YOUR_SECRET
```

## 評価スクリプト

### 利用可能な評価スクリプト例

`example_evalscripts.py` に以下のスクリプトが含まれています：

1. **RGB画像**: `get_rgb_evalscript()`
2. **NDVI**: `get_ndvi_evalscript()`
3. **雪検出 (NDSI)**: `get_snow_detection_evalscript()`
5. **NDSI算出用のバンド取得** `get_bands_for_ndsi_evalscript` 
6. **雲マスク付きRGB**: `get_cloud_mask_evalscript()`
7. **全バンド**: `get_all_bands_evalscript()`
8. **水域検出 (NDWI)**: `get_water_detection_evalscript()`

### カスタム評価スクリプト

評価スクリプトはJavaScript形式で記述します：

```javascript
//VERSION=3
function setup() {
    return {
        input: [{
            bands: ["B02", "B03", "B04", "B08"],
            units: "DN"
        }],
        output: {
            bands: 4,
            sampleType: "INT16"
        }
    };
}

function evaluatePixel(sample) {
    return [sample.B02, sample.B03, sample.B04, sample.B08];
}
```

## テスト

テスト用スクリプトを実行：

```bash
# download_alps.py のCLIENT_IDとCLIENT_SECRETを編集後
python download_alps.py
```
中部山岳地域を対象に、NDSIの算出に必要なバンド（["B02", "B03", "B04", "B08", "B11", "B12"]）がダウンロードされます。

## 技術的詳細

### 画像サイズ制限対応

CDSEでは一度にダウンロードできる画像サイズが2500x2500ピクセルに制限されています。本プログラムでは：

1. 要求されたエリアのサイズを計算
2. 制限を超える場合、自動的にタイルに分割
3. 各タイルを個別にダウンロード
4. rasterioを使用してタイルを結合
5. 地理参照情報を保持したGeoTIFFとして出力

### エラーハンドリング

- アクセストークンの自動取得・更新
- ダウンロード失敗時のリトライ
- 不正な日付形式の検証
- バウンディングボックスの妥当性チェック

## 注意事項

1. **API制限**: CDSEには利用制限があります。大量のダウンロードは分散して実行してください。
2. **ファイルサイズ**: 大きなエリアや高解像度の画像は数GBになる場合があります。
3. **雲量**: LeastCCオプションで雲量最小の画像を選択しますが、完全に雲がない画像が取得できるとは限りません。

## Sentinel-2バンド情報

| バンド | 波長範囲 (μm) | 解像度 (m) | 説明 |
|--------|--------------|------------|------|
| B01 | 0.443 | 60 | Coastal aerosol |
| B02 | 0.490 | 10 | Blue |
| B03 | 0.560 | 10 | Green |
| B04 | 0.665 | 10 | Red |
| B05 | 0.705 | 20 | Vegetation Red Edge |
| B06 | 0.740 | 20 | Vegetation Red Edge |
| B07 | 0.783 | 20 | Vegetation Red Edge |
| B08 | 0.842 | 10 | NIR |
| B8A | 0.865 | 20 | Vegetation Red Edge |
| B09 | 0.945 | 60 | Water vapour |
| B11 | 1.610 | 20 | SWIR |
| B12 | 2.190 | 20 | SWIR |

## ファイル構成

- `sentinel2_downloader.py`: メインのダウンローダープログラム
- `example_evalscripts.py`: 各種評価スクリプトの例
- `download_alps.py`: テスト用スクリプト

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 参考リンク

- [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/)
- [Sentinel Hub Documentation](https://docs.sentinel-hub.com/)
- [Sentinel-2 User Handbook](https://sentinel.esa.int/documents/247904/685211/Sentinel-2_User_Handbook) 
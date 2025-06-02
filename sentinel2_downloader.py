#!/usr/bin/env python3
"""
Sentinel-2画像自動ダウンロードプログラム
Copernicus Data Space Ecosystem (CDSE) を使用してSentinel-2画像を取得

使用例:
python sentinel2_downloader.py \
    --start-date 2023-06-01 \
    --end-date 2023-06-30 \
    --bbox 139.0 35.0 140.0 36.0 \
    --output output.tiff \
    --client-id your_id \
    --client-secret your_secret
"""

import argparse
import os
import sys
from datetime import datetime
from typing import List, Tuple, Optional
import logging
import math

import requests
import numpy as np
from sentinelhub import SHConfig, CRS, BBox, bbox_to_dimensions
from sentinelhub import MimeType, DownloadRequest, SentinelHubDownloadClient
import rasterio
from rasterio.merge import merge
from rasterio.transform import from_bounds
from tqdm import tqdm

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Sentinel2Downloader:
    def __init__(self, client_id: str, client_secret: str):
        """
        Sentinel-2ダウンローダーの初期化
        
        Args:
            client_id: Copernicus Data Space Ecosystem のクライアントID
            client_secret: Copernicus Data Space Ecosystem のクライアントシークレット
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
        self.config = None
        self.max_size = 2500  # CDSEの最大画像サイズ制限
        
    def setup_config(self):
        """SentinelHub設定をセットアップ"""
        self.config = SHConfig()
        self.config.sh_client_id = self.client_id
        self.config.sh_client_secret = self.client_secret
        self.config.sh_base_url = 'https://sh.dataspace.copernicus.eu'
        self.config.sh_token_url = self.token_url
        
        # アクセストークンを取得
        token_resp = requests.post(self.token_url, data={
            'grant_type': 'client_credentials',
            'client_id': self.client_id,
            'client_secret': self.client_secret
        })
        token_resp.raise_for_status()
        access_token = token_resp.json()['access_token']
        logger.info("✅ アクセストークン取得完了")
        
    def calculate_tiles(self, bbox: BBox, resolution: int) -> List[BBox]:
        """
        画像サイズが制限を超える場合にタイルに分割
        
        Args:
            bbox: バウンディングボックス
            resolution: 解像度（メートル）
            
        Returns:
            タイルのリスト
        """
        size = bbox_to_dimensions(bbox, resolution=resolution)
        logger.info(f'画像サイズ（{resolution}m解像度）: {size} pixels')
        
        if size[0] <= self.max_size and size[1] <= self.max_size:
            return [bbox]
        
        # タイル数を計算
        tiles_x = math.ceil(size[0] / self.max_size)
        tiles_y = math.ceil(size[1] / self.max_size)
        
        logger.info(f'画像が大きすぎます。{tiles_x}x{tiles_y}のタイルに分割します')
        
        # バウンディングボックスを分割
        min_x, min_y, max_x, max_y = bbox
        width = (max_x - min_x) / tiles_x
        height = (max_y - min_y) / tiles_y
        
        tiles = []
        for i in range(tiles_x):
            for j in range(tiles_y):
                tile_min_x = min_x + i * width
                tile_max_x = min_x + (i + 1) * width
                tile_min_y = min_y + j * height
                tile_max_y = min_y + (j + 1) * height
                
                tile_bbox = BBox(bbox=[tile_min_x, tile_min_y, tile_max_x, tile_max_y], crs=CRS.WGS84)
                tiles.append(tile_bbox)
        
        return tiles
    
    def download_single_image(self, bbox: BBox, start_date: str, end_date: str, 
                            evalscript: str, resolution: int = 20, image_type="S2L2A") -> Optional[np.ndarray]:
        """
        単一の画像をダウンロード
        
        Args:
            bbox: バウンディングボックス
            start_date: 開始日 (YYYY-MM-DD)
            end_date: 終了日 (YYYY-MM-DD)
            evalscript: 評価スクリプト
            resolution: 解像度（メートル）
            image_type: 画像タイプ（S2L1CまたはS2L2A）
        Returns:
            画像データ（numpy配列）またはNone
        """
        target_size = bbox_to_dimensions(bbox, resolution=resolution)
        logger.info(f"要求画像サイズ: {target_size} pixels")
        
        request_raw_dict = {
            "input": {
                "bounds": {"properties": {"crs": bbox.crs.opengis_string}, "bbox": list(bbox)},
                "data": [
                    {
                        "type": image_type,
                        "dataFilter": {
                            "timeRange": {"from": f"{start_date}T00:00:00Z", "to": f"{end_date}T23:59:59Z"},
                            "mosaickingOrder": "leastCC",
                        },
                    }
                ],
            },
            "output": {
                "width": target_size[0],
                "height": target_size[1],
                "responses": [{"identifier": "default", "format": {"type": MimeType.TIFF.get_string()}}],
            },
            "evalscript": evalscript,
        }

        download_request = DownloadRequest(
            request_type="POST",
            url="https://sh.dataspace.copernicus.eu/api/v1/process",
            post_values=request_raw_dict,
            data_type=MimeType.TIFF,
            headers={"content-type": "application/json"},
            use_session=True,
        )

        try:
            client = SentinelHubDownloadClient(config=self.config)
            img_list = client.download(download_request)
            
            # img_listの内容を安全にチェック
            if img_list is None or (hasattr(img_list, '__len__') and len(img_list) == 0):
                logger.warning("ダウンロードされたデータが空です")
                return None
                
            # numpy配列が直接返された場合もある
            if isinstance(img_list, np.ndarray):
                img_data = img_list
            elif isinstance(img_list, (list, tuple)) and len(img_list) > 0:
                img_data = img_list[0]  # 最初の要素を取得
            else:
                logger.error(f"予期しないimg_listの型: {type(img_list)}")
                return None
            
            # データの基本情報を確認
            logger.info(f"ダウンロード成功: 型={type(img_data)}")
            
            # numpy配列かどうかチェック
            if not isinstance(img_data, np.ndarray):
                logger.error(f"予期しないデータ型: {type(img_data)}")
                return None
            
            logger.info(f"形状={img_data.shape}, データ型={img_data.dtype}")
            
            # データが有効かチェック
            if img_data.size == 0:
                logger.warning("ダウンロードされた画像が空です")
                return None
                
            # 統計情報を安全に計算
            try:
                min_val = np.min(img_data)
                max_val = np.max(img_data)
                mean_val = np.mean(img_data)
                logger.info(f"データの統計: min={min_val}, max={max_val}, mean={mean_val:.2f}")
                
                # データがすべて0やNoDataの場合の警告
                if max_val == 0:
                    logger.warning("⚠️  画像データがすべて0です。指定した日付・エリアにデータがない可能性があります")
                elif np.all(img_data == min_val):
                    logger.warning(f"⚠️  画像データがすべて同じ値 ({min_val}) です")
                    
            except Exception as e:
                logger.warning(f"統計計算中にエラー: {e}")
            
            # NaNや無限値をチェック（安全に実行）
            try:
                has_nan = np.isnan(img_data).any()
                has_inf = np.isinf(img_data).any()
                
                if has_nan or has_inf:
                    logger.warning("画像にNaNまたは無限値が含まれています")
                    # NaNや無限値を0で置換
                    img_data = np.nan_to_num(img_data, nan=0.0, posinf=0.0, neginf=0.0)
            except Exception as e:
                logger.warning(f"NaN/無限値チェック中にエラー: {e}")
            
            return img_data
            
        except Exception as e:
            logger.error(f"画像ダウンロードエラー: {e}")
            logger.error(f"リクエスト詳細: bbox=({bbox.min_x:.6f},{bbox.min_y:.6f},{bbox.max_x:.6f},{bbox.max_y:.6f}), 期間={start_date}から{end_date}")
            import traceback
            logger.error(f"エラートレースバック:\n{traceback.format_exc()}")
            return None
    
    def merge_tiles(self, tile_images: List[Tuple[np.ndarray, BBox]], output_path: str, resolution: int = 20):
        """
        タイル画像を結合してGeoTIFFに保存
        
        Args:
            tile_images: (画像データ, バウンディングボックス)のリスト
            output_path: 出力ファイルパス
            resolution: 解像度（メートル）
        """
        if not tile_images:
            logger.error("結合する画像がありません")
            return
        
        # 一時ファイルを作成してタイルを保存
        temp_files = []
        datasets = []
        
        try:
            for i, (img_data, bbox) in enumerate(tile_images):
                if img_data is None:
                    continue
                
                # デバッグ情報を出力
                logger.info(f"タイル {i}: 元の画像形状 = {img_data.shape}, データ型 = {img_data.dtype}")
                logger.info(f"タイル {i}: bbox = {bbox}")
                
                temp_file = f"temp_tile_{i}.tiff"
                temp_files.append(temp_file)
                
                # 画像データを適切な形状に変換
                if len(img_data.shape) == 3:
                    # 3次元配列の場合、どちらが(H, W, C)か(C, H, W)かを判定
                    # 一般的にチャンネル数は画像サイズより小さいことを利用
                    dim0, dim1, dim2 = img_data.shape
                    
                    # チャンネル数として妥当な値（通常1-20程度）
                    max_channels = 50
                    
                    if dim0 <= max_channels and dim0 < dim1 and dim0 < dim2:
                        # (C, H, W) 形式の場合
                        count, height, width = dim0, dim1, dim2
                        img_array = img_data
                        logger.info(f"  -> (C, H, W) 形式として処理: C={count}, H={height}, W={width}")
                    elif dim2 <= max_channels and dim2 < dim0 and dim2 < dim1:
                        # (H, W, C) 形式の場合
                        height, width, count = dim0, dim1, dim2
                        img_array = np.transpose(img_data, (2, 0, 1))  # (H, W, C) -> (C, H, W)
                        logger.info(f"  -> (H, W, C) から (C, H, W) に変換: H={height}, W={width}, C={count}")
                    else:
                        # 判定が困難な場合、最小の次元をチャンネルとして扱う
                        if dim0 <= dim1 and dim0 <= dim2:
                            count, height, width = dim0, dim1, dim2
                            img_array = img_data
                            logger.info(f"  -> 推定 (C, H, W) 形式: C={count}, H={height}, W={width}")
                        else:
                            height, width, count = dim0, dim1, dim2
                            img_array = np.transpose(img_data, (2, 0, 1))
                            logger.info(f"  -> 推定 (H, W, C) から (C, H, W) に変換: H={height}, W={width}, C={count}")
                elif len(img_data.shape) == 2:
                    # グレースケール画像の場合
                    height, width, count = img_data.shape[0], img_data.shape[1], 1
                    img_array = img_data[np.newaxis, :, :]  # (H, W) -> (1, H, W)
                    logger.info(f"  -> グレースケール画像: H={height}, W={width}")
                else:
                    logger.error(f"サポートされていない画像形状: {img_data.shape}")
                    continue
                
                logger.info(f"タイル {i}: 変換後の形状 = {img_array.shape} (count={count}, height={height}, width={width})")
                
                # バウンディングボックスから地理参照情報を計算
                # bbox は [min_x, min_y, max_x, max_y] の順序
                min_x, min_y, max_x, max_y = bbox
                transform = from_bounds(min_x, min_y, max_x, max_y, width, height)
                
                # データ型を確認・調整
                if img_array.dtype == np.float64:
                    img_array = img_array.astype(np.float32)
                elif img_array.dtype not in [np.uint8, np.uint16, np.int16, np.int32, np.float32]:
                    logger.warning(f"予期しないデータ型 {img_array.dtype}、float32に変換します")
                    img_array = img_array.astype(np.float32)
                
                # 一時ファイルに保存
                with rasterio.open(
                    temp_file, 'w',
                    driver='GTiff',
                    height=height,
                    width=width,
                    count=count,
                    dtype=img_array.dtype,
                    crs='EPSG:4326',
                    transform=transform,
                    compress='lzw'  # 圧縮を追加
                ) as dst:
                    if count == 1:
                        dst.write(img_array[0], 1)
                    else:
                        for band in range(count):
                            dst.write(img_array[band], band + 1)
                
                # データセットを開く
                dataset = rasterio.open(temp_file)
                datasets.append(dataset)
                
                logger.info(f"タイル {i}: 一時ファイル {temp_file} に保存完了")
            
            if datasets:
                logger.info(f"{len(datasets)}個のタイルを結合中...")
                
                # 画像を結合
                mosaic, out_trans = merge(datasets)
                
                logger.info(f"結合後のモザイク形状: {mosaic.shape}")
                
                # 結果を保存
                out_meta = datasets[0].meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": out_trans,
                    "count": mosaic.shape[0],
                    "compress": "lzw"
                })
                
                with rasterio.open(output_path, "w", **out_meta) as dest:
                    dest.write(mosaic)
                
                logger.info(f"✅ 結合された画像を保存: {output_path}")
                logger.info(f"   - サイズ: {mosaic.shape[2]} x {mosaic.shape[1]} pixels")
                logger.info(f"   - バンド数: {mosaic.shape[0]}")
                logger.info(f"   - データ型: {mosaic.dtype}")
            
        except Exception as e:
            logger.error(f"画像結合エラー: {e}")
            raise
        finally:
            # データセットを閉じる
            for dataset in datasets:
                try:
                    dataset.close()
                except:
                    pass
            
            # 一時ファイルを削除
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass
    
    def download(self, start_date: str, end_date: str, bbox_coords: List[float], 
                evalscript: str, output_path: str, resolution: int = 20, image_type="S2L2A"):
        """
        Sentinel-2画像をダウンロード
        
        Args:
            start_date: 開始日 (YYYY-MM-DD)
            end_date: 終了日 (YYYY-MM-DD)
            bbox_coords: バウンディングボックス座標 [min_lon, min_lat, max_lon, max_lat]
            evalscript: 評価スクリプト
            output_path: 出力ファイルパス
            resolution: 解像度（メートル）
        """
        # 設定をセットアップ
        self.setup_config()
        
        # バウンディングボックスを作成
        bbox = BBox(bbox=bbox_coords, crs=CRS.WGS84)
        
        # タイルに分割
        tiles = self.calculate_tiles(bbox, resolution)
        
        # 各タイルをダウンロード
        tile_images = []
        for i, tile_bbox in enumerate(tqdm(tiles, desc="タイルをダウンロード中")):
            logger.info(f"タイル {i+1}/{len(tiles)} をダウンロード中...")
            img_data = self.download_single_image(tile_bbox, start_date, end_date, evalscript, resolution, image_type)
            if img_data is not None:
                tile_images.append((img_data, tile_bbox))
        
        # タイルを結合
        if tile_images:
            self.merge_tiles(tile_images, output_path, resolution)
        else:
            logger.error("ダウンロードされた画像がありません")

def get_default_evalscript():
    """デフォルトの評価スクリプトを返す"""
    return """
    //VERSION=3
    function setup() {
        return {
            input: [{
                bands: ["B02", "B03", "B04", "B08", "B11", "B12"],
                units: "DN"
            }],
            output: {
                bands: 6,
                sampleType: "INT16"
            }
        };
    }

    function evaluatePixel(sample) {
        return [sample.B02, sample.B03, sample.B04, sample.B08, sample.B11, sample.B12];
    }
    """

def parse_date_range(start_date: str, end_date: Optional[str] = None) -> Tuple[str, str]:
    """
    日付範囲を解析
    
    Args:
        start_date: 開始日
        end_date: 終了日（Noneの場合は開始日と同じ）
    
    Returns:
        (開始日, 終了日)のタプル
    """
    if end_date is None:
        end_date = start_date
    
    # 日付フォーマットを検証
    try:
        datetime.strptime(start_date, '%Y-%m-%d')
        datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError as e:
        raise ValueError(f"日付フォーマットエラー (YYYY-MM-DD形式で入力してください): {e}")
    
    return start_date, end_date

def main():
    parser = argparse.ArgumentParser(description='Sentinel-2画像自動ダウンロードプログラム')
    
    # 必須引数
    parser.add_argument('--start-date', required=True, 
                       help='開始日 (YYYY-MM-DD形式)')
    parser.add_argument('--bbox', nargs=4, type=float, required=True, metavar=('MIN_LON', 'MIN_LAT', 'MAX_LON', 'MAX_LAT'),
                       help='バウンディングボックス座標 (WGS84, 緯度経度)')
    parser.add_argument('--output', required=True,
                       help='出力ファイルパス (GeoTIFF)')
    parser.add_argument('--client-id', required=True,
                       help='Copernicus Data Space Ecosystem のクライアントID')
    parser.add_argument('--client-secret', required=True, 
                       help='Copernicus Data Space Ecosystem のクライアントシークレット')
    parser.add_argument('--type', required=False, default="S2L2A",
                       help='画像タイプ（S2L1CまたはS2L2A）。デフォルトはS2L2A。')
    
    # オプション引数
    parser.add_argument('--end-date',
                       help='終了日 (YYYY-MM-DD形式、指定しない場合は開始日と同じ)')
    parser.add_argument('--evalscript',
                       help='評価スクリプト（ファイルパスまたは直接スクリプト）')
    parser.add_argument('--resolution', type=int, default=20,
                       help='解像度（メートル、デフォルト: 20）')
    
    args = parser.parse_args()
    
    try:
        # 日付範囲を解析
        start_date, end_date = parse_date_range(args.start_date, args.end_date)
        
        # 評価スクリプトを取得
        if args.evalscript:
            if os.path.isfile(args.evalscript):
                with open(args.evalscript, 'r') as f:
                    evalscript = f.read()
            else:
                evalscript = args.evalscript
        else:
            evalscript = get_default_evalscript()
        
        # ダウンローダーを初期化
        downloader = Sentinel2Downloader(args.client_id, args.client_secret)
        
        # ダウンロード実行
        logger.info(f"ダウンロード開始:")
        logger.info(f"  期間: {start_date} から {end_date}")
        logger.info(f"  範囲: {args.bbox}")
        logger.info(f"  出力: {args.output}")
        logger.info(f"  解像度: {args.resolution}m")
        logger.info(f"  画像タイプ: {args.type}")
        
        downloader.download(start_date, end_date, args.bbox, evalscript, args.output, args.resolution, args.type)
        
        logger.info("✅ ダウンロード完了!")
        
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
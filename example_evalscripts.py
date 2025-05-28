#!/usr/bin/env python3
"""
Sentinel-2画像処理用の評価スクリプト例集
"""

def get_rgb_evalscript():
    """
    RGB画像用の評価スクリプト
    バンド: B04 (Red), B03 (Green), B02 (Blue)
    """
    return """
    //VERSION=3
    function setup() {
        return {
            input: [{
                bands: ["B02", "B03", "B04"],
                units: "DN"
            }],
            output: {
                bands: 3,
                sampleType: "INT16"
            }
        };
    }

    function evaluatePixel(sample) {
        return [sample.B04, sample.B03, sample.B02];  // R, G, B
    }
    """

def get_ndvi_evalscript():
    """
    NDVI計算用の評価スクリプト
    バンド: B04 (Red), B08 (NIR)
    """
    return """
    //VERSION=3
    function setup() {
        return {
            input: [{
                bands: ["B04", "B08"],
                units: "DN"
            }],
            output: {
                bands: 1,
                sampleType: "FLOAT32"
            }
        };
    }

    function evaluatePixel(sample) {
        let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);
        return [ndvi];
    }
    """

def get_snow_detection_evalscript():
    """
    雪検出用の評価スクリプト
    Normalized Difference Snow Index (NDSI)を使用
    バンド: B03 (Green), B11 (SWIR1)
    """
    return """
    //VERSION=3
    function setup() {
        return {
            input: [{
                bands: ["B02", "B03", "B04", "B08", "B11"],
                units: "DN"
            }],
            output: {
                bands: 4,
                sampleType: "FLOAT32"
            }
        };
    }

    function evaluatePixel(sample) {
        // NDSI計算
        let ndsi = (sample.B03 - sample.B11) / (sample.B03 + sample.B11);
        
        // 雪の閾値判定（NDSI > 0.4 かつ NIR > 0.11）
        let snow_mask = (ndsi > 0.4 && sample.B08 > 0.11) ? 1.0 : 0.0;
        
        return [ndsi, snow_mask, sample.B08, sample.B03];  // NDSI, Snow mask, NIR, Green
    }
    """

def get_bands_for_ndsi_evalscript():
    """
    NDSI計算用の評価スクリプト
    バンド: B03 (Green), B11 (SWIR1)
    """
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
        return [
            sample.B02, sample.B03, sample.B04, sample.B08, sample.B11, sample.B12
        ];
    }
    """


def get_cloud_mask_evalscript():
    """
    雲マスク付きRGB画像用の評価スクリプト
    Scene Classification Layer (SCL)を使用
    """
    return """
    //VERSION=3
    function setup() {
        return {
            input: [{
                bands: ["B02", "B03", "B04", "SCL"],
                units: "DN"
            }],
            output: {
                bands: 4,
                sampleType: "INT16"
            }
        };
    }

    function evaluatePixel(sample) {
        // 雲や雲の影をマスク（SCL値: 3=雲の影, 8=雲（中）, 9=雲（高）, 10=薄い雲）
        let cloud_mask = (sample.SCL == 3 || sample.SCL == 8 || sample.SCL == 9 || sample.SCL == 10) ? 0 : 1;
        
        return [sample.B04, sample.B03, sample.B02, cloud_mask];  // R, G, B, Cloud mask
    }
    """

def get_all_bands_evalscript():
    """
    全バンド取得用の評価スクリプト
    """
    return """
    //VERSION=3
    function setup() {
        return {
            input: [{
                bands: ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"],
                units: "DN"
            }],
            output: {
                bands: 12,
                sampleType: "INT16"
            }
        };
    }

    function evaluatePixel(sample) {
        return [
            sample.B01, sample.B02, sample.B03, sample.B04,
            sample.B05, sample.B06, sample.B07, sample.B08,
            sample.B8A, sample.B09, sample.B11, sample.B12
        ];
    }
    """

def get_water_detection_evalscript():
    """
    水域検出用の評価スクリプト
    NDWI (Normalized Difference Water Index)を使用
    """
    return """
    //VERSION=3
    function setup() {
        return {
            input: [{
                bands: ["B03", "B08"],
                units: "DN"
            }],
            output: {
                bands: 2,
                sampleType: "FLOAT32"
            }
        };
    }

    function evaluatePixel(sample) {
        // NDWI計算
        let ndwi = (sample.B03 - sample.B08) / (sample.B03 + sample.B08);
        
        // 水域の閾値判定（NDWI > 0.3）
        let water_mask = ndwi > 0.3 ? 1.0 : 0.0;
        
        return [ndwi, water_mask];
    }
    """

# 使用例
if __name__ == "__main__":
    print("利用可能な評価スクリプト:")
    print("1. RGB画像: get_rgb_evalscript()")
    print("2. NDVI: get_ndvi_evalscript()")
    print("3. 雪検出: get_snow_detection_evalscript()")
    print("4. 雲マスク付きRGB: get_cloud_mask_evalscript()")
    print("5. 全バンド: get_all_bands_evalscript()")
    print("6. 水域検出: get_water_detection_evalscript()")
    
    # 例: 雪検出スクリプトを保存
    with open("snow_detection.js", "w") as f:
        f.write(get_snow_detection_evalscript())
    print("\n雪検出スクリプトを 'snow_detection.js' に保存しました") 
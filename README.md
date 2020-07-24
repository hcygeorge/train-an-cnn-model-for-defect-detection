# AOI影像瑕疵辨識

## 專案介紹

本專案目的為藉由AOI影像訓練深度學習模型辨識產品表面瑕疵，使用框架為Pytorch。實作結果顯示，預訓練VGG16模型的測試準確已達到99.0%。(目前排行榜上最高分為99.9%)
若使用更新的模型架構(如ResNet、DenseNet)，相信能進一步提升測試準確率，有心追求更高名次的人可自行嘗試。

## 軟硬體配置

- CPU： Intel-8700 (6-cores)
- GPU： GTX1080Ti (11 GB)
- Memory： 64 GB
- OS： Window10

- Pytorch 1.3
- Numpy 1.16.2

## 影像資料來源

本次影像資料是由工研院電光所在Aidea(人工智慧共創平台)釋出作為開放性議題，提供參賽者建立瑕疵辨識模型。但基於保密原則，平台並未透漏影像資料源自何種產線和產品。

資料來源：https://aidea-web.tw/topic/a49e3f76-69c9-4a4a-bcfc-c882840b3f27


## 影像資料基本資訊

1. 訓練資料： 2,528張(隨機抽取20%作為驗證資料)
2. 測試資料：10,142張
3. 影像類別：6 個類別(正常類別 + 5種瑕疵類別)
4. 影像尺寸：512x512

## 瑕疵分類

如下圖所示，除了Normal外，其餘皆屬於瑕疵影像，可觀察到Horizontal Defect外觀與Normal較為接近，可能是相對難以辨識的瑕疵種類。


<img src="https://github.com/hcygeorge/aoi_defect_detection/blob/master/aoi_example.png" alt="alt text" width="360" height="300">


## 影像前處理

- 影像隨機水平翻轉(p=0.5)
- 影像隨機旋轉正負 15 度
- 影像大小縮放成 224 x 224

## 模型

1. LeNet5
2. VGG16
3. Pretrained VGG16

## 成果

下表為建模結果，可看出以預訓練VGG16輸入AOI影像訓練後的辨識結果最佳，對10,142張測試資料的準確度(Accuracy)已達到99%。測試資料的準確度是將預測結果上傳Aidea平台，由Aidea平台評分而得。  



| 模型結構           | 訓練準確率 | 驗證準確率 | 測試準確率 |
| ------------------ | ---------: | ---------: | ---------: |
| LeNet5             |      97.7% |      94.4% |      94.9% |
| VGG16              |     100.0% |      98.4% |      98.2% |
| VGG16 (pretrained) |     100.0% |      99.8% |      99.0% |



## 參考

Aidea-AOI瑕疵分類  
https://aidea-web.tw/topic/a49e3f76-69c9-4a4a-bcfc-c882840b3f27  
LeNet5文獻  
http://yann.lecun.com/exdb/lenet/   
VGG文獻  
https://arxiv.org/abs/1409.1556  

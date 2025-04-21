# Turkish Lira Classifier

A deep learning project that classifies Turkish Lira banknotes and credit cards using PyTorch.

[Türkçe README](#türk-lirası-sınıflandırıcı)

## Overview

This project uses a Convolutional Neural Network (CNN) to classify images of Turkish Lira banknotes (1, 5, 10, 20, 50, 100, 200 TL) and credit cards. It includes a training module, a prediction script, and a GUI application for real-time classification using a webcam or uploaded images.

##
[Project Explanation Video](https://youtu.be/MPXDiy37R2U)


## Features

- Classification of 7 different Turkish Lira denominations and credit cards
- GUI application with webcam support
- Image augmentation for improved model robustness
- Detailed metrics and confusion matrix visualization
- Class-weighted loss to handle imbalanced datasets

## Requirements

- Python 3.6+
- PyTorch
- OpenCV
- See `requirements.txt` for complete dependencies

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/turkish-lira-classifier.git
cd turkish-lira-classifier
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare your dataset in the following structure:
```
dataset/
├── 1lira/
├── 5lira/
├── 10lira/
├── 20lira/
├── 50lira/
├── 100lira/
├── 200lira/
└── kredi_karti/
```

## Usage

### Training

To train the model from scratch:

```bash
python train.py
```

This will:
- Load and preprocess the dataset
- Train a CNN model with data augmentation
- Save the best model as `turkish_lira_classifier_pytorch.pt`
- Generate performance metrics in the `metrics/` directory

### Prediction

For simple prediction on an image:

```bash
python predict/predict.py --image path/to/image.jpg
```

### GUI Application

To use the graphical interface with webcam support:

```bash
python predict/predict_app.py
```

The GUI allows you to:
- Upload images for classification
- Use your webcam to capture and classify in real-time
- See confidence scores for predictions

## Model Architecture

The CNN model includes:
- 5 convolutional blocks (each with Conv2D, ReLU, BatchNorm, and MaxPool)
- Fully connected layers with dropout for regularization
- Class-weighted cross-entropy loss to handle class imbalance

## Project Structure

```
.
├── dataset/           # Training and validation data
├── metrics/           # Generated metrics and visualizations
├── predict/           # Prediction scripts
│   ├── predict_app.py # GUI application
│   └── predict.py     # Command-line prediction script
├── train.py           # Model training script
└── turkish_lira_classifier_pytorch.pt  # Trained model
```

---

# Türk Lirası Sınıflandırıcı

PyTorch kullanarak Türk Lirası banknotları ve kredi kartlarını sınıflandıran bir derin öğrenme projesi.

## Genel Bakış

Bu proje, Türk Lirası banknotlarını (1, 5, 10, 20, 50, 100, 200 TL) ve kredi kartlarını sınıflandırmak için Evrişimli Sinir Ağı (CNN) kullanır. Eğitim modülü, tahmin betiği ve web kamerası veya yüklenen görüntülerle gerçek zamanlı sınıflandırma için bir GUI uygulaması içerir.

## Özellikler

- 7 farklı Türk Lirası banknotu ve kredi kartı sınıflandırması
- Web kamerası desteği ile GUI uygulaması
- İyileştirilmiş model sağlamlığı için görüntü artırma (augmentation)
- Ayrıntılı metrikler ve karmaşıklık matrisi görselleştirme
- Dengesiz veri setlerini işlemek için sınıf ağırlıklı kayıp fonksiyonu

## Gereksinimler

- Python 3.6+
- PyTorch
- OpenCV
- Tam bağımlılıklar için `requirements.txt` dosyasına bakın

## Kurulum

1. Depoyu klonlayın:
```bash
git clone https://github.com/kullaniciadi/turkish-lira-classifier.git
cd turkish-lira-classifier
```

2. Bağımlılıkları yükleyin:
```bash
pip install -r requirements.txt
```

3. Veri setinizi aşağıdaki yapıda hazırlayın:
```
dataset/
├── 1lira/
├── 5lira/
├── 10lira/
├── 20lira/
├── 50lira/
├── 100lira/
├── 200lira/
└── kredi_karti/
```

## Kullanım

### Eğitim

Modeli sıfırdan eğitmek için:

```bash
python train.py
```

Bu işlem:
- Veri setini yükler ve ön işler
- Veri artırma ile bir CNN modeli eğitir
- En iyi modeli `turkish_lira_classifier_pytorch.pt` olarak kaydeder
- `metrics/` dizinine performans metrikleri oluşturur

### Tahmin

Bir görüntü üzerinde basit tahmin için:

```bash
python predict/predict.py --image goruntu/yolu.jpg
```

### GUI Uygulaması

Web kamerası desteği ile grafiksel arayüzü kullanmak için:

```bash
python predict/predict_app.py
```

GUI şunları yapmanıza olanak tanır:
- Sınıflandırma için görüntü yükleme
- Gerçek zamanlı yakalama ve sınıflandırma için web kameranızı kullanma
- Tahminler için güven puanlarını görme

## Model Mimarisi

CNN modeli şunları içerir:
- 5 evrişim bloğu (her biri Conv2D, ReLU, BatchNorm ve MaxPool içerir)
- Düzenlileştirme için dropout içeren tam bağlantılı katmanlar
- Sınıf dengesizliğini ele almak için sınıf ağırlıklı çapraz entropi kaybı

## Proje Yapısı

```
.
├── dataset/           # Eğitim ve doğrulama verileri
├── metrics/           # Oluşturulan metrikler ve görselleştirmeler
├── predict/           # Tahmin betikleri
│   ├── predict_app.py # GUI uygulaması
│   └── predict.py     # Komut satırı tahmin betiği
├── train.py           # Model eğitim betiği
└── turkish_lira_classifier_pytorch.pt  # Eğitilmiş model
```

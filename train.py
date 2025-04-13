import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
from tqdm import tqdm

# GPU kullanımını kontrol et
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch kullanılan cihaz: {device}")
if device.type == "cuda":
    print(f"GPU modeli: {torch.cuda.get_device_name(0)}")
    print(f"Kullanılabilir GPU sayısı: {torch.cuda.device_count()}")
    print(f"Mevcut CUDA sürümü: {torch.version.cuda}")

# Sabit değişkenler
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32
EPOCHS = 100
NUM_CLASSES = 8  # 1TL, 5TL, 10TL, 20TL, 50TL, 100TL, 200TL, Kredi Kartı
SPLIT_RATIO = 0.2  # %20 test, %80 eğitim

# Sınıf adları
class_names = ['5 lira', '1 Lira', '50 lira', '200 Lira', 'Kredi Kartı', '10 lira', '20 Lira', '100 lira']

class CurrencyCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CurrencyCNN, self).__init__()
        
        # Konvolüsyon katmanları
        self.conv_layers = nn.Sequential(
            # İlk konvolüsyon bloğu
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            
            # İkinci konvolüsyon bloğu
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            
            # Üçüncü konvolüsyon bloğu
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            
            # Dördüncü konvolüsyon bloğu
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
            
            # Beşinci konvolüsyon bloğu
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2)
        )
        
        # Tam bağlantılı katmanlar
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            
            nn.Dropout(0.5),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            
            nn.Dropout(0.3),
            
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(data_loader, desc="Eğitim"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Gradyanları sıfırla
        optimizer.zero_grad()
        
        # İleri yönlü geçiş
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Geri yayılım ve optimizasyon
        loss.backward()
        optimizer.step()
        
        # İstatistik
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate_epoch(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Doğrulama"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # İleri yönlü geçiş
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # İstatistik
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Tahminleri topla
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels

def plot_training_history(train_losses, train_accs, val_losses, val_accs):
    """Eğitim geçmişini grafik olarak gösterir"""
    plt.figure(figsize=(12, 4))
    
    # Doğruluk grafiği
    plt.subplot(1, 2, 1)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    # Kayıp grafiği
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history_pytorch.png')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Karmaşıklık matrisini grafikleştirir"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix_pytorch.png')
    plt.show()

def predict_image(model, img_path, class_names, transform):
    """Tek görüntü üzerinde tahmin yapar"""
    model.eval()
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        
    predicted_class = predicted.item()
    confidence = probabilities[0][predicted_class].item() * 100
    
    print(f"Tahmin: {class_names[predicted_class]} (Güven: %{confidence:.2f})")
    return class_names[predicted_class], confidence

def calculate_acc_per_class(model, data_loader, device, num_classes):
    """Her sınıf için doğruluk oranını hesaplar"""
    model.eval()
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            # Her sınıf için doğruluk
            for i in range(len(labels)):
                label = labels[i]
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1
    
    # Sınıf başına doğruluk
    class_accuracy = {}
    for i in range(num_classes):
        if class_total[i] > 0:
            accuracy = 100 * class_correct[i] / class_total[i]
            class_accuracy[i] = accuracy
    
    return class_accuracy

def main():
    # Veri yolu
    data_dir = 'dataset'
    
    # Veri dönüşümleri
    train_transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=0.2),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Veri setlerini yükle
    full_dataset = ImageFolder(root=data_dir, transform=train_transform)
    
    # Sınıf adlarını alıp yazdır
    class_to_idx = full_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    print("Sınıf etiketleri:", class_to_idx)
    
    # Sınıf dengesi kontrolü
    class_counts = [0] * NUM_CLASSES
    for _, label in full_dataset.samples:
        class_counts[label] += 1
    
    print("\nSınıf başına örnek sayısı:")
    for i in range(NUM_CLASSES):
        if i in idx_to_class:
            print(f"{idx_to_class[i]}: {class_counts[i]} örnek")
    
    # Eğitim ve doğrulama setlerini ayır
    train_size = int((1 - SPLIT_RATIO) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Veri yükleyicileri
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True if device.type == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True if device.type == "cuda" else False
    )
    
    print(f"Eğitim veri seti boyutu: {len(train_dataset)}")
    print(f"Doğrulama veri seti boyutu: {len(val_dataset)}")
    
    # Modeli oluştur ve cihaza taşı
    model = CurrencyCNN(num_classes=NUM_CLASSES)
    model = model.to(device)
    
    # Model özeti için param sayısını yazdır
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Toplam parametre sayısı: {total_params:,}")
    print(f"Eğitilebilir parametre sayısı: {trainable_params:,}")
    
    # Kayıp fonksiyonu ve optimize edici
    # Sınıf dengesizliğine karşı ağırlıklı kayıp fonksiyonu
    class_weights = torch.FloatTensor([1.0/max(count, 1) for count in class_counts]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        min_lr=1e-6, 
        verbose=True
    )
    
    # Eğitim için gerekli listeler
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_val_acc = 0.0
    patience = 30
    patience_counter = 0
    
    # Eğitim döngüsü
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        # Eğitim
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        print(f"Eğitim - Kayıp: {train_loss:.4f}, Doğruluk: {train_acc:.4f}")
        
        # Doğrulama
        val_loss, val_acc, val_preds, val_labels = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        print(f"Doğrulama - Kayıp: {val_loss:.4f}, Doğruluk: {val_acc:.4f}")
        
        # Öğrenme oranını güncelle
        scheduler.step(val_loss)
        
        # En iyi modeli kaydet
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'best_model_pytorch.pth')
            print(f"En iyi model kaydedildi (Doğruluk: {val_acc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Erken durdurma
        if patience_counter >= patience:
            print(f"Erken durdurma: Doğrulama doğruluğu {patience} epoch boyunca iyileşmedi.")
            break
    
    # Eğitim geçmişini görselleştir
    plot_training_history(train_losses, train_accs, val_losses, val_accs)
    
    # En iyi modeli yükle
    checkpoint = torch.load('best_model_pytorch.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Test seti üzerinde değerlendirme
    print("\nEn iyi model ile test değerlendirmesi...")
    _, final_acc, final_preds, final_labels = validate_epoch(model, val_loader, criterion, device)
    print(f"Test doğruluğu: {final_acc:.4f}")
    
    # Sınıf başına doğruluk
    class_accuracy = calculate_acc_per_class(model, val_loader, device, NUM_CLASSES)
    print("\nSınıf başına doğruluk:")
    for class_idx, accuracy in class_accuracy.items():
        class_name = idx_to_class[class_idx]
        print(f"{class_name}: {accuracy:.2f}%")
    
    # Sınıflandırma raporu
    class_labels = [idx_to_class.get(i, f"Sınıf {i}") for i in range(NUM_CLASSES)]
    print("\nSınıflandırma Raporu:")
    print(classification_report(final_labels, final_preds, target_names=class_labels))
    
    # Karmaşıklık matrisini görselleştir
    plot_confusion_matrix(final_labels, final_preds, class_labels)
    
    # Modeli script formatında kaydet (daha taşınabilir)
    model_scripted = torch.jit.script(model)
    model_scripted.save('turkish_lira_classifier_pytorch.pt')
    print("\nModel başarıyla kaydedildi: turkish_lira_classifier_pytorch.pt")
    
    # Örnek test
    print("\nÖrnek test görüntüsü üzerinde tahmin...")
    test_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                test_files.append(os.path.join(root, file))
    
    if test_files:
        sample_image = np.random.choice(test_files)
        print(f"Seçilen test görüntüsü: {sample_image}")
        predict_image(model, sample_image, class_labels, test_transform)
    
    print("\nEğitim ve değerlendirme tamamlandı!")

if __name__ == "__main__":
    main()
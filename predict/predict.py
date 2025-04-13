# Modeli yükleme ve tahmin yapma örneği
import torch
from PIL import Image
from torchvision import transforms

# Model dosyasını yükle
model = torch.jit.load('turkish_lira_classifier_pytorch.pt')
model.eval()

# Görüntü dönüşümü
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Sınıf isimleri
class_names = ['1lira', '5lira', '10lira', '20lira', '50lira', '100lira', '200lira', 'kredi_karti']

# Görüntüyü yükle ve tahmin et
def predict(image_path):
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        
    predicted_class = predicted.item()
    confidence = probabilities[0][predicted_class].item() * 100
    
    print(f"Tahmin: {class_names[predicted_class]} (Güven: %{confidence:.2f})")
    return class_names[predicted_class], confidence

# Örnek kullanım
predict('test_image.jpg')
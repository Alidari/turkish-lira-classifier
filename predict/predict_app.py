import os
import sys
import torch
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from torchvision import transforms
import traceback
import threading
import time

class TurkishLiraClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Türk Lirası Sınıflandırıcı (Hata Ayıklama)")
        self.root.geometry("800x750")  # Biraz daha büyük yaptım
        self.root.configure(bg="#f0f0f0")
        
        # Kullanılabilir cihazı belirle
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Kullanılan cihaz: {self.device}")
        
        # Model dosyasını yükle
        try:
            # Önce model dosyasının var olup olmadığını kontrol et
            if not os.path.exists('turkish_lira_classifier_pytorch.pt'):
                messagebox.showerror("Hata", "Model dosyası bulunamadı: turkish_lira_classifier_pytorch.pt")
                print("Model dosyası bulunamadı! Lütfen dosya yolunu kontrol edin.")
                self.root.destroy()
                return
                
            print("Model yükleniyor...")
            # Modeli yükle ve cihaza taşı
            self.model = torch.jit.load('turkish_lira_classifier_pytorch.pt', map_location=self.device)
            self.model.eval()
            print("Model başarıyla yüklendi!")
        except Exception as e:
            error_msg = f"Model yüklenemedi: {str(e)}"
            messagebox.showerror("Hata", error_msg)
            print(f"HATA: {error_msg}")
            print(traceback.format_exc())
            self.root.destroy()
            return
            
        # Görüntü dönüşümü
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Sınıf isimleri
        self.class_names = ['5 lira', '1 Lira', '50 lira', '200 Lira', 'Kredi Kartı', '10 lira', '20 Lira', '100 lira']
        
        # Uygulama değişkenleri
        self.current_image = None
        self.photo = None
        self.cap = None
        self.camera_active = False
        
        # Debug log
        self.debug_log = []
        
        # UI elemanlarını oluştur
        self.create_widgets()
        
        print("Uygulama başlatıldı ve hazır.")
    
    def create_widgets(self):
        # Ana çerçeve
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        # Başlık
        title_label = tk.Label(main_frame, text="Türk Lirası Tanıma Sistemi", 
                              font=("Helvetica", 22, "bold"), bg="#f0f0f0", fg="#0066cc")
        title_label.pack(pady=10)
        
        # Durum bilgisi (CPU/GPU)
        device_label = tk.Label(main_frame, 
                               text=f"Çalışma modu: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}", 
                               font=("Helvetica", 10), bg="#f0f0f0", fg="#666666")
        device_label.pack()
        
        # Görüntü çerçevesi
        image_container = tk.Frame(main_frame, bg="#d1d1d1", width=450, height=350, bd=2, relief=tk.RAISED)
        image_container.pack(pady=10)
        image_container.pack_propagate(False)  # Boyutu sabit tut
        
        self.image_frame = tk.Label(image_container, bg="#e0e0e0")
        self.image_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # İlerleme çubuğu ve durum
        self.status_var = tk.StringVar(value="Hazır")
        status_label = tk.Label(main_frame, textvariable=self.status_var, 
                               font=("Helvetica", 10), bg="#f0f0f0")
        status_label.pack(pady=5)
        
        # Buton çerçevesi
        button_frame = tk.Frame(main_frame, bg="#f0f0f0")
        button_frame.pack(pady=10)
        
        self.resim_sec_btn = tk.Button(button_frame, text="RESİM SEÇ", 
                                     command=self.open_image, width=18, height=2,
                                     bg="#4CAF50", fg="white", font=("Helvetica", 12, "bold"))
        self.resim_sec_btn.grid(row=0, column=0, padx=15, pady=10)
        
        self.kamera_btn = tk.Button(button_frame, text="KAMERA AÇ/KAPA", 
                                  command=self.toggle_camera, width=18, height=2,
                                  bg="#2196F3", fg="white", font=("Helvetica", 12, "bold"))
        self.kamera_btn.grid(row=0, column=1, padx=15, pady=10)
        
        self.cek_btn = tk.Button(button_frame, text="FOTOĞRAF ÇEK", 
                               command=self.capture_image, width=18, height=2,
                               bg="#FFC107", fg="black", font=("Helvetica", 12, "bold"),
                               state=tk.DISABLED)
        self.cek_btn.grid(row=1, column=0, padx=15, pady=10)
        
        self.siniflandir_btn = tk.Button(button_frame, text="SINIFLANDIR", 
                                       command=self.start_classification, width=18, height=2,
                                       bg="#FF5722", fg="white", font=("Helvetica", 12, "bold"),
                                       state=tk.DISABLED)
        self.siniflandir_btn.grid(row=1, column=1, padx=15, pady=10)
        
        # Sonuç çerçevesi
        result_frame = tk.Frame(main_frame, bg="#e6e6e6", bd=2, relief=tk.GROOVE)
        result_frame.pack(pady=10, fill=tk.X, padx=50)
        
        self.result_label = tk.Label(result_frame, text="Lütfen bir resim seçin veya kamera ile çekin",
                                   font=("Helvetica", 14), bg="#e6e6e6", pady=10)
        self.result_label.pack()
        
        self.confidence_label = tk.Label(result_frame, text="",
                                       font=("Helvetica", 12), bg="#e6e6e6", pady=5)
        self.confidence_label.pack(pady=5)
        
        # Hata log çerçevesi
        log_frame = tk.Frame(main_frame, bg="#ffe0e0", bd=2, relief=tk.GROOVE)
        log_frame.pack(pady=10, fill=tk.X, padx=50)
        
        log_title = tk.Label(log_frame, text="Hata Ayıklama Bilgisi", 
                            font=("Helvetica", 10, "bold"), bg="#ffe0e0")
        log_title.pack(pady=(5,0))
        
        self.log_label = tk.Label(log_frame, text="Henüz hata yok", 
                                 font=("Helvetica", 9), bg="#ffe0e0", wraplength=600)
        self.log_label.pack(pady=5)
    
    def log_debug(self, message):
        """Hata ayıklama mesajlarını kaydet ve göster"""
        print(f"DEBUG: {message}")
        timestamp = time.strftime("%H:%M:%S")
        self.debug_log.append(f"[{timestamp}] {message}")
        
        # En son 3 log mesajını göster
        if len(self.debug_log) > 3:
            display_log = self.debug_log[-3:]
        else:
            display_log = self.debug_log
        
        self.log_label.config(text="\n".join(display_log))
    
    def open_image(self):
        # Kamerayı kapat (açıksa)
        if self.camera_active:
            self.toggle_camera()
        
        # Resim seçme diyaloğunu aç
        file_path = filedialog.askopenfilename(
            title="Resim Seç",
            filetypes=[("Resim Dosyaları", "*.jpg *.jpeg *.png")]
        )
        
        if file_path:
            try:
                self.log_debug(f"Resim yükleniyor: {file_path}")
                self.current_image = Image.open(file_path).convert('RGB')
                self.display_image(self.current_image)
                self.siniflandir_btn.config(state=tk.NORMAL)
                self.result_label.config(text="Resim yüklendi. Sınıflandır butonuna tıklayın.")
                self.confidence_label.config(text="")
                self.log_debug("Resim başarıyla yüklendi")
            except Exception as e:
                error_msg = f"Resim yüklenemedi: {str(e)}"
                messagebox.showerror("Hata", error_msg)
                self.log_debug(error_msg)
    
    def toggle_camera(self):
        if self.camera_active:
            # Kamerayı kapat
            self.log_debug("Kamera kapatılıyor")
            if self.cap and self.cap.isOpened():
                self.cap.release()
            self.camera_active = False
            self.cek_btn.config(state=tk.DISABLED)
            self.kamera_btn.config(text="KAMERA AÇ/KAPA")
            self.image_frame.config(image=None)
            self.image_frame.image = None
            self.log_debug("Kamera kapatıldı")
            return
        
        # Kamerayı aç
        self.log_debug("Kamera açılıyor")
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            error_msg = "Kamera açılamıyor. Kamera bağlı ve kullanılabilir durumda mı?"
            messagebox.showerror("Hata", error_msg)
            self.log_debug(error_msg)
            return
        
        self.camera_active = True
        self.cek_btn.config(state=tk.NORMAL)
        self.kamera_btn.config(text="KAMERA KAPAT")
        self.log_debug("Kamera açıldı")
        self.update_camera()
    
    def update_camera(self):
        if self.camera_active:
            ret, frame = self.cap.read()
            if ret:
                # OpenCV'den gelen BGR formatını RGB'ye dönüştür
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                
                # Ekranda göster
                self.display_image(img)
                
                # Yeniden çağır
                self.root.after(10, self.update_camera)
            else:
                error_msg = "Kameradan görüntü alınamadı."
                messagebox.showerror("Hata", error_msg)
                self.log_debug(error_msg)
                self.toggle_camera()
    
    def capture_image(self):
        if self.camera_active and self.cap:
            self.log_debug("Fotoğraf çekiliyor")
            ret, frame = self.cap.read()
            if ret:
                # BGR'den RGB'ye dönüştür
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.current_image = Image.fromarray(frame_rgb)
                
                # Kamerayı kapat ve resmi göster
                self.toggle_camera()
                self.display_image(self.current_image)
                self.siniflandir_btn.config(state=tk.NORMAL)
                self.result_label.config(text="Fotoğraf çekildi. Sınıflandır butonuna tıklayın.")
                self.confidence_label.config(text="")
                self.log_debug("Fotoğraf çekildi")
    
    def display_image(self, img):
        # Resmi uygun boyuta getir
        display_img = img.copy()
        display_img.thumbnail((430, 330))
        
        # Tkinter PhotoImage nesnesine dönüştür
        self.photo = ImageTk.PhotoImage(display_img)
        
        # Ekranda göster
        self.image_frame.config(image=self.photo)
        self.image_frame.image = self.photo  # Garbage collection'dan koruma için referans
    
    def start_classification(self):
        """Sınıflandırma işlemini ayrı bir thread'de başlat"""
        if self.current_image is None:
            messagebox.showwarning("Uyarı", "Lütfen önce bir resim seçin veya çekin.")
            return
        
        # Butonları devre dışı bırak
        self.siniflandir_btn.config(state=tk.DISABLED)
        self.resim_sec_btn.config(state=tk.DISABLED)
        self.kamera_btn.config(state=tk.DISABLED)
        
        # Durum bilgisini güncelle
        self.status_var.set("Sınıflandırılıyor...")
        self.log_debug("Sınıflandırma başlatılıyor")
        
        # Sınıflandırma işlemini ayrı bir thread'de başlat
        thread = threading.Thread(target=self.classify_image)
        thread.daemon = True
        thread.start()
    
    def classify_image(self):
        """Resmi sınıflandırma işlemi (ayrı thread'de çalışır)"""
        try:
            self.log_debug("Görüntü ön işleme yapılıyor")
            # Resmi modelin beklediği formata dönüştür
            img_tensor = self.transform(self.current_image).unsqueeze(0)
            
            self.log_debug(f"Görüntü tensör şekli: {img_tensor.shape}")
            self.log_debug(f"Tensör aygıta taşınıyor: {self.device}")
            # Tensörü uygun cihaza taşı
            img_tensor = img_tensor.to(self.device)
            
            # Tahmin yap
            self.log_debug("Model tahmin işlemi yapılıyor")
            with torch.no_grad():
                # Modelin giriş ve çıkış şekillerini kontrol et
                self.log_debug(f"Model girişi şekli: {img_tensor.shape}")
                
                # Her adımda ne olduğunu kontrol et
                outputs = self.model(img_tensor)
                self.log_debug(f"Model çıktısı şekli: {outputs.shape}")
                
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                self.log_debug(f"Olasılıklar hesaplandı")
                
                _, predicted = torch.max(outputs, 1)
                self.log_debug(f"En yüksek olasılıklı sınıf belirlendi")
            
            # Sonuçları al
            predicted_class = predicted.item()
            confidence = probabilities[0][predicted_class].item() * 100
            
            self.log_debug(f"Tahmin: {self.class_names[predicted_class]}, Güven: %{confidence:.2f}")
            
            # Sonuçları göster (ana thread'e geri dön)
            self.root.after(0, self.update_results, predicted_class, confidence)
            
        except Exception as e:
            error_msg = f"Sınıflandırma sırasında bir hata oluştu: {str(e)}"
            self.log_debug(error_msg)
            self.log_debug(traceback.format_exc())
            
            # Hata mesajını göster (ana thread'e geri dön)
            self.root.after(0, self.show_error, error_msg)
    
    def update_results(self, predicted_class, confidence):
        """Tahmin sonuçlarını güncelle (ana thread'de çağrılır)"""
        # Sonuçları göster
        self.result_label.config(text=f"Tahmin: {self.class_names[predicted_class]}")
        self.confidence_label.config(text=f"Güven: %{confidence:.2f}")
        
        # Durum bilgisini güncelle
        self.status_var.set("Sınıflandırma tamamlandı")
        
        # Butonları tekrar aktif et
        self.siniflandir_btn.config(state=tk.NORMAL)
        self.resim_sec_btn.config(state=tk.NORMAL)
        self.kamera_btn.config(state=tk.NORMAL)
    
    def show_error(self, error_msg):
        """Hata mesajını göster (ana thread'de çağrılır)"""
        messagebox.showerror("Hata", error_msg)
        
        # Durum bilgisini güncelle
        self.status_var.set("Hata oluştu")
        
        # Butonları tekrar aktif et
        self.siniflandir_btn.config(state=tk.NORMAL)
        self.resim_sec_btn.config(state=tk.NORMAL)
        self.kamera_btn.config(state=tk.NORMAL)
    
    def __del__(self):
        # Uygulama kapanırken kamerayı kapat
        if self.cap and self.cap.isOpened():
            self.cap.release()

# Manuel model sınıflandırma fonksiyonu (UI dışında test etmek için)
def test_model_directly(image_path):
    """Model dosyasını doğrudan test et"""
    print("\n----- MODEL TEST -----")
    print(f"Test görüntüsü: {image_path}")
    
    try:
        # Cihaz kontrolü
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Kullanılan cihaz: {device}")
        
        # Model dosyasını yükle
        print("Model yükleniyor...")
        model = torch.jit.load('turkish_lira_classifier_pytorch.pt', map_location=device)
        model.eval()
        print("Model başarıyla yüklendi!")
        
        # Görüntü ön işleme
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Görüntüyü yükle
        print(f"Görüntü yükleniyor: {image_path}")
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        img_tensor = img_tensor.to(device)
        print(f"Görüntü tensör şekli: {img_tensor.shape}")
        
        # Tahmin yap
        print("Tahmin yapılıyor...")
        with torch.no_grad():
            outputs = model(img_tensor)
            print(f"Model çıktısı şekli: {outputs.shape}")
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
        
        # Sonuçları yazdır
        # class_names = ['1 Lira', '5 Lira', '10 Lira', '20 Lira', '50 Lira', '100 Lira', '200 Lira', 'Kredi Kartı']
        class_names = ['5 lira', '1 Lira', '50 lira', '200 Lira', 'Kredi Kartı', '10 lira', '20 Lira', '100 lira']
        predicted_class = predicted.item()
        confidence = probabilities[0][predicted_class].item() * 100
        
        print(f"Tahmin: {class_names[predicted_class]}")
        print(f"Güven: %{confidence:.2f}")
        print("----- TEST TAMAMLANDI -----\n")
        
        return class_names[predicted_class], confidence
        
    except Exception as e:
        print(f"TEST BAŞARISIZ: {str(e)}")
        print(traceback.format_exc())
        return None, None

if __name__ == "__main__":
    # Eğer komut satırı argümanı varsa, modeli doğrudan test et
    if len(sys.argv) > 1 and sys.argv[1] == "--test" and len(sys.argv) > 2:
        test_image_path = sys.argv[2]
        test_model_directly(test_image_path)
    else:
        # Normal uygulama başlat
        root = tk.Tk()
        app = TurkishLiraClassifierApp(root)
        root.mainloop()
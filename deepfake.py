import os
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from deepface import DeepFace
import random
from time import sleep
from datetime import datetime
from termcolor import colored
import logging
from rich.console import Console
from rich.table import Table
from rich.progress import track, Progress
from fpdf import FPDF
import plotly.express as px
from threading import Thread
from multiprocessing import Pool
import pandas as pd
import tensorflow as tf


# إعداد تسجيل الأحداث (Logging)
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# إعداد واجهة المستخدم باستخدام Rich
console = Console()

# دعم اللغات المتعددة
LANGUAGES = {
    "en": {
        "menu_title": "🌟 Face and Image Analysis System 🌟",
        "camera_analysis": "🎥 Start Face Analysis from Camera",
        "image_analysis": "📷 Analyze Images from Dataset",
        "exit": "🚪 Exit",
        "select_option": "Select an option (1/2/3): ",
        "invalid_choice": "Invalid choice. Please enter 1, 2, or 3.",
        "starting_camera": "Starting camera...",
        "camera_failed": "Failed to open camera.",
        "camera_active": "Camera is active. Press 'q' to quit.",
        "analysis_failed": "Analysis failed: {}",
        "total_images": "🖼️ Total images analyzed: {}",
        "real_images": "✅ Real images: {}",
        "fake_images": "❌ Fake images: {}",
        "emotion_distribution": "===== Emotion Distribution =====",
        "report_generated": "Report generated: {}",
    },
    "tr": {
        "menu_title": "🌟 Yüz ve Görüntü Analizi Sistemi 🌟",
        "camera_analysis": "🎥 Kameradan Yüz Analizi Başlat",
        "image_analysis": "📷 Veri Setinden Görüntü Analizi Yap",
        "exit": "❌ Çıkış",
        "select_option": "Seçiminizi yapın (1/2/3): ",
        "invalid_choice": "Geçersiz seçim. Lütfen 1, 2 veya 3 girin.",
        "starting_camera": "Kamerayı başlatıyor...",
        "camera_failed": "Kamera açılamadı.",
        "camera_active": "Kamera aktif. Çıkmak için 'q' tuşuna basın.",
        "analysis_failed": "Analiz başarısız: {}",
        "total_images": "🖼️ Analiz edilen toplam görüntü sayısı: {}",
        "real_images": "✅ Gerçek görüntü sayısı: {}",
        "fake_images": "❌ Sahte görüntü sayısı: {}",
        "emotion_distribution": "===== Duyguların Dağılımı =====",
        "report_generated": "Rapor oluşturuldu: {}",
    }
}

# اللغة الافتراضية
LANGUAGE = "tr"

def t(key):
    """ترجمة النص بناءً على اللغة المحددة."""
    return LANGUAGES[LANGUAGE].get(key, key)

# تحسين واجهة المستخدم
def print_menu():
    """طباعة قائمة الخيارات."""
    console.print("=" * 50, style="bold cyan")
    console.print(t("menu_title"), style="bold cyan")
    console.print("=" * 50, style="bold cyan")
    console.print(f"1. {t('camera_analysis')}", style="green")
    console.print(f"2. {t('image_analysis')}", style="yellow")
    console.print(f"3. {t('exit')}", style="red")
    console.print("=" * 50, style="bold cyan")

# اختيار المستخدم
def user_input():
    """الحصول على اختيار المستخدم."""
    print_menu()
    secim = console.input(f"[blue]{t('select_option')}[/blue]")
    
    if secim == "1":
        return "camera"
    elif secim == "2":
        return "image_analysis"
    elif secim == "3":
        return "exit"
    else:
        console.print(f"[red]{t('invalid_choice')}[/red]")
        return user_input()

# تحليل الكاميرا
def camera_analysis():
    """تحليل الوجوه من الكاميرا."""
    console.print(f"[cyan]{t('starting_camera')}[/cyan]")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        console.print(f"[red]{t('camera_failed')}[/red]")
        logging.error(t("camera_failed"))
    else:
        console.print(f"[green]{t('camera_active')}[/green]")
        emotions_count = {'happy': 0, 'sad': 0, 'angry': 0, 'surprise': 0, 'fear': 0, 'neutral': 0}
        
        while True:
            ret, frame = cap.read()
            if not ret:
                console.print("[red]Kamera görüntüsü alınamıyor.[/red]")
                logging.error("Kamera görüntüsü alınamıyor.")
                break
            
            try:
                analysis = DeepFace.analyze(frame, actions=['emotion', 'age', 'gender', 'race'], enforce_detection=False)
                emotion = analysis[0]['dominant_emotion']
                age = analysis[0]['age']
                gender = analysis[0]['dominant_gender']
                race = analysis[0]['dominant_race']
                
                emotions_count[emotion] += 1
                
                cv2.putText(frame, f"Duygu: {emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Yaş: {age}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Cinsiyet: {gender}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Irk: {race}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            except Exception as e:
                console.print(f"[red]{t('analysis_failed').format(e)}[/red]")
                logging.error(f"Analiz başarısız: {e}")
            
            cv2.imshow("Kamera Görüntüsü - Analiz", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # إنشاء التقرير
        generate_report(emotions_count, source="camera")

# تحليل الصور
def image_analysis():
    """تحليل الصور من مجموعة البيانات."""
    console.print("[cyan]Görüntü analiz kodunu çalıştırıyor...[/cyan]")
    
    input_shape = (170, 170, 3)
    data_dir = r'.\Data'  # Veri klasörünün yolu

    real_data = [f for f in os.listdir(os.path.join(data_dir, 'training_real')) if f.endswith('.jpg')]
    fake_data = [f for f in os.listdir(os.path.join(data_dir, 'training_fake')) if f.endswith('.jpg')]

    X = []
    Y = []

    for img in track(real_data, description="[green]Gerçek görüntüler yükleniyor...[/green]"):
        img_path = os.path.join(data_dir, 'training_real', img)
        img = load_img(img_path, target_size=(170, 170))
        X.append(img_to_array(img) / 255.0)
        Y.append(1)

    for img in track(fake_data, description="[red]Sahte görüntüler yükleniyor...[/red]"):
        img_path = os.path.join(data_dir, 'training_fake', img)
        img = load_img(img_path, target_size=(170, 170))
        X.append(img_to_array(img) / 255.0)
        Y.append(0)

    X = np.array(X)
    Y = to_categorical(Y, 2)

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=5)

    emotions_count = {'happy': 0, 'sad': 0, 'angry': 0, 'surprise': 0, 'fear': 0, 'neutral': 0}

    num_samples = 5 
    random_indices = random.sample(range(len(X_val)), num_samples)

    real_count = 0
    fake_count = 0

    for i in random_indices:
        plt.figure(figsize=(8, 6))
        plt.imshow(X_val[i])
        plt.axis('off')
        
        label = "Gerçek" if np.argmax(Y_val[i]) == 1 else "Sahte"
        plt.title(f"{label}", fontsize=16, color='darkgreen')

        if label == "Gerçek":
            real_count += 1
            try:
                rgb_image = cv2.cvtColor((X_val[i] * 255).astype("uint8"), cv2.COLOR_BGR2RGB)
                analysis = DeepFace.analyze(rgb_image, actions=['emotion'], enforce_detection=False)
                if isinstance(analysis, list):
                    analysis = analysis[0]
                emotion = analysis.get('dominant_emotion', "Bilinmiyor")
                emotions_count[emotion] += 1
            except Exception as e:
                emotion = "Bilinmiyor"
            
            plt.gca().text(0.5, -0.1, f"Duygular: {emotion}", ha='center', va='top', fontsize=14, color='blue', transform=plt.gca().transAxes)
        else:
            fake_count += 1
            plt.gca().text(0.5, -0.1, "Duygular: -", ha='center', va='top', fontsize=14, color='gray', transform=plt.gca().transAxes)
        
        plt.show()
        sleep(1)
    console.print(f"\n[bold cyan]{t('total_images').format(num_samples)}[/bold cyan]")
    console.print(f"✅ [green]{t('real_images').format(real_count)}[/green]")
    console.print(f"❌ [red]{t('fake_images').format(fake_count)}[/red]")

    console.print(f"\n[bold cyan]{t('emotion_distribution')}[/bold cyan]")
    for emotion, count in emotions_count.items():
        console.print(f"{emotion.capitalize()} duygusu sayısı: {count}")

    # إنشاء التقرير
    generate_report(emotions_count, source="image_analysis", real_images_count=real_count, fake_images_count=fake_count)

# إنشاء تقرير PDF
def generate_pdf_report(emotions_count, source, real_images_count=None, fake_images_count=None):
    """إنشاء تقرير PDF من نتائج التحليل."""
    pdf = FPDF()
    pdf.add_page()
    
    # إضافة خط يدعم Unicode (مثل DejaVuSans)
    pdf.add_font('Monomakh', '', './Fonts/Monomakh/Monomakh-Regular.ttf', uni=True)
    pdf.set_font('Monomakh', size=12)
    
    pdf.cell(200, 10, txt="===== Analiz Raporu =====", ln=True, align='C')
    pdf.cell(200, 10, txt=f"Rapor Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
    
    if source == "camera":
        pdf.cell(200, 10, txt="1. Kameradan Yüz Analizi:", ln=True)
        pdf.cell(200, 10, txt="   - Duygu, yaş ve cinsiyet analizi yapıldı.", ln=True)
    elif source == "image_analysis":
        pdf.cell(200, 10, txt="1. Veri Setinden Görüntü Analizi:", ln=True)
        pdf.cell(200, 10, txt=f"   - Analiz edilen toplam görüntü sayısı: {real_images_count + fake_images_count}", ln=True)
        pdf.cell(200, 10, txt=f"   - Gerçek görüntü sayısı: {real_images_count}", ln=True)
        pdf.cell(200, 10, txt=f"   - Sahte görüntü sayısı: {fake_images_count}", ln=True)
        pdf.cell(200, 10, txt="______________________________________________", ln=True)
        pdf.cell(200, 10, txt="\n2. Duyguların Dağılımı:", ln=True)
    for emotion, count in emotions_count.items():
        pdf.cell(200, 10, txt=f"   - {emotion.capitalize()}: {count}", ln=True)
    
    report_dir = "Reports"
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    
    report_name = f"report_{datetime.now().strftime('%Y-%m-%d')}.pdf"
    report_path = os.path.join(report_dir, report_name)
    pdf.output(report_path)
    console.print(f"[green]{t('report_generated').format(report_path)}[/green]")

# إنشاء تقرير مفصل
def generate_report(emotions_count, source, real_images_count=None, fake_images_count=None):
    """إنشاء تقرير مفصل من نتائج التحليل."""
    generate_pdf_report(emotions_count, source, real_images_count, fake_images_count)
    
    # إنشاء الرسم البياني باستخدام Plotly
    if source == "image_analysis":
         # مخطط توزيع المشاعر
        fig1 = px.bar(x=list(emotions_count.keys()), y=list(emotions_count.values()), 
             labels={'x': 'Duygu', 'y': 'Sayı'}, 
             title="Duyguların Dağılımı", 
             color=list(emotions_count.keys()),
             text=list(emotions_count.values())) 
        fig1.update_traces(textposition='outside')
        fig1.show()
        # مخطط توزيع الصور الحقيقية والمزيفة
        fig2 = px.pie(names=['Gerçek Görüntüler', 'Sahte Görüntüler'], 
                     values=[real_images_count, fake_images_count], 
                     title="Görüntü Dağılımı: Gerçek vs Sahte",
                     color=['Gerçek Görüntüler', 'Sahte Görüntüler'],
                     color_discrete_map={'Gerçek Görüntüler': 'green', 'Sahte Görüntüler': 'red'})
        fig2.show()

# الدالة الرئيسية
def main():
    """الدالة الرئيسية لتشغيل النظام."""
    while True:
        choice = user_input()
        
        if choice == "camera":
            camera_analysis()
        elif choice == "image_analysis":
            image_analysis()
        elif choice == "exit":
            console.print("[red]Çıkıyor... Teşekkür ederiz! 👋[/red]", style="bold")
            break  

if __name__ == "__main__":
    main()
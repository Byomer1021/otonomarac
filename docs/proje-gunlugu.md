# Proje Günlüğü

Ön rapordaki plandan sapmalar ve gerekçeleri buraya yazılır. Hafta 8'deki
"hata analizi" bölümünün ham malzemesi bu dosya olacak — o hafta geriye dönüp
hatırlamaya çalışmak yerine, kararlar alındıkları anda kaydediliyor.

---

## Hafta 1 — İskelet ve tespit

**Tarih:** 15 Ağustos 2026

### Yapılanlar

- Depo iskeleti: `src/` düzeni, `pyproject.toml`, YAML config, CLI girişi
- `video_io.py` — kare okuma/yazma, kare atlama, ölçekleme
- `detection.py` — Ultralytics YOLO sarmalayıcı, `Detection` dataclass'ı
- `visualize.py` — kutu/etiket çizimi, yarı saydam HUD
- `pipeline.py` — katmanların birleştiği yer + katman bazlı süre ölçümü
- `scripts/kitti_to_video.py` — KITTI raw zip → mp4

### Plandan sapma 1: donanım varsayımı

**Rapor ne diyordu:** "Kişisel donanım bulunmadığından tamamen bulut tabanlı bir
düzen kurulacaktır" — Colab birincil, Kaggle yedek.

**Gerçek durum:** Geliştirme makinesinde **NVIDIA GTX 1080 (8 GB)** var.

**Yeni karar:** Geliştirme yerel GPU'da yapılacak. Colab birincil ortam olmaktan
çıkıp yedeğe düşüyor.

**Neden önemli:** Colab'ın oturum kopması riskler tablosunda "Orta" etkiyle
listelenmişti; yerel geliştirme bu riski tamamen ortadan kaldırıyor. Ayrıca
iterasyon hızı belirgin biçimde artıyor — her denemede Drive'a bağlanıp veriyi
yeniden yüklemek gerekmiyor.

**Getirdiği kısıt:** GTX 1080 Pascal mimarisi (sm_61) ve makinedeki sürücü
526.47. CUDA 12.x wheel'ları hem sürücü 527+ istiyor hem de yeni sürümlerde
Pascal desteği düşürüldü. Bu yüzden **cu118** yapısı kuruldu
(`torch 2.7.1+cu118`). Sürücü güncellenirse daha yeni CUDA'ya geçilebilir, ama
şu an buna ihtiyaç yok.

### Plandan sapma 2: demo videosu kaynağı

**Plan:** "Serbest sürüş videoları — demo görselleri için".

**Sorun:** Pexels ve Pixabay scripted indirmeyi Cloudflare ile kapatıyor
(403/404). Otomatik indirme mümkün olmadı.

**Karar:** Geliştirme ve geometri çalışması **KITTI** üzerinden yürüyecek
(`2011_09_26_drive_0005`, 154 kare, 10 Hz). Zaten rapor da KITTI'yi projeksiyon
çalışmasında birincil kaynak olarak işaretlemişti; kalibrasyon dosyalarının
hazır gelmesi Hafta 4 için doğrudan avantaj.

**Açık kalan:** Vitrin GIF'i için daha "güzel" görünen bir dashcam klibi hâlâ
gerekiyor — KITTI 1242x375 gibi alışılmadık bir en-boy oranına ve 10 Hz'e sahip,
demo olarak zayıf duruyor. Bu klip tarayıcıdan elle indirilecek.

### Karşılaşılan üç uyumluluk sorunu

1. **`pip install -e .` çalışmıyor.** setuptools, egg-info üretirken geçici
   dosyayı yeniden adlandırırken `WinError 5 (erişim engellendi)` alıyor.
   `%TEMP%` yerine proje içi bir klasör göstermek de çözmedi — yani klasör
   izni değil, dosya kilidi sorunu (büyük ihtimalle antivirüs gerçek zamanlı
   taraması). **Çözüm:** venv'in `site-packages` klasörüne `perception_src.pth`
   dosyası konarak `src/` doğrudan yola eklendi. Geliştirme açısından editable
   kurulumla işlevsel olarak aynı; sadece `perception` konsol komutu gelmiyor,
   `python -m perception.cli` kullanılıyor.

2. **Ultralytics 8.4'te `half` kaldırıldı.** Yerine `quantize` geldi
   (`16` = FP16, `None` = FP32). Sarmalayıcı çeviriyi kendi içinde yapıyor,
   config'deki `half: bool` alanı olduğu gibi kaldı — dışarıya bakan arayüz
   değişmedi. Ayrıca 8.4'ün varsayılan takipçisi artık `bytetrack.yaml` değil
   `tracktrack.yaml`; **Hafta 2'de takipçi açıkça belirtilmeli**, varsayılana
   güvenilmemeli.

3. **OpenCV 5.0 kuruldu.** `cv2.VideoWriter_fourcc` hâlâ modül düzeyinde
   duruyor, kod değişikliği gerekmedi. (Kontrol edildi, varsayılmadı.)

### Ölçüm: warmup'ın performans tablosuna etkisi

İlk çalıştırmada pipeline **22.4 FPS** raporladı. Sebep ölçüm hatasıydı:
ilk çıkarım CUDA kernel derlemesi yüzünden 4.03 saniye sürüyor ve bu tek kare
154 karelik ortalamayı aşağı çekiyordu.

`YOLODetector.warmup()` eklenip bu maliyet döngü başlamadan ödendikten sonra:

| Aşama | Çağrı | Ort (ms) | FPS |
|---|---|---|---|
| kare_toplam | 154 | 19.7 | **50.7** |
| tespit | 154 | 14.1 | 71.1 |
| çizim | 154 | 0.9 | 1058.5 |

*(KITTI 1242x375, yolov8n, FP16, GTX 1080)*

Aynı sistem, aynı kod — sadece nereden ölçtüğün değişti. Hafta 8'de performans
tablosu sunulurken warmup'ın ayrıldığı açıkça yazılmalı; aksi halde rakam
yanlış olur.

### Bilinçli erken yatırımlar

Hafta 1'de kesinlikle gerekmeyen ama sonradan eklemesi pahalı olacak iki şey
şimdiden konuldu:

1. **`utils.Profiler`** — katman bazlı süre ölçümü pipeline'a en baştan gömülü.
   Hafta 8'deki performans tablosu bu sınıftan üretilecek; sonradan
   enstrümantasyon eklemek her katmana dokunmayı gerektirirdi.
2. **`Detection` dataclass'ındaki boş alanlar** — `track_id`, `depth`, `bev_xy`
   şimdilik `None`. Sonraki haftalarda katmanlar bu alanları dolduracak; veri
   sözleşmesi baştan sabit olduğu için çizim katmanı da tek noktadan
   güncellenecek (`visualize._build_label` zaten üçünü de okuyor).

### Sonraki hafta için not

ByteTrack Ultralytics içinde hazır geliyor; `model.track(persist=True)` çağrısı
`boxes.id` alanını dolduruyor ve `detection._to_detections` bunu **zaten**
okuyor. Yani Hafta 2'nin asıl işi entegrasyon değil, **kimlik kaybı analizi**
olacak: hangi durumlarda ID atlıyor, kaç kare sürüyor, TTC hesabını nasıl
bozuyor.

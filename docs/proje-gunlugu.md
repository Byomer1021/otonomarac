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

---

## Hafta 2 — Takip ve kimlik sürekliliği

**Tarih:** 15 Ağustos 2026

### Yapılanlar

- `tracking.py` — `BYTETracker` sarmalayıcı, `TrackStats`, hareket izi geçmişi
- `visualize.draw_trails` — boşluk farkındalıklı iz çizimi
- `pipeline.analyze()` — çizim ve video yazma olmadan işleme (tarama için)
- `scripts/tracking_sweep.py` — parametre taraması
- `Config.validate()` — bölümler arası tutarsızlık denetimi

### Karar: `model.track()` değil, `BYTETracker` doğrudan

Hafta 1 notu `model.track(persist=True)` kullanmayı öneriyordu. Vazgeçildi.

O çağrı tespit ve takibi tek işleme gömüyor; sonucunda (a) iki aşamanın süresi
ayrı ölçülemiyor, (b) tespit çıktısı takibe girmeden elden geçirilemiyor.
Mimari diyagramda bu iki kutu ayrı çizildiği için kodda da ayrı duruyorlar.

**Bedeli:** Ultralytics'in iç API'sine bağımlılık. Bağımlılık tek bir sınıfta
(`_ResultsAdapter`) toplandı — kütüphane arayüzü değişirse düzeltilecek tek yer
orası. Ölçüm bunu haklı çıkardı: takip katmanı kare başına yalnızca **1.5 ms**,
yani ayrı ölçülmeseydi tespitin 15 ms'i içinde görünmez kalacaktı.

### Tuzak: detektör eşiği ByteTrack'i sessizce devre dışı bırakıyor

ByteTrack'in ayırt edici özelliği, düşük güvenli tespitleri atmak yerine
**ikinci bir eşleştirme turunda kayıp izleri kurtarmak için** kullanması.
Detektör `conf=0.35` ile filtrelerse `track_low_thresh=0.1` bandına hiçbir
tespit ulaşmaz ve algoritma sıradan IoU takibine düşer — hata vermeden,
sessizce.

Bu yüzden `detection.conf` varsayılanı **0.05**'e çekildi; çıktıyı asıl
temizleyen eşik zaten `new_track_thresh` (0.25). `Config.validate()` de bu
çakışmayı yakalayıp uyarıyor.

Teori ölçümle doğrulandı — taramadaki `conf_015` deneyi eşiği 0.15'e çıkarınca
parçalanma **%44'ten %53'e yükseldi.**

### Sezgiye aykırı bulgu: `match_thresh` yükseltmek eşleştirmeyi *gevşetir*

Kaynak okundu, varsayılmadı: `iou_distance()` maliyet olarak `1 - IoU`
döndürüyor ve `linear_assignment` bunu `lap.lapjv(cost_limit=match_thresh)`
ile üst sınır olarak kullanıyor. Yani `match_thresh=0.9` → `IoU >= 0.1` kabul.

KITTI 10 Hz kayıtlı; kareler arası yer değiştirme büyük olduğu için ByteTrack
varsayılanı 0.8 (`IoU >= 0.2`) izleri koparıyordu. 0.9'a çıkarmak tek başına
parçalanmayı **%44'ten %24'e** indirdi.

### Parametre taraması

`python scripts/tracking_sweep.py data/kitti_0005.mp4 --config configs/default.yaml`

| deney | kimlik | medyan iz | parçalanma | delikli | FPS |
|---|---|---|---|---|---|
| baz (yolov8n, varsayılan) | 52 | 6 | 44% | 33% | 56.7 |
| yolov8s | 50 | 11 | 28% | 26% | 56.1 |
| giriş 960px | 57 | 6 | 42% | 23% | 57.3 |
| detektör eşiği 0.15 | 53 | 4 | **53%** | 36% | 52.3 |
| track_buffer 60 | 52 | 6 | 44% | 33% | 51.6 |
| match_thresh 0.9 | 41 | 10 | 24% | 39% | 50.0 |
| new_track_thresh 0.40 | 40 | 8 | 40% | 28% | 52.0 |
| **yolov8s + match 0.9** | **39** | **14** | **21%** | 33% | 52.9 |

Baz dışındaki her satır bazdan tek eksende ayrılıyor; böylece farkın hangi
değişiklikten geldiği belirsiz kalmıyor.

**yolov8s bu donanımda neredeyse bedava** (56.7 → 56.1 FPS). KITTI çözünürlüğünde
GTX 1080 zaten doymuyor, darboğaz model kapasitesi değil. Zayıf donanımda bu
tercih değişir — `configs/default.yaml`'da not düşüldü.

`track_buffer` hiçbir şeyi değiştirmedi (52 → 52, %44 → %44). Riskler tablosunda
"takip parametrelerinin ayarlanması" önlemi vardı; en akla yatkın parametre
işe yaramayan parametre çıktı.

### Kendi ölçüm betiğimdeki hata

Taramanın ilk çalıştırmasında `baz` 16.1 FPS, `model_s` 22.8 FPS raporladı —
büyük model küçüğünden hızlı görünüyordu. Sebep: zamanlayıcıyı pipeline
kurulmadan **önce** başlatmıştım, yani ağırlık indirme ve CUDA warmup süresi
ölçüme karışıyordu. Zamanlayıcı kurulumdan sonraya alınınca tüm satırlar
50-57 FPS bandına oturdu.

Hafta 1'deki warmup dersinin aynısı, farklı yerde. Ders güncellendi: **ölçüm
sınırını her yeni ölçüm aracında baştan sorgula.**

### Çizim hatası: iz, nesnenin gitmediği yolu gösteriyordu

İlk çıktıda hareket izleri görüntüyü yatay olarak katediyordu. Sebep: bir iz
okluzyon yüzünden kaybolup başka bir noktada geri geldiğinde, geçmişteki son
nokta ile yeni nokta düz çizgiyle birleştiriliyordu.

**Çözüm:** iz noktaları artık `(kare_no, x, y)` olarak saklanıyor;
`draw_trails` ardışık noktalar arasında `max_gap`'ten fazla kare varsa çizgiyi
kesiyor. Bu ayrım hata analizi için kritik — aksi halde okluzyon kaynaklı
boşluk, gerçek bir kimlik atlamasıyla karıştırılır.

İkinci düzeltme: iz uzunluğu kare yerine **saniye** cinsinden tanımlandı
(`trail_seconds: 1.5`). 30 karelik iz, 10 Hz KITTI'de 3 saniyelik dev bir
süpürme izi, 30 fps dashcam'de 1 saniyelik kısa bir iz demek. Saniye
sabitlenince görsel yoğunluk kaynak videodan bağımsız kalıyor.

### Hafta 2 sonucu

| Ölçüm | Önce | Sonra |
|---|---|---|
| Benzersiz kimlik | 55 | 41 |
| Medyan iz uzunluğu | 4 kare | 15 kare |
| Parçalanma (<5 kare) | 51% | **17%** |
| En uzun iz | 126 kare | 154 kare (tüm video) |

Uçtan uca 44.3 FPS (tespit 15.0 ms, takip 1.5 ms, çizim 1.3 ms).

### Sonraki hafta için not (Hafta 2)

`delikli iz` oranı %29'da kaldı ve `match_thresh` yükseltilince **arttı**
(%33 → %39). Beklenen davranış: gevşek eşleştirme, izi okluzyon boyunca
hayatta tutuyor. Kimlik korunduğu için bu iyi haber, ama Hafta 5'te göreli hız
çıkarılırken **iz içindeki boşluklar es geçilemez** — iki nokta arasında 8 kare
varsa hız hesabı o farkı hesaba katmalı. `trails` yapısında kare numarası
zaten saklandığı için veri hazır.

---

## Ara kayıt — kendi çekimimiz devreye girdi

**Tarih:** 17 Ağustos 2026

Hafta 1'de "vitrin GIF'i için temiz lisanslı bir dashcam klibi lazım" diye açık
bırakılan madde kapandı: video kendimiz çekildi (Maltepe, İstanbul). Lisans
sorunu tamamen ortadan kalktı ve hazır veri setlerinde bulunmayan bir trafik
karakteri kazanıldı.

**Kaynak:** 2560×1440, 59.94 fps, VP9/MKV, 75.5 dakika, 6.5 GB.

### Klip üretimi

6.5 GB'lık dosyayı olduğu gibi işlemek anlamsız. `ffmpeg` ile iki klip kesildi:

| Klip | Kaynak konumu | Süre | Amaç |
|---|---|---|---|
| `maltepe_city.mp4` | 14:30 | 60 sn | Ana geliştirme klibi, yoğun şehir trafiği |
| `maltepe_test.mp4` | 10:00 | 20 sn | Hızlı iterasyon |

H.264, 1920×1080, 30 fps (60'tan yarıya indirildi — ardışık kareler algı için
gereksiz derecede benzer), CRF 20.

**Ses bilinçli olarak atıldı (`-an`).** Kaynakta opus ses akışı var ve araç içi
konuşma içerebilir; bu klipler herkese açık demoya gidecek. Algı için de
gereksiz.

Videonun ilk ~40 dakikası şehir içi (asıl malzeme), sonrası kırsal ve büyük
ölçüde boş yol. Kırsal bölüm Hafta 8'de "nesne yokken sistem ne yapıyor"
senaryosu için ayrıldı.

### Yanlış çıkan tespitim: mercek distorsiyonu

İlk karelere bakıp **"belirgin balık gözü distorsiyonu var, Hafta 4'teki
homografi bozulur"** dedim. Ölçtüm, yanılmışım.

Kırsal bölümdeki ufuk çizgisi gerçek dünyada dümdüz olmak zorunda ve görüntünün
tam genişliğini kat ediyor. Gökyüzü-zemin geçişi sütun sütun bulunup iki ucu
birleştiren düz çizgiyle karşılaştırıldı: **sol uç y=891, sağ uç y=891**,
2560 pikselin tamamında sapma yok. Ortadaki 48 piksellik fark uzaktaki
tepelerden geliyor, mercekten değil.

Ufuk çizgisi dikey merkezin 171 piksel altında, yani merkezden uzakta —
barrel distorsiyon olsaydı uçlarda aşağı doğru kıvrılırdı. Kıvrılmıyor.

**Sonuç:** kamera ya rektilineer ya da düzeltmeyi kendi içinde yapıyor.
Hafta 4 için kalibrasyon detouru gerekmiyor. Binaların kenarlarında gördüğümü
sandığım eğrilik, geniş açıda normal olan perspektif yakınsamasıymış.

*Ders: "gözle bakıp karar verme" hatası, Hafta 1 ve 2'deki ölçüm derslerinin
üçüncüsü. Bu sefer iddiayı yayınlamadan önce ölçtüm.*

### Gerçek olan sorun: kaput ve torpido

Görüntünün altında araç içi var ve bu yol değil. Zamansal varyans yeterince
keskin ayrım vermedi (kaputta yansıma oynuyor), bu yüzden ızgara bindirilip
elle okundu:

- **y ≈ 915**: silecek ve torpido başlıyor
- **y ≈ 940 altı**: tamamen araç içi
- Kullanılabilir yol alanı: **üstteki %83** (1080 pikselin ~900'ü)

Bu Hafta 4 için iki yerde lazım olacak: homografi kaynak noktaları bu sınırın
üstünden seçilmeli, ve alt kenarı bu sınırın altına düşen tespitlerin "zemine
değme noktası" geçersiz sayılmalı — o piksel yolu değil kaputu gösteriyor.

### Ölçüm: `match_thresh` hipotezim de yanlış çıktı

Hafta 2 sonunda "`match_thresh=0.9` bulgusu KITTI'nin 10 Hz'ine özgü olabilir,
30 fps'te kareler arası hareket küçük olacağı için varsayılan 0.8 daha iyi
çıkabilir" diye tahmin etmiştim. Test edildi:

| deney | kimlik | medyan iz | parçalanma | delikli |
|---|---|---|---|---|
| 0.7 (sıkı) | 263 | 4 | **%55** | %21 |
| 0.8 (ByteTrack varsayılanı) | 128 | 22 | %23 | %27 |
| **0.9 (gevşek)** | **104** | **36** | **%16** | %29 |
| new_track_thresh 0.40 | 91 | 45 | %15 | %24 |

30 fps'te de gevşek eşik kazandı, tahminimin tersine. Sebep muhtemelen kare
hızı değil nesne ölçeği: şehir içinde yakın araçlar karede büyük ve tespit
kutusunun boyutu kare kare oynuyor; sabit duran bir nesnede bile IoU bu yüzden
düşüyor ve dar eşik izi koparıyor.

`new_track_thresh: 0.40` biraz daha iyi görünüyor ama **bedava değil**: eş
zamanlı iz sayısı 9.5'ten 8.8'e düşüyor, yani parçalanmanın bir kısmı "daha az
şey takip ederek" kazanılıyor. Uzaktaki küçük araçlar hiç iz almıyor.
Varsayılan değiştirilmedi; Hafta 5'te TTC'nin hangi nesnelere ihtiyaç duyduğu
netleşince yeniden bakılacak.

### İlk gerçek çalıştırma

`maltepe_test.mp4` (20 sn, 600 kare) üzerinde mevcut ayarlarla:

| Ölçüm | KITTI | Maltepe |
|---|---|---|
| Parçalanma | %17 | %16 |
| Medyan iz | 15 kare | 36 kare |
| En uzun iz | 154 kare | 312 kare (10 sn) |
| Uçtan uca | 44.3 FPS | 28.2 FPS |

Tespit güveni belirgin daha yüksek (0.83-0.91 aralığı), çünkü görüntü KITTI'nin
1242×375'ine göre çok daha yüksek çözünürlüklü ve nesneler karede büyük.
FPS düşüşü de bundan: 1280 genişlikte işlerken kare başına tespit 13.7 ms'den
23.3 ms'e çıkıyor.

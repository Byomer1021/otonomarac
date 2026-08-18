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

---

## Hafta 3 — Derinlik ve füzyon

**Tarih:** 18 Ağustos 2026

### Yapılanlar

- `depth.py` — Depth Anything V2 sarmalayıcı, kutu-derinlik füzyonu
- `CameraConfig` — kaput sınırı; Hafta 4'te homografi noktaları da buraya
- `visualize.colorize_depth` / `stack_panels` — çift panelli çıktı altyapısı
- `configs/maltepe.yaml` — kendi çekimimize özel ayarlar
- `scripts/depth_stride_check.py` — derinlik yeniden kullanımının hata ölçümü

### Donanım kısıtı: GPU ağır yük altında düşüyor

**Bu haftanın en belirleyici olayı teknik bir tercih değil, donanım.**

Depth Anything ölçümleri sırasında GTX 1080 **iki kez** ekran bağlantısını
düşürdü. `nvidia-smi` çıktısı: `GPU is lost. Reboot the system to recover this
GPU`. Her seferinde makineyi kapatıp açmak gerekti.

Ayrım önemli: tespit ve takip onlarca çalıştırmada, yüzlerce kare boyunca hiç
sorun çıkarmadı. Kartı düşüren özellikle derinlik modelinin sürekli yükü.

**Karar: derinlik CPU'da çalışacak.** 321 ms/kare, `every_n_frames=3` ile
60 saniyelik klip ~3 dakikada bitiyor — gözetimsiz çalıştırılabilir. Tespit ve
takip GPU'da kalıyor.

Bu, Hafta 1'deki "GPU var, Colab yedeğe düştü" kaydını kısmen geri alıyor.
Yerel GPU tespit için güvenilir, ağır katmanlar için değil. Hafta 7-8'deki
yüksek kaliteli render'lar için Colab yeniden masada.

*İtiraf: ilk çöküşten sonra strateji değiştirmek yerine ölçüme devam ettim ve
kartı ikinci kez düşürdüm. Tekrarlayan donanım hatası, parametre hatası gibi
ele alınmamalı.*

### FP16 Pascal'da tuzak

Ölçüldü: derinlik modeli **FP16'da 56.5 ms, FP32'de 41.0 ms.** Yarı hassasiyet
%38 daha yavaş. Sebep mimari — Pascal (sm_61) hızlı FP16 yoluna sahip değil.
Üstelik çöken çalıştırma da FP16'ydı.

`utils.resolve_half()` eklendi: sm_70 altındaki kartlarda FP16 isteği
**gerekçesiyle birlikte reddediliyor**, sessizce yok sayılmıyor. Hem tespit
hem derinlik katmanı bu kontrolden geçiyor.

Hafta 1 ve 2'deki tüm `--half` çalıştırmaları geriye dönük olarak şüpheli;
o ölçümler muhtemelen olduğundan yavaş.

### `input_width` ayarım hiçbir şey yapmıyormuş

CPU ölçümünde genişliği 518'den 308'e düşürmek süreyi değiştirmedi (365 → 368
ms). Bu, ayarın uygulanmadığının işaretiydi.

Sebep: HuggingFace `AutoImageProcessor` kendi `size` ayarına (518×518,
`ensure_multiple_of=14`) göre yeniden ölçekliyor. Önceden küçültülmüş kareyi
bile **geri büyütüyor**. Yani harici ön-ölçekleme modelin maliyetini hiç
değiştirmiyordu.

Düzeltildikten sonra (`size` processor'e açıkça verilerek) ayar gerçekten
çalışıyor:

| tensor genişliği | ms (CPU) | hızlanma |
|---|---|---|
| 518 | 321 | 1.00x |
| 392 | 183 | 1.75x |
| 294 | 109 | 2.96x |
| 224 | 71 | 4.53x |

### Tasarım hatası: kare bazlı normalizasyon

İlk sürüm haritayı her karede `[0,1]`'e taşıyordu. Görsel olarak cazip ama
**kareler arası karşılaştırmayı imkânsız kılıyor**: sahneye tek bir yakın nesne
girdiğinde tüm haritanın ölçeği kayıyor ve hiçbir şey hareket etmemiş olsa bile
her nesnenin "mesafesi" değişiyor.

Hafta 5'teki hız çıkarımı tam da kareler arası farka dayandığı için bu kabul
edilemez. `infer()` artık **ham** disparity döndürüyor; normalizasyon yalnızca
çizim katmanında, yalnızca renk haritası için yapılıyor
(`normalize_for_display`).

Bu hata ölçüm sırasında ortaya çıktı: çözünürlükler arası hata oranları
anlamsız görünüyordu, çünkü ölçtüğüm şeyin bir kısmı normalizasyon oynamasıydı.

### Ham disparity negatif olabiliyor

Ölçülen aralık: **-0.30 .. 10.41**. `1/d` orada anlamlı bir mesafe üretemez.

`relative_distance()` artık geçersiz disparity için `None` döndürüyor. Tabana
kırpıp devasa bir sayı üretmek daha kolaydı ama o sayı ölçülmüş gibi görünürdü.

### Çözünürlük düşürmenin gerçek bedeli

Ham değerler çözünürlükler arası doğrudan karşılaştırılamıyor (modelin çıktı
ölçeği girdi boyutuna bağlı). Bu yüzden ölçekten bağımsız bir ölçüt kullanıldı:
**kare içinde nesne sıralamasının Spearman korelasyonu.**

| genişlik | Spearman (ort) | en kötü kare |
|---|---|---|
| 392 | **0.917** | 0.798 |
| 294 | 0.757 | 0.226 |
| 224 | 0.661 | 0.305 |

392 güvenli ve iki kat hızlı. 294'te sıralama bozuluyor — oraya inilmemeli.
Varsayılan 518'de bırakıldı (referans kalite, CPU'da maliyeti kabul edilebilir).

### Küçük kazanç: büyütmeyi cv2 yapıyor

Haritayı kare boyutuna geri büyütmek torch bicubic ile 8.9 ms, `cv2.INTER_CUBIC`
ile 1.3 ms. İki sonuç arasındaki fark maksimum %0.001. Yedi kat hızlı, aynı çıktı.

### Sonuç

| Aşama | ms/kare |
|---|---|
| derinlik (CPU, her 3 karede bir) | 134.4 |
| tespit (GPU) | 21.2 |
| çizim (çift panel) | 20.2 |
| takip | 2.5 |
| füzyon | 0.7 |
| **uçtan uca** | **197.7 (5.1 FPS)** |

Nesne derinlikleri fiziksel olarak doğru sıralanıyor: en yakın kamyonet `~0.2`,
sonra `~0.5`, `~0.8`, `~1.1`, en uzak araç `~2.0`. Hafta 3'ün asıl doğrulaması
buydu — mutlak değer değil, sıralamanın tutarlılığı.

### Sonraki hafta için not

Kaput sınırı (`camera.hood_top = 0.85`) zaten ölçülü ve config'de. Hafta 4'te
homografi kaynak noktaları bu sınırın **üstünden** seçilmeli; alt kenarı
sınırın altına düşen tespitlerin zemine değme noktası da geçersiz sayılmalı.

Ölçek belirsizliği hâlâ açık: `relative_distance` birimsiz. Hafta 5'te şerit
genişliği (3.5 m) referansıyla kalibre edilecek.

---

## Hafta 4 — Kuşbakışı projeksiyon

**Tarih:** 18 Ağustos 2026

### Yapılanlar

- `bev.py` — homografi (Yöntem A), zemin projeksiyonu, harita çizimi
- `CameraConfig.road_quad` — kalibrasyon dörtgeni, piksel değil oran olarak
- `scripts/calibrate_bev.py` — dörtgen seçme + **bağımsız** doğrulama
- Çift panelli çıktı: solda kamera, sağda harita

### Kalibrasyon: gözle değil ölçümle

Homografiyi kurmak kolay, doğru kurduğunu bilmek zor. Süreç üç adımda ilerledi
ve ikisinde yanıldım.

**Adım 1 — otomatik şerit tespiti denendi, bırakıldı.** Renk maskesi + Hough
ile sarı orta çizgi temiz bulundu, ama beyaz maske kaldırımı, park etmiş
araçları ve binaları da yakaladı; sarı maske de trafik konilerine takıldı.
Satır satır incelendiğinde gerçek çizginin yalnızca y 552-611 arasında olduğu
görüldü. Otomatik şerit çıkarımı zaten **Hafta 6'nın işi** — buraya çekmek
kapsam kayması olurdu, bırakıldı.

**Adım 2 — ilk doğrulamam döngüseldi.** Dörtgeni sarı çizgiden kurup sonra
"düzleştirilmiş görünümde sarı çizgi dikey mi" diye ölçüyordum. Elbette
dikeydi. Belirti açıktı ama fark etmesi zaman aldı: **onlarca farklı ufuk
değeri tam olarak aynı mükemmel skoru veriyordu**, yani ölçüm kalibrasyonu hiç
kısıtlamıyordu. Bir doğrulama hiçbir yapılandırmayı elemiyorsa doğrulama değildir.

**Adım 3 — bağımsız referans: araç genişliği.** Kalibrasyonda hiç kullanılmayan
bir büyüklük seçildi. Tespit edilen araçların kutu alt kenarı zemine yansıtılıp
genişliği ölçülüyor. İki beklenti var:

- medyan tipik araç genişliğine (~1.8 m) yakın olmalı
- genişlik **mesafeden bağımsız** olmalı — asıl test bu; korelasyon sıfırdan
  uzaksa perspektif doğru kaldırılmamış demektir

Bu ölçütle ufuk yüksekliği tarandı:

| ufuk y | örnek | medyan genişlik | korelasyon |
|---|---|---|---|
| 470 | 48 | 1.46 m | −0.20 |
| **480** | **44** | **1.64 m** | **−0.02** |
| 490 | 30 | 1.77 m | +0.27 |
| 500 | 18 | 2.09 m | +0.46 |
| 510 | 13 | 2.39 m | +0.49 |

y=480'de korelasyon sıfırlanıyor. Not: iki yol-paralel çizginin kesişiminden
hesaplanan kaybolma noktası y=510 demişti; araç genişliği testi y=480 diyor.
Aradaki fark beyaz çizgi fit'inin belirsizliğinden geliyor ve **çözülmüş
değil** — fiziğe daha doğrudan bağlı olan ikinci ölçüt tercih edildi.

Ölçek: `quad_width_m` şerit genişliği varsayılarak değil, araç genişliğinden
**ölçülerek** 3.84 m'ye ayarlandı.

### Ölçek belirsizliği somutlaştı

`quad_depth_m` 6'dan 22 m'ye değiştirildiğinde:

| quad_depth_m | medyan genişlik | korelasyon | en uzak araç |
|---|---|---|---|
| 6 | 1.62 m | +0.04 | 16.9 m |
| 12 | 1.75 m | −0.04 | 33.8 m |
| 22 | 1.94 m | −0.07 | 63.4 m |

Genişlik ve korelasyon değişmiyor, sadece mutlak mesafeler ölçekleniyor.
Yani **bu test boylamsal ölçeği kısıtlayamıyor.** Ön rapordaki "ölçek
belirsizliği" maddesinin somut hâli bu: yanal ölçek bir referansla
sabitlenebildi, boylamsal ölçek sabitlenemedi. 12 m varsayıldı ve config'de
varsayım olduğu açıkça yazıldı.

### Doğrulama betiğindeki ikinci hata

Betik ilk çalıştığında medyan genişlik 1.14 m çıktı, kendi ölçümüm 1.64
demişti. Sebep: betik `config.detection` kullanıyordu ve orada `conf: 0.05`
yazıyor — o eşik ByteTrack'in ikinci eşleştirme turu için **bilinçli olarak**
düşük. Doğrulama için zararlı: kısmi ve örtülü kutular gerçekte olduklarından
dar ölçülüp medyanı aşağı çekiyor. Doğrulama artık kendi eşiğini (0.40)
kullanıyor.

Ders: bir parametrenin bir katmandaki doğru değeri, başka bir katmanda yanlış
olabilir. Config'i olduğu gibi devralmak sessiz bir hata kaynağı.

### Nihai durum

Doğrulama, 20 kare / 46 araç örneğiyle: medyan yansıtılmış genişlik
**1.89 m** (hedef 1.80, %5 sapma), mesafe korelasyonu **−0.25**. "Kabul
edilebilir" — kalan korelasyon ufuk tahminindeki gerçek belirsizliği yansıtıyor
ve gürültüye aşırı uydurmak yanlış olurdu.

Projeksiyon kare başına **0.1 ms**. Uçtan uca (derinlik kapalı) 30.2 FPS.

### Modelleme notu: haritanın orijini araç değil

Koordinat orijini kalibrasyon dörtgeninin yakın kenarının ortası — yani şeridin
ortası. Kamera oranın tam üzerinde olmak zorunda değil, bu yüzden ego aracın
yanal konumu görüntünün orta sütununun zemin karşılığından hesaplanıyor
(`ego_offset_m`).

Boylamsal konum ise **bilinmiyor**: dörtgenin yakın kenarının araca uzaklığı
ölçülmedi. Harita "yakın referans satırından itibaren" mesafe gösteriyor,
aracın burnundan değil. Hafta 5'te TTC hesaplanırken bu sabit kaydırma önemli
hale gelecek.

### Sonraki hafta için not

Hafta 5'in iki işi var ve ikisi de bu haftanın açık bıraktığı yerden başlıyor:
boylamsal ölçeği bağımsız bir referansla sabitlemek (şerit kesik çizgi
periyodu bir aday), ve ego'nun harita üzerindeki gerçek konumunu belirlemek.
İkisi de yapılmadan TTC sayısı üretmek, ölçülmüş gibi görünen bir tahmin
üretmek olur.

---

## Hafta 5 — Hız ve çarpışma riski

**Tarih:** 18 Ağustos 2026

### Yapılanlar

- `risk.py` — iz geçmişinden yaklaşma hızı, TTC, risk sınıflandırması
- Riskli nesnelerin kamerada ve haritada vurgulanması
- Çalıştırma sonunda risk özeti

### Ölçek endişesi gereksizmiş: TTC ölçek-değişmez

Hafta 4'ün sonunda "boylamsal ölçek sabitlenmeden TTC üretmek yanlış olur"
diye not düşmüştüm. **Yanlıştı.**

Bütün mesafeler bilinmeyen bir k katsayısıyla çarpılırsa, yaklaşma hızı da
aynı k ile çarpılır:

```
TTC = mesafe / yaklaşma_hızı = (k·d) / (k·v) = d / v
```

Ölçüldü — `quad_depth_m` iki katına çıkarıldığında mesafe tam iki katına
çıkıyor ama TTC değişmiyor:

| quad_depth_m | en düşük TTC | o anki mesafe |
|---|---|---|
| 6 | 1.29 s | 1.3 m |
| 12 | 1.29 s | 2.6 m |
| 24 | 1.29 s | 5.2 m |

**Projenin en çok işe yarayan çıktısı, en zayıf varsayımından bağımsız çıktı.**
Haritadaki metre değerleri hâlâ bir ölçek katsayısı kadar belirsiz; saniye
cinsinden TTC metrik olarak doğru.

### Yan not: yanal referanslar boylamsal ölçeği neden veremez

Hafta 4'te bunu ölçümle görmüştüm ama sebebini bu hafta çıkardım. Düz yol,
kamera yüksekliği h, ufuk satırı v_h, odak f:

```
X = (u − u_c) · h / (v − v_h)      ← f sadeleşir
Z = f · h / (v − v_h)              ← f kalır
```

Yani araç genişliği veya şerit genişliği gibi **yanal** ölçümler yalnızca
kamera yüksekliğini verir, mesafeyi vermez. Ön rapordaki "şerit genişliği
üzerinden ölçek kalibre edilecek" planı bu noktada eksikti.

Kamera yüksekliği türetildi ve **1.43 m** çıktı (107 araç örneği, çeyrekler
1.12-1.89). Ön cama monte bir dashcam için beklenen aralık; yanal
kalibrasyonun bağımsız bir fiziksel doğrulaması.

### Üç yanlış sonuç, üç düzeltme

**1. Park halindeki her araç "yaklaşıyor" çıktı.** İlk çalıştırmada 104 izin
**57'si kritik**. Sebep kavramsal değil hatalı da değil: ego araç hareket
ettiği için yol kenarındaki her nesne gerçekten yaklaşıyor. Ama yol kenarındaki
araçları çarpışma riski sayan bir uyarı sistemi işe yaramaz.

Çözüm: ego güzergâh koridoru (`path_half_width_m: 1.7`). 104 izden 6'sı
koridora giriyor.

**2. Düşük güvenli tespitler sahte hız üretti.** `detection.conf` ByteTrack'in
ikinci turu için bilinçli olarak 0.05'te; o zayıf kutular takibi ayakta tutuyor
ama kare kare titriyor. Risk için ayrı bir eşik kondu (`min_confidence: 0.35`).

**3. Asıl sebep örtülmeymiş.** Güven eşiği `#235`'i elemedi — medyan güveni
0.42, eşiği geçiyor. Kareye bakınca gerçek sebep göründü: o araç önündeki
Duster'ın **arkasında kısmen örtülü.** Örtülünce kutunun görünen alt kenarı
gerçek zemin temas noktasının üstünde kalıyor, homografi de aracı olduğundan
uzağa koyuyor (22 m raporlandı, görsel olarak ~10 m). Örtülme kare kare
değiştiği için mesafe zıplıyor ve 14 m/s'lik uydurma bir yaklaşma hızı çıkıyor.

Bu, zemine değme varsayımının bilinen kırılma noktası ve güven eşiğiyle
çözülmez. Çözüm fit kalitesi kapısı: gerçekten yaklaşan nesnenin mesafe-zaman
eğrisi düz bir doğrudur, örtülme artefaktı zıplar.

| artık eşiği | uyarı veren izler |
|---|---|
| kapalı | #3(3), #6(39), **#235(13)** |
| 0.10 | #3(3), #6(21), **#235(13)** |
| 0.04 | #3(3), #6(21) |

**Bedeli dürüstçe:** 0.04 eşiği sahte izi eliyor ama gerçek uyarının
gözlemlerini de 39'dan 21'e düşürüyor. Uyarı yaklaşma boyunca yine çıkıyor,
sadece daha seyrek. Sahte kritik uyarı üretmemek daha önemli sayıldı.

### Tasarım kararları

- **Zaman kare sayısıyla değil saniyeyle ölçülür.** İzlerin %29'u delikli;
  iki gözlem arası 1 kare de olabilir 8 kare de. Hafta 2'de bırakılan not
  burada kullanıldı.
- **Hız iki noktadan değil pencereye fit edilerek çıkarılır.** Ardışık iki
  kare arasındaki fark, kutu titremesinin yanında kaybolur.
- **Yetersiz veride sayı üretilmez.** Kısa iz, düşük yaklaşma hızı, koridor
  dışı, düşük güven, kötü fit — hepsinde `None`. Ekranda görünen bir sayı
  ölçülmüş sayılır.

### Sonuç

400 karelik test: koridorda 6 iz, 3 uyarı, 2 kritik. En düşük TTC **1.3 s**
(#6, 2.6 m, 2.0 m/s yaklaşıyor) — tam önümüzdeki SUV, görselde kalın kırmızı
kutuyla işaretli. Risk katmanı kare başına **0.2 ms**.

### Bilinen kısıt

Koridor kapısı uzak mesafede güvenilirliğini yitiriyor: homografinin yanal
hatası mesafeyle büyüdüğü için 20 m ötedeki bir nesnenin koridorda olup
olmadığı kesin değil. Hafta 8'deki hata analizinde ölçülmeli.

---

## Hafta 6 — Şerit ve sürülebilir alan

**Tarih:** 19 Ağustos 2026

### Yapılanlar

- `segmentation.py` — SegFormer (Cityscapes) yol maskesi + şerit boyası
- `BEVProjector.warp_mask` — maskeyi harita düzlemine taşıma
- Haritada sürülebilir alan, şerit boyası ve görüş engeli lejantı

### Hafta 4'ten kalan borç kapandı

Hafta 4'te şerit çizgilerini renkten bulmayı denemiş ve bırakmıştım: beyaz
maske kaldırımı, park etmiş araçları ve bina cephelerini de yakalıyordu.
"Şerit çıkarımı Hafta 6'nın işi" diye ertelemiştim.

Çözüm ayrı bir dedektör değil, **kapılama** oldu. Aynı HSV eşiği artık yalnızca
yol maskesinin içinde uygulanıyor ve sarı orta çizgiyi temiz yakalıyor. Sorun
eşikte değil, eşiğin uygulandığı bölgedeymiş.

### Model seçimi ve kaput

Cityscapes üzerinde eğitilmiş SegFormer-b0 (3.7 M parametre, 19 sınıf).
`road` sınıfı doğrudan sürülebilir alanı veriyor.

İlk denemede model **kaputu yol sanıyordu** — kadrajın alt %15'i araç ve
"yol" pikselinin büyük kısmı oraya düşüyordu. Kaput kesilince hem bu hata
gitti hem de kadraj Cityscapes'in eğitildiği çerçeveye yaklaştı: kaputun
üstündeki yol oranı %3.5'ten %6.4'e çıktı.

Bu arada bir ölçüm yanılgısına da düştüm: "yol pikseli %11" düşük görünüyordu.
Değilmiş — geniş açılı bu kadrajın %36'sı gökyüzü, %27'si bina. Yüzde yanlış
metrikti; maskeye bakınca yol temiz çıkmıştı.

### Haritadaki ışınsal çizgiler hata değil

Maske haritaya taşındığında ufuktan yayılan ışınsal desenler çıktı. İlk
tahminim ufka yakın satırların devasa alanlara yayılmasıydı ve menzil kırpması
ekledim.

**Kırpma bu klipte sıfır piksel siliyor.** Ölçtüm: kırpma satırı v=498, yol
maskesi ise 508-611 satırları arasında — ufka hiç yaklaşmıyor. Kod yine de
duruyor, çünkü yolun daha yükseğe uzandığı bir çekimde gerekli olacak.

Gerçek sebep başkaydı: **görüş engeli gölgeleri.** Görüntüde önünde araç duran
her yön haritada boş kalıyor. Bu fiziksel olarak doğru — harita gördüğümüz
yolu gösteriyor, var olduğunu bildiğimiz yolu değil. Ama etiketlenmezse hata
gibi okunuyordu, o yüzden haritaya lejant eklendi.

Ölçülen sayı: haritanın %16.2'si sürülebilir işaretli.

### Sonuç

| Aşama | ms/kare |
|---|---|
| derinlik (CPU, 3 karede bir) | 117.0 |
| segmentasyon (CPU, 5 karede bir) | 57.2 |
| tespit (GPU) | 20.4 |
| çizim | 9.5 |
| takip | 2.8 |
| füzyon | 0.8 |
| risk | 0.4 |
| projeksiyon | 0.1 |
| **uçtan uca** | **220.8 (4.5 FPS)** |

İki ağır katman da CPU'da; GPU'ya yalnızca tespit biniyor ve kart bu hafta hiç
sorun çıkarmadı.

### Sonraki hafta için not

4.5 FPS canlı demo için fazla yavaş. Hafta 7'de Hugging Face Spaces ücretsiz
CPU katmanında çalışacak, yani tespit de CPU'ya inecek. Oradaki asıl kaldıraç
`every_n_frames` ve çözünürlük; ölçüm zaten var, tahmin gerekmiyor.

---

## Hafta 7 — Yayınlama

**Tarih:** 19 Ağustos 2026

### Yapılanlar

- `app.py` — Gradio arayüzü, çevrimdışı işleme
- `configs/spaces.yaml` — ücretsiz CPU katmanı için ayarlanmış config
- `examples/maltepe.mp4` — kalibrasyonu bilinen 10 saniyelik örnek (1.7 MB)
- `docs/spaces-README.md` — Spaces frontmatter'ı

### CPU ayarları ölçüldü

Ücretsiz katmanda GPU yok; tespit de CPU'ya iniyor.

| Ayar | ms/kare | FPS |
|---|---|---|
| mevcut yerel ayarlar | 342 | 2.9 |
| yolov8n + imgsz 480 | 240 | 4.2 |
| + genişlik 960 | 222 | 4.5 |
| + derinlik 392, her 5 karede | 170 | 5.9 |
| + segmentasyon 512, her 10 karede | **126** | **7.9** |

15 saniyelik klip yaklaşık bir dakikada bitiyor.

### Beklenmedik bulgu: iş parçacığı sayısı önemsiz

Spaces 2 vCPU veriyor, bu makinede 12 mantıksal çekirdek var. Gerçekçi bir
tahmin için torch'u 2 iş parçacığına sınırlayıp tekrar ölçtüm.

**Fark neredeyse yok: 121 ms'e karşı 126 ms.** Beklentim belirgin bir yavaşlama
yönündeydi. Bu model boyutlarında (yolov8n@480, SegFormer-b0@512,
Depth-Anything-Small@392) darboğaz paralel hesap değil, işlem başı ek yük.

Pratik sonucu: Spaces tahminini "çekirdek sayısı oranında yavaşlar" diye
kurmak yanlış olurdu.

### Karar: yüklenen video için kuşbakışı harita kapalı

Homografi, kameraya ve montaja özgü dört nokta ister. Bir ziyaretçinin
yüklediği klip için o bilgi yok.

Makul görünen genel bir dörtgen koymak mümkündü ve demo daha zengin görünürdü.
Yapılmadı: o dörtgen, ölçülmüş gibi okunan ama ölçülmemiş mesafeler üretirdi —
projenin baştan beri kaçındığı şey tam olarak bu. Arayüz bunu gerekçesiyle
yazıyor ve kendi videosu için kalibrasyon komutunu veriyor.

Tespit, takip ve derinlik kameradan bağımsız olduğu için her videoda çalışıyor.
Depodaki Maltepe örneğinin gerçek kalibrasyonu var, harita orada açılıyor.

### İki çıktı hatası

**Rapor "0 kare" diyordu.** `kare_toplam` sayacı `pipeline.run()` içinde
artıyor; arayüz `process_frame` + `render` çağırdığı için hiç artmıyordu.
Sayım arayüze taşındı.

**Çıktı 10 saniye için 18.45 MB'tı.** `VideoWriter` mp4v (MPEG-4 Part 2)
yazıyor — hem şişkin hem de tarayıcıların çoğu oynatmıyor, yani Gradio'nun
oynatıcısında boş kutu görünürdü. H.264'e yeniden kodlama eklendi:
**2.53 MB**, yedi kat küçük ve oynuyor.

### Ölçülen sonuç

Örnek video (300 kare, 960x540, hepsi CPU): **49.5 saniye, 6.1 kare/sn**.
Tespit 45.2 ms, derinlik 38.9, segmentasyon 16.4, çizim 9.2, kalanlar 3 ms altı.

### Kalan

Deploy adımı benim çalıştırabileceğim bir şey değil — Hugging Face kimlik
bilgisi gerekiyor. Komutlar README'de hazır, `docs/spaces-README.md` Space'in
frontmatter'ı olarak duruyor.

---

## Hafta 8 — Dokümantasyon ve hata analizi

**Tarih:** 19 Ağustos 2026

### Yapılanlar

- `scripts/failure_analysis.py` — dört ölçümlü hata analizi
- `docs/hata-analizi.md` — sonuçların yorumlanmış hâli
- İki README'ye "nerede başarısız oluyor" ve konsolide performans tablosu
- `data/maltepe_rural.mp4` — ikinci test sahnesi (kaynağın 72. dakikası)

### Ölçülen dört şey

**1. Bilgi hunisi.** Ham tespitin yalnızca **%2'si** TTC'ye dönüşüyor. Bu kayıp
değil tasarım: %52 kimlik alamıyor (o band ByteTrack'in kurtarma turu için
var), kalanın çoğu güzergâh koridorunun dışında, geri kalanı fit kalitesi
kapısına takılıyor. Sistem üretebileceği sayıların çoğunu bilerek üretmiyor.

**2. Mesafe kararlılığı bandlara ayrıldı.** Gürültü 0-8 m'de medyan 3.9 m/s,
25-40 m'de 14.6 m/s — **3.7 kat**. 90. yüzdelik 54.4 m/s, yani saatte 196 km.
`bev.max_range_m = 40` artık keyfi bir sayı değil, bu tablodan geliyor.

**3. Örtülme sayıya döküldü.** Hafta 5'te tekil bir örnekte görülmüştü; ölçüm
genel olduğunu gösterdi: örtülü kutuların gürültüsü örtüşmeyenlerin **iki katı**
(10.0'a karşı 5.2 m/s medyan).

**4. Sahne bağımlılığı.** Kırsalda parçalanma %40 ama delikli iz **%0**.
Şehirde %17 ve %25. Birlikte okununca ortaya çıkan şey: **takip zorluğu nesne
sayısından değil, nesnelerin birbirini örtmesinden geliyor.**

### Beklenmedik bulgu: boş sahne kaputu hayalet araca çeviriyor

Kırsal klipte tespitlerin **%71'i** kaput bölgesindeydi — hepsi araç sınıfı,
medyan güven 0.13. Şehirde aynı oran %4.

Sebep iki katmanın etkileşimi. `detection.conf` ByteTrack için 0.05'te ve
sahnede gerçek nesne yokken o eşik kaputun yansımalarını araba olarak
etiketliyor. Şehirde gerçek trafiğin arasında kayboluyorlar, boş yolda baskın
hale geliyorlar.

Haritaya ulaşmıyorlardı ama takip onlara kimlik harcıyordu.

**Düzeltme ve ölçütün seçimi:** ilk sürüm kutunun **tamamının** kaput altında
olmasını aradı ve neredeyse hiçbir şeyi elemedi — hayaletlerin üst kenarı
genelde çizginin biraz üstünde başlıyor. Ölçüm yapıldı: kutusunun yarısından
fazlası kaputta olanlar şehirde 142/168, kırsalda 184/234. Ölçüt yarıya çekildi.

| Ölçüm | Önce | Sonra |
|---|---|---|
| Kırsal ham tespit | 890 | 330 |
| Kırsal zemin konumu | %20 | %55 |
| Şehir ham tespit | 10.578 | 10.322 |
| Şehir parçalanma | %17 | %17 |

Hayaletleri eliyor, gerçek nesnelere dokunmuyor.

### Plandan son sapma: Hugging Face ücretsiz katmanı daraldı

Ön rapor yayınlama hedefini "Hugging Face Spaces (ücretsiz CPU katmanı)" diye
yazmıştı. 2026'da HF bunu değiştirdi: statik Space'ler herkese açık kaldı ama
Gradio/Docker Space'i `cpu-basic`'te barındırmak PRO abonelik istiyor, ücretsiz
hesaplar ZeroGPU'ya yönlendiriliyor.

Dış bir değişiklik, planlama hatası değil. PRO ile devam edildi; ZeroGPU yolu
da betikte duruyor (`--hardware zero-a10g`).

### Dürüstçe eksik kalanlar

- **Gece, yağmur, sis test edilmedi.** Çekimde o koşullar yok. Ön rapor
  BDD100K'yı dayanıklılık testi için işaretlemişti, yapılmadı.
- **Eğimli yol denenmedi.** Maltepe düz; düzlem varsayımı hiç zorlanmadı.
  Çalışmadığı değil, bilinmediği için kısıt.
- **Yöntem B (derinlik tabanlı projeksiyon) eklenmedi.** Rapor iki yöntemi
  karşılaştırmayı öneriyordu; yalnızca A uygulandı. Derinlik ve homografi
  birbirini çapraz doğruladı ama sistematik karşılaştırma yapılmadı.

### Sekiz haftanın özeti

| Hafta | Çıktı | Ölçülen sonuç |
|---|---|---|
| 1 | İskelet ve tespit | 50.7 FPS (ısınma ayrıldıktan sonra) |
| 2 | Takip | Parçalanma %51 → %17 |
| 3 | Derinlik ve füzyon | Nesne başına göreli mesafe, sıralama doğru |
| 4 | Kuşbakışı projeksiyon | Araç genişliği 1.89 m (hedef 1.80) |
| 5 | Hız ve TTC | TTC ölçek-değişmez olduğu ispatlandı |
| 6 | Şerit ve sürülebilir alan | Hafta 4'ün şerit borcu kapandı |
| 7 | Yayınlama | CPU'da 342 → 121 ms/kare |
| 8 | Hata analizi | Dört ölçüm, bir düzeltme |


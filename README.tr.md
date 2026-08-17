# Tek Kameradan Sürüş Algısı ve Kuşbakışı Haritalama

English version: [README.md](README.md)

**Tek bir öne bakan kameradan** alınan görüntüyü işleyip şu soruları yanıtlayan
bir algı sistemi:

- Sahnede hangi nesneler var? (araç, yaya, bisikletli, trafik ışığı)
- Bu nesneler kareler arasında nasıl hareket ediyor? (takip)
- Araca ne kadar uzaklıktalar? (monoküler derinlik)
- Sürülebilir alan ve şeritler nerede? (segmentasyon)
- Çarpışma riski var mı, ne kadar süre kaldı? (TTC)

Nihai çıktı iki panelli bir video: solda işlenmiş kamera görüntüsü, sağda
araçların ve şeritlerin üstten gösterildiği dinamik harita.

> **Durum: 8 haftanın 1. haftası — tespit katmanı.** Bu README hedeflenen
> sistemi anlatıyor; bugün gerçekten ne çalışıyor için [Yol haritası](#yol-haritası)
> bölümüne bak.

---

## Neden bu proje?

Otonom sürüş algı yığını, bilgisayarlı görünün hemen tüm temel problemlerini tek
çatı altında barındırır: tespit, takip, derinlik, segmentasyon ve geometrik
projeksiyon. Tek proje, geniş bir yetkinlik yüzeyi.

Çıktısı da doğrudan görsel — projeyi inceleyen kişi tek satır kod okumadan
10 saniyede ne yapıldığını anlıyor.

---

## Mimari

```
                          Video karesi
                               |
        +----------------------+----------------------+
        v                      v                      v
  +-----------+        +--------------+        +--------------+
  |  Tespit   |        |   Derinlik   |        | Segmentasyon |
  |  (YOLO)   |        | (Depth Any.) |        | (şerit/zemin)|
  +-----+-----+        +------+-------+        +------+-------+
        |                     |                       |
        v                     |                       |
  +-----------+               |                       |
  |   Takip   |               |                       |
  |(ByteTrack)|               |                       |
  +-----+-----+               |                       |
        |                     |                       |
        +----------+----------+-----------------------+
                   v
            +--------------+
            |    Füzyon    |
            |(kutu+derinlik|
            +------+-------+
                   |
        +----------+----------+
        v                     v
  +-----------+        +--------------+
  | Kuşbakışı |        |     Risk     |
  | projeksiyon        |  (hız, TTC)  |
  +-----+-----+        +------+-------+
        |                     |
        +----------+----------+
                   v
             Çıktı videosu
```

### Katmanlar

| Katman | Görevi |
|---|---|
| **Tespit** | Ön-eğitimli YOLO nesneleri sınırlayıcı kutularla işaretler. COCO sınıfları yeterli, ayrı eğitim yok. |
| **Takip** | ByteTrack her nesneye kareler boyunca kalıcı bir kimlik verir. Hız ve TTC hesabı bu kimlik sürekliliğine dayandığı için takibin kararlılığı projenin en kritik teknik noktası. |
| **Derinlik** | Depth Anything her piksel için göreli derinlik üretir. Nesnenin derinliği, kutusunun alt orta bölgesindeki değerlerin **medyanı** — ortalama değil, birkaç aykırı piksel tahmini kaydırmasın diye. |
| **Füzyon** | Kutu ve derinlik birleştirilir; takip edilen her nesne için `(kimlik, sınıf, görüntü konumu, derinlik)` dörtlüsü çıkar. |
| **Projeksiyon** | Kutunun alt kenarı nesnenin zemine değdiği nokta kabul edilir ve homografi ile zemin düzlemine yansıtılır. |
| **Risk** | Takip geçmişinden göreli hız çıkarılır; mesafe / yaklaşma hızı oranından TTC hesaplanır. Eşiğin altındaki nesneler vurgulanır. |

---

## Kurulum

Python 3.10+ gerekir.

```bash
git clone <repo-url> otonomarac
cd otonomarac
python -m venv .venv
```

Ortamı aktifleştir — Windows'ta `.venv\Scripts\activate`, diğerlerinde
`source .venv/bin/activate` — sonra **önce** donanımına uygun PyTorch'u kur:

```bash
# NVIDIA Pascal GPU (GTX 10xx) veya 527'den eski sürücü:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Güncel sürücülü yeni NVIDIA GPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# GPU yok:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

```bash
pip install -r requirements.txt
pip install -e .
python scripts/smoke_test.py
```

Model ağırlıkları ilk çalıştırmada otomatik iner.

> **`pip install -e .` Windows'ta `WinError 5` veriyorsa**, setuptools metadata
> yazarken engelleniyordur (genelde antivirüs gerçek zamanlı taraması).
> Kaynakları doğrudan yola ekle — geliştirme açısından aynı işi görür:
>
> ```bash
> python -c "import sysconfig,pathlib; pathlib.Path(sysconfig.get_paths()['purelib'], 'perception_src.pth').write_text(str(pathlib.Path('src').resolve()))"
> ```

---

## Kullanım

```bash
# En basit hali — outputs/result.mp4 üretir
python -m perception.cli --input data/surus.mp4

# İlk 100 kareyle hızlı deneme
python -m perception.cli --input data/surus.mp4 --max-frames 100

# Daha büyük model, yarı hassasiyet, cihazı elle seç
python -m perception.cli --input data/surus.mp4 --model yolov8s.pt --half --device cuda

# Config dosyası; verdiğin her bayrak dosyayı ezer
python -m perception.cli --config configs/default.yaml --input data/surus.mp4
```

Tüm bayraklar için `python -m perception.cli --help`. Her çalıştırma sonunda
katman bazlı süre tablosu ve takip raporu basılır.

### Takip parametre taraması

Takip ayarları tahmin edilmedi, ölçüldü:

```bash
python scripts/tracking_sweep.py data/surus.mp4 --config configs/default.yaml
```

Her deney bazdan tek eksende ayrılıyor, böylece rakamlardaki fark bir
değişikliğe atfedilebiliyor. Betik benzersiz kimlik sayısı, medyan iz uzunluğu,
parçalanma oranı ve hız raporluyor. Sonuçlar ve varsayılanların gerekçesi
[docs/proje-gunlugu.md](docs/proje-gunlugu.md) içinde.

Ayar yapmadan önce bilinmesi gereken iki bulgu:

- **`detection.conf`, `tracking.track_low_thresh`'in altında kalmalı.**
  ByteTrack'in tüm amacı düşük güvenli tespitleri ikinci bir eşleştirme turunda
  kayıp izleri kurtarmak için kullanmak. Detektör onları önce filtrelerse o tur
  boş kalır ve algoritma sessizce sıradan IoU takibine düşer. Detektör eşiğini
  0.15'e çıkarmak parçalanmayı %44'ten %53'e taşıdı. `Config.validate()` uyarır.
- **`match_thresh` yükseltmek eşleştirmeyi gevşetir, sıkılaştırmaz.** Maliyet
  matrisi `1 - IoU` ve eşik bir üst sınır (`lap.lapjv(cost_limit=…)`), yani
  `0.9` "IoU ≥ 0.1 kabul" demek. 10 Hz kayıtta varsayılan 0.8 fazla dar kalıyor
  ve izleri koparıyor.

### Derinlik göreli, rakamlar da bunu söylüyor

Etiketlerde `~2.0` yazıyor, `2.0 m` değil. Tek kameradan mutlak mesafe
çıkarılamaz ve birim yazmak sistemin sahip olmadığı bir kesinliği iddia ederdi.
Değer birimsiz ve yalnızca nesneleri karşılaştırmak için anlamlı; sabit bir
referansla ölçek kalibrasyonu Hafta 5'in işi.

Saklanan harita modelin **ham** çıktısı, bilinçli olarak normalize edilmiyor.
Her kareyi `[0,1]`'e taşımak daha iyi görünür ama değerleri kareler arası
karşılaştırılamaz kılar — sahneye tek bir yakın nesne girdiğinde tüm haritanın
ölçeği kayar, hiçbir şey hareket etmemiş olsa bile her nesnenin "mesafesi"
değişir. Hız tahmini tam da o kareler arası farka dayanıyor. Normalizasyon
sadece çizim katmanında, sadece renk haritası için yapılıyor.

### Kuşbakışı haritanın kalibrasyonu

Homografi yol düzleminde dört nokta ister; bu noktalar kameraya ve montaja özgü:

```bash
python scripts/calibrate_bev.py data/surus.mp4 --frame 900 \
    --quad 0.414,0.839 0.654,0.839 0.579,0.728 0.493,0.728 --verify
```

`--verify` kalibrasyonu, kalibrasyonda **kullanılmayan** bir şeyle sınar:
tespit edilen araçların zemine yansıtılmış genişliği. İki sayı önemli — medyan
tipik araç genişliğine (~1.8 m) yakın olmalı, ve genişlik mesafeden bağımsız
olmalı. Asıl test ikincisi; sıfırdan uzak bir korelasyon perspektifin doğru
kaldırılmadığını gösterir.

Bu kontrolün ilk sürümü, sarı orta çizginin düzleştirmeden sonra dikey çıkıp
çıkmadığına bakıyordu. O ölçüm döngüseldi — dörtgen zaten o çizgiden
kuruluyordu, sonuç her zaman mükemmel geliyordu ve onlarca farklı ufuk değeri
aynı skoru alıyordu. Hiçbir yapılandırmayı elemeyen bir doğrulama, doğrulama
değildir.

`quad_depth_m` bir **varsayım, ölçüm değil**. 6 m'den 22 m'ye değiştirildiğinde
medyan genişlik de mesafe korelasyonu da değişmiyor, sadece haritadaki bütün
mutlak mesafeler ölçekleniyor. Yanal ölçek bir referansa bağlanabiliyor,
boylamsal ölçek bu testle bağlanamıyor. Monoküler ölçek belirsizliği tam olarak
budur; bir sayının arkasına saklanmak yerine açıkça yazıldı.

### Yavaş donanım için ayarlar

| Bayrak | Etkisi |
|---|---|
| `--frame-stride 2` | Bir kare atlayarak işle; çıktı FPS'i de bölündüğü için oynatma gerçek zamanlı kalır |
| `--resize-width 960` | Kareyi çıkarımdan önce küçült |
| `--imgsz 480` | Model giriş boyutunu düşür |
| `--model yolov8n.pt` | En küçük ağırlıklar (varsayılan) |

---

## Klasör yapısı

```
configs/         YAML ayarları
src/perception/
  config.py      Dataclass config, YAML okuma, CLI ezme
  video_io.py    Kare okuma/yazma, kare atlama, ölçekleme
  detection.py   YOLO sarmalayıcı -> Detection nesneleri
  tracking.py    ByteTrack sarmalayıcı, takip istatistikleri, hareket izleri
  visualize.py   Tüm OpenCV çizimleri
  pipeline.py    Katmanların birleştiği yer
  cli.py         Komut satırı girişi
  utils.py       Cihaz seçimi, katman bazlı süre ölçer
scripts/         Smoke test, KITTI dönüştürme, parametre taraması
data/            Girdi videoları (git dışı)
outputs/         Üretilen sonuçlar (git dışı)
docs/            Proje raporu ve mühendislik günlüğü
```

---

## Yol haritası

| Hafta | Hedef | Durum |
|---|---|---|
| 1 | Depo iskeleti, video I/O, YOLO tespiti | **bitti** — GTX 1080'de 50.7 FPS |
| 2 | ByteTrack, kimlikler, hareket izleri | **bitti** — %17 iz parçalanması |
| 3 | Depth Anything, kutu–derinlik füzyonu | **bitti** — nesne başına göreli derinlik |
| 4 | Homografi, kuşbakışı harita, çift panel | **bitti** — 0.1 ms/kare |
| 5 | Göreli hız, ölçek kalibrasyonu, TTC | |
| 6 | Şerit / sürülebilir alan segmentasyonu | |
| 7 | Gradio arayüzü, Hugging Face Spaces | |
| 8 | Dokümantasyon, hata analizi, performans tablosu | |

Kural: **her hafta sonunda çalışan bir çıktı olacak.** Bir sonraki katmana ancak
mevcut katman uçtan uca çalıştıktan sonra geçilir; böylece proje hangi haftada
durursa dursun elde gösterilebilir bir sonuç kalır.

---

## Kritik kararlar

**Kuşbakışı projeksiyon — iki yol.** Yöntem A yolu düz bir düzlem kabul edip
görüntüdeki dört noktayı zemindeki dört noktaya homografi ile eşler. Derinlik
modeline hiç ihtiyaç duymaz, hızlı ve kararlıdır; ama eğimli ve tümsekli yolda
bozulur. Yöntem B nesneyi derinlik + kamera iç parametreleriyle 3B'ye taşıyıp
üstten izdüşümünü alır; eğimli yolda daha doğrudur, ama monoküler derinlik göreli
olduğu için ölçek belirsizdir, kalibrasyon ister. **Önce A çalışır hale gelecek,
sonra B eklenip ikisi ayrı bir bölümde karşılaştırılacak** — "iki yöntemi
denedim, farkları şunlar" anlatısı tek yöntem sunmaktan çok daha güçlü.

**Ölçek belirsizliği.** Tek kameradan mutlak mesafe ölçülemez; bu monoküler
görünün bilinen temel kısıtı. Ölçek katsayısı sabit bir referansla kalibre
edilecek: şerit genişliği (Türkiye'de tipik 3.5 m) veya tipik araç genişliği
(~1.8 m). Bu varsayım ve getirdiği hata payı gizlenmeyip açıkça yazılacak.

**Model eğitimi neden yok?** Hazır ağırlık kullanmak tembellik değil
önceliklendirme. Bu projede gösterilecek asıl beceri **sistem entegrasyonu,
geometri ve hata analizi**; bir sınıflandırıcıyı yeniden eğitmek değil.

---

## Bilinçli olarak kapsam dışı

Aşağıdakiler ya donanım gerektirdiği ya da projeyi bitirilemez hale getireceği
için dışarıda bırakıldı:

- Gerçek araç üzerinde çalıştırma
- Karar verme / kontrol katmanı (direksiyon, fren komutu)
- LiDAR, radar veya çoklu sensör füzyonu
- Simülatör (CARLA) entegrasyonu
- Sıfırdan model eğitimi
- Metrik (mutlak) derinlik ölçümü

Neyi neden yapmadığını bilmek eksiklik değil, mühendislik olgunluğudur.

---

## Başarı kriterleri

- [ ] Rastgele bir sürüş videosu hatasız işlenip çıktı üretiyor
- [ ] Kuşbakışı harita araçların göreli konumlarını tutarlı gösteriyor
- [ ] Canlı demo linki çalışıyor ve dışarıdan erişilebiliyor
- [ ] README'de sistemin ne yaptığı, nasıl çalıştığı ve **nerede başarısız olduğu** yazılı
- [ ] Performans ölçümleri raporlanmış

Kriterler arasında doğruluk metriği (mAP vb.) bilinçli olarak yok. Bu bir model
eğitimi projesi değil, sistem entegrasyonu projesi; başarısı çalışıp
çalışmadığıyla ve anlatımının netliğiyle ölçülür.

---

## Lisans

MIT

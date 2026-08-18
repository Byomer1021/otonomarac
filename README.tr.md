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

![Maltepe'de çekilen dashcam görüntüsü üzerinde tespit, takip ve kuşbakışı projeksiyon](docs/assets/demo.gif)

*Kendi dashcam kaydımız, Maltepe/İstanbul. Solda: kalıcı kimlikli tespitler,
hareket izleri ve göreli derinlik. Sağda: homografiyle çıkarılan zemin
konumları. Haritadaki mesafeler aracın burnundan değil kalibrasyon referans
satırından ölçülüyor — bkz. [Kuşbakışı haritanın kalibrasyonu](#kuşbakışı-haritanın-kalibrasyonu).*

> **Durum: tamamlandı.** Sekiz haftanın hepsi bitti — tespit, takip, derinlik,
> kuşbakışı projeksiyon, TTC ve sürülebilir alan segmentasyonu uçtan uca
> çalışıyor, demo yayında, ve [Nerede başarısız oluyor](#nerede-başarısız-oluyor)
> neyin ne kadar bozulduğunu ölçüyor.

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

## Canlı demo

Gradio arayüzü tüm yığını yüklenen bir klip üzerinde çalıştırır:

```bash
pip install -r requirements-app.txt
python app.py
```

Ücretsiz CPU katmanı için tasarlandı, gerçek zamanlı olma iddiası yok: klip
girer, ilk 15 saniyesi çevrimdışı işlenir, işaretlenmiş video ve ölçüm raporu
döner.

**CPU ayarları tahmin değil ölçüm.** `yolov8n` ve 480 px'e inmek, 960 px
genişlikte işlemek, derinliği 5, segmentasyonu 10 karede bir hesaplamak
pipeline'ı 342 ms/kareden 121 ms'e indiriyor — 15 saniyelik klip yaklaşık bir
dakikada bitiyor. Kaydedilmeye değer bir sonuç: torch'u altı yerine iki iş
parçacığına sınırlamak neredeyse hiçbir şeyi değiştirmedi, yani bu model
boyutlarında darboğaz paralel hesap değil işlem başı ek yük. Ayarların tamamı
ve gerekçesi: [configs/spaces.yaml](configs/spaces.yaml).

**Yüklenen video için kuşbakışı harita kapalı.** Homografi, yol düzleminde
kameraya ve montaja özgü dört nokta ister; bir yabancının klibi için o bilgi
yok. Makul görünen varsayılan bir dörtgen koymak, ölçülmüş gibi okunan ama
ölçülmemiş mesafeler üretirdi. Tespit, takip ve derinlik kameradan bağımsız
çalışır; depodaki Maltepe örneğinin gerçek kalibrasyonu olduğu için haritayı da
gösterir.

### Hugging Face Spaces'e yükleme

Bir kez giriş yap, sonra betiği çalıştır:

```bash
hf auth login                      # token: huggingface.co/settings/tokens (Write)
python scripts/deploy_space.py     # <kullanici>/otonomarac olusturur ve yukler
```

`--dry-run` yüklemeyi Hub'a dokunmadan hazırlar; tam olarak neyin gideceğini
görebilirsin (20 dosya, 1.8 MB — kaynaklar, config'ler, arayüz ve örnek klip;
veri ve çıktı yok).

Betik Space'in `requirements.txt` dosyasını kopyalamak yerine **üretiyor**, iki
sebeple. Depodaki dosya torch'u bilinçli olarak dışarıda bırakıyor çünkü doğru
tekerlek makineye bağlı; olduğu gibi bırakılsaydı Space'te pip torch'u
ultralytics üzerinden çözer ve **CUDA** yapısını çekerdi — CPU-only bir
konteynere birkaç gigabayt. Üretilen dosya CPU tekerleğini açıkça istiyor.
Ayrıca `sdk_version` yerelde kurulu gradio'dan yazılıyor, böylece Space
uygulamanın gerçekten test edildiği sürümle ayağa kalkıyor; elle tutulan bir
sürüm numarası kaçınılmaz olarak kayar.

Model ağırlıkları ilk çalıştırmada indiği için ilk istek yukarıdaki
sürelerden uzun sürer.

> **Spaces fiyatlandırması 2026'da değişti.** Gradio Space'ini ücretsiz
> `cpu-basic` katmanında barındırmak artık PRO abonelik istiyor; ücretsiz
> hesaplara statik Space'ler ve ZeroGPU kalıyor. Betik varsayılan olarak
> `cpu-basic` kullanıyor; ücretsiz GPU katmanı için `--hardware zero-a10g`
> vermek gerekiyor, o da `app.py`'de `@spaces.GPU` dekoratörü istiyor.


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
| 5 | Göreli hız, ölçek kalibrasyonu, TTC | **bitti** — TTC ölçek-değişmez |
| 6 | Şerit / sürülebilir alan segmentasyonu | **bitti** — haritada görülen serbest alan |
| 7 | Gradio arayüzü, Hugging Face Spaces | **bitti** — CPU'da 342 → 121 ms/kare |
| 8 | Dokümantasyon, hata analizi, performans tablosu | **bitti** |

Kural: **her hafta sonunda çalışan bir çıktı olacak.** Bir sonraki katmana ancak
mevcut katman uçtan uca çalıştıktan sonra geçilir; böylece proje hangi haftada
durursa dursun elde gösterilebilir bir sonuç kalır.

---

## Nerede başarısız oluyor

İddia değil ölçüm — [`scripts/failure_analysis.py`](scripts/failure_analysis.py)
buradaki her rakamı üretiyor, tamamı [docs/hata-analizi.md](docs/hata-analizi.md)
içinde. Aynı kameranın yoğun şehir klibi ile neredeyse boş kırsal klibi yan yana
konuyor.

Herhangi bir çıktıya güvenmeden önce bilinmesi gereken dört bulgu:

**Projeksiyon uzakta bozuluyor.** Bir izin kareler arası mesafe gürültüsü yakın
banttan uzağa 3.7 kat büyüyor: 0-8 m'de medyan 3.9 m/s, 25-40 m'de 14.6 m/s ve
90. yüzdelik 54 m/s — saatte 196 km, şehir içinde hiçbir şey böyle hareket
etmiyor. Sebep geometrik: zemin 40 m görüntüde satır 497.6, 60 m ise 492.3, yani
beş piksellik şerit yirmi metreyi kaplıyor. Haritada 25 m ötesi gösterge
amaçlıdır, ölçüm değil.

**Örtülme zemine değme varsayımını kırıyor.** Sistem kutunun alt kenarını
nesnenin yere değdiği yer sayıyor. Nesne kısmen örtülünce görünen alt kenar
gerçek temas noktasının üstünde kalıyor ve homografi aracı olduğundan uzağa
koyuyor. Örtülü kutuların mesafe gürültüsü örtüşmeyenlerin iki katı (medyan 10.0
karşı 5.2 m/s). Fit kalitesi kapısı bunun ürettiği sahte uyarıları bastırıyor,
bedeli gerçek uyarının 39 gözlemden 21'e inmesi. Mesafenin kendisi hâlâ yanlış,
yalnızca alarm susturuluyor.

**Boş sahne kaputu hayalet araca çeviriyor.** `detection.conf` 0.05'te, çünkü
ByteTrack'in ikinci eşleştirme turu buna ihtiyaç duyuyor. Karede gerçek nesne
yokken aynı eşik kaputun yansımalarını araba olarak etiketliyor — kırsal klipte
tespitlerin %71'i, medyan güven 0.13; şehirde %4. Kutusunun yarısından fazlası
kaput çizgisinin altında kalan tespitler artık takibe girmiyor: kırsalda ham
tespit 890'dan 330'a düşüyor, kullanılabilir zemin konumu %20'den %55'e
çıkıyor, şehir sahnesi %2 oynuyor.

**Takip zorluğu nesne sayısından değil örtülmeden geliyor.** Kırsal klipte
şehrin 110 izine karşı 5 iz var, ama delikli iz oranı %25'e karşı sıfır —
arkasına saklanacak bir şey olmadığı için hiçbir iz kaybolup geri gelmiyor.

Çalışmadığı değil, denenmediği için bilinmeyenler: gece, yağmur ve sis (çekimde
yok), ve eğimli yol (Maltepe düz, düzlem varsayımı hiç zorlanmadı).

### Performans

Kare başına, 1280 px genişlik, tespit GTX 1080'de, iki ağır model CPU'da:

| Aşama | ms | Not |
|---|---|---|
| derinlik | 117.0 | CPU, 3 karede bir |
| segmentasyon | 57.2 | CPU, 10 karede bir |
| tespit | 20.4 | GPU |
| çizim | 9.5 | iki panel |
| takip | 2.8 | |
| füzyon | 0.8 | |
| risk | 0.4 | |
| projeksiyon | 0.1 | |
| **uçtan uca** | **220.8** | **4.5 FPS** |

Ücretsiz CPU katmanında ayarlanmış profille: 121 ms/kare, ~8 FPS. Isınma hariç —
ilk CUDA çıkarımı 4 saniye sürüyor ve dahil edilseydi Hafta 1 pipeline'ı
50.7 yerine 22.4 FPS raporlanacaktı.

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

**Ölçek belirsizliği ve TTC'nin ondan kaçışı.** Tek kameradan mutlak mesafe
ölçülemez. Başlangıçtaki plandan daha kötüsü: şerit veya araç genişliği gibi
bir **yanal** referans, boylamsal ölçeği hiçbir zaman sabitleyemez — odak
uzaklığı yanal bağıntıda sadeleşir, derinlik bağıntısında sadeleşmez:

```
X = (u − u_c) · h / (v − v_h)      ← f sadeleşir
Z = f · h / (v − v_h)              ← f kalır
```

Yani yanal kalibrasyon kamera yüksekliğini veriyor — 107 araç örneğinde
1.43 m, ön cam montajı için doğru aralık — ama `bev.quad_depth_m` varsayım
olarak kalıyor. Haritadaki metre değerleri bir katsayı kadar belirsiz.

**Çarpışmaya kalan süre değil.** Bütün mesafeleri bilinmeyen bir *k* ile
çarpın, yaklaşma hızı da aynı *k* ile çarpılır; `TTC = d / v` değişmez.
Ölçüldü: `quad_depth_m` iki katına çıkınca mesafe de iki katına çıkıyor
(1.3 → 2.6 → 5.2 m) ama TTC üçünde de 1.29 s. Projenin en çok işe yarayan
çıktısı, en zayıf varsayımından bağımsız çıktı.

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

- [x] Rastgele bir sürüş videosu hatasız işlenip çıktı üretiyor
- [x] Kuşbakışı harita araçların göreli konumlarını tutarlı gösteriyor — kareden
      başka hiçbir girdiyi paylaşmayan monoküler derinlikle çapraz doğrulandı
- [x] Canlı demo linki çalışıyor ve dışarıdan erişilebiliyor —
      [huggingface.co/spaces/byomer1021/otonomarac](https://huggingface.co/spaces/byomer1021/otonomarac)
- [x] README'de sistemin ne yaptığı, nasıl çalıştığı ve **nerede başarısız olduğu** yazılı
      — [Nerede başarısız oluyor](#nerede-başarısız-oluyor), ölçümler [docs/hata-analizi.md](docs/hata-analizi.md)
- [x] Performans ölçümleri raporlanmış — katman bazlı, ısınma hariç

Kriterler arasında doğruluk metriği (mAP vb.) bilinçli olarak yok. Bu bir model
eğitimi projesi değil, sistem entegrasyonu projesi; başarısı çalışıp
çalışmadığıyla ve anlatımının netliğiyle ölçülür.

---

## Lisans

MIT

# Hata Analizi

Ön rapordaki başarı kriterlerinden biri şuydu: *"README'de sistemin ne yaptığı,
nasıl çalıştığı ve **nerede başarısız olduğu** yazılı."*

Bu belge o maddeyi karşılıyor. İçindeki her rakam ölçüm, iddia değil:

```bash
python scripts/failure_analysis.py --frames 400
```

İki sahne karşılaştırılıyor: **şehir içi** (Maltepe, yoğun trafik) ve **kırsal**
(aynı çekimin 72. dakikası, neredeyse boş yol). Aynı kamera, aynı ayarlar.

---

## 1. Bilgi hunisi — sistem neyi kaybediyor

| Adım | Şehir içi | Kırsal |
|---|---|---|
| Ham tespit | 10.322 (100%) | 330 (100%) |
| Kimlik atandı | 4.931 (48%) | 247 (75%) |
| Zemin konumu var | 4.749 (46%) | 180 (55%) |
| **TTC üretildi** | **162 (2%)** | **6 (2%)** |

Okunması gereken satır sonuncusu: **ham tespitlerin yalnızca %2'si bir TTC
değerine dönüşüyor.** Bu bir kayıp değil, tasarım. Aradaki elemeler:

- **%52'si kimlik alamıyor** — `detection.conf` ByteTrack'in ikinci eşleştirme
  turu için 0.05'te tutuluyor; o bandın büyük kısmı zaten iz başlatmaya aday
  değil, kaybolan izleri kurtarmak için var.
- **Kalanın çoğu güzergâh koridorunun dışında** — yol kenarındaki park etmiş
  araçlar. Ego araç hareket ettiği için hepsi teknik olarak "yaklaşıyor";
  koridor kapısı olmadan 104 izin 57'si kritik çıkıyordu.
- **Geri kalanı fit kalitesi kapısına takılıyor** — mesafesi düzgün bir doğru
  çizmeyen izden hız çıkarılmıyor.

Her eleme kasıtlı ve gerekçeli. Sistem üretebileceği sayıların çoğunu
**bilerek üretmiyor.**

---

## 2. Mesafe kararlılığı — projeksiyon uzakta bozuluyor

Ardışık iki gözlem arasındaki mesafe değişim hızı. Duran bir nesnede bu değer
ölçüm gürültüsüdür; büyümesi projeksiyonun o mesafede güvenilmez olduğunu
gösterir.

**Şehir içi:**

| Band | Örnek | Medyan (m/s) | 90. yüzdelik |
|---|---|---|---|
| 0-8 m | 1.634 | **3.9** | 11.8 |
| 8-15 m | 1.542 | 6.1 | 21.3 |
| 15-25 m | 830 | 9.3 | 35.2 |
| 25-40 m | 359 | **14.6** | **54.4** |

Gürültü mesafeyle **3.7 kat** büyüyor. 25-40 m bandında 90. yüzdelik 54.4 m/s —
saatte 196 km. Hiçbir şehir içi nesnesi böyle hareket etmiyor; o rakam hareket
değil, ölçümün kendisi.

Sebep geometrik ve kaçınılmaz: homografi görüntü satırını zemin mesafesine
çeviriyor ve ufka yaklaştıkça bu dönüşüm patlıyor. Ölçüldü — bu kalibrasyonda
zemin **40 m** görüntüde satır **497.6**, **60 m** ise **492.3**. Beş piksellik
bir şerit yirmi metreyi kaplıyor, yani tek piksellik bir kutu titremesi dört
metrelik mesafe hatası demek.

**Pratik sonuç:** haritadaki 25 m ötesi gösterge amaçlıdır, ölçüm değil.
`bev.max_range_m` 40 m'de duruyor ve bu sayı keyfi değil, bu tablodan geliyor.

---

## 3. Örtülme — zemine değme varsayımının kırılma noktası

Sistemin merkezinde bir varsayım var: **kutunun alt kenarı, nesnenin yere
değdiği yerdir.** Nesne kısmen örtüldüğünde bu varsayım çöker — görünen alt
kenar gerçek temas noktasının üstünde kalır ve homografi aracı olduğundan
uzağa koyar.

| Sahne | Durum | Örnek | Medyan (m/s) | 90. yüzdelik |
|---|---|---|---|---|
| Şehir içi | örtüşmeyen | 3.950 | 5.2 | 21.7 |
| Şehir içi | **örtülü** | 521 | **10.0** | **45.7** |

Örtülme gürültüyü **iki katına** çıkarıyor. Hafta 5'te tekil bir örnekte
görülmüştü: Duster'ın arkasındaki bir araç 22 m'de 14 m/s yaklaşıyor
raporlanmıştı, görüntüde ~10 m'deydi. Bu tablo o gözlemin genel olduğunu
gösteriyor.

**Alınan önlem:** hız fitine artık bir kalite kapısı var
(`risk.max_fit_residual_ratio`). Gerçekten yaklaşan nesnenin mesafe-zaman
eğrisi düz bir doğrudur; örtülme artefaktı zıplar. Kapı sahte kritik uyarıyı
eliyor ama **bedeli var**: gerçek uyarının gözlemlerini de 39'dan 21'e
düşürüyor. Sahte kritik uyarı üretmemek daha önemli sayıldı.

**Çözülmedi:** örtülü aracın gerçek mesafesi hâlâ bilinmiyor, sadece o mesafeye
dayanan uyarı bastırılıyor.

---

## 4. Boş sahne — düşük eşiğin bedeli

Kırsal klipte bir şey beklenmedik biçimde bozuldu: **tespitlerin %71'i kaput
bölgesindeydi**, hepsi araç sınıfı, medyan güven **0.13**. Şehirde aynı oran
%4.

Sebep, iki katmanın etkileşimi. `detection.conf` ByteTrack'in ikinci
eşleştirme turu için bilinçli olarak 0.05'te — o zayıf tespitler kaybolan
izleri kurtarmak için gerekli. Ama sahnede gerçek nesne yokken aynı eşik
**kaputun yansımalarını hayalet araca çeviriyor.** Şehirde bu hayaletler
gerçek trafiğin arasında kayboluyor; boş yolda baskın hale geliyorlar.

Haritaya ulaşmıyorlardı — `BEVProjector.project` zaten kaput altındaki temas
noktalarını eliyor — ama takip onlara kimlik harcıyor ve istatistiği
kirletiyordu.

**Alınan önlem:** kutusunun yarısından fazlası kaput çizgisinin altında kalan
tespitler takibe hiç girmiyor. Ölçüt "yarı", "tamamı" değil: ilk sürüm kutunun
tamamının altta olmasını arıyordu ve neredeyse hiçbir şeyi elemedi, çünkü
hayaletlerin üst kenarı genelde çizginin biraz üstünde başlıyor.

Yarım eşiği güvenli, çünkü zemine değme noktası kaputla örtülen **gerçek** bir
aracın kutusu çizgiye dayanır, yarısını geçmez.

| Ölçüm | Önce | Sonra |
|---|---|---|
| Kırsal ham tespit | 890 | **330** |
| Kırsal zemin konumu üretilen | %20 | **%55** |
| Kırsal kaput elemesi | 370 | 66 |
| Şehir ham tespit | 10.578 | 10.322 |
| Şehir benzersiz kimlik | 111 | 110 |
| Şehir parçalanma | %17 | %17 |

Hayaletleri eliyor, gerçek nesnelere dokunmuyor.

---

## 5. Sahne bağımlılığı

| Ölçüm | Şehir içi | Kırsal |
|---|---|---|
| Benzersiz kimlik | 110 | 5 |
| Eş zamanlı iz (ort) | 12.3 | 0.5 |
| Medyan iz uzunluğu | 24 kare | 33 kare |
| Parçalanma | %17 | %40 |
| Delikli iz | %25 | **%0** |

Kırsalda parçalanma daha yüksek görünüyor ama sayı yanıltıcı: **5 izin 2'si**
kısa. Buna karşılık delikli iz **hiç yok** — sahnede birbirini örtecek nesne
olmadığı için hiçbir iz kaybolup geri gelmiyor. İki tablo birlikte okunduğunda
ortaya çıkan şey şu: **projenin takip zorluğu nesne sayısından değil,
nesnelerin birbirini örtmesinden geliyor.**

---

## 6. Hava koşulu — yağmur ve sis

İkinci bir çekim eklendi: **İstanbul, sağanak yağmur** (3840×2160, 18.6 dakika).
Farklı kadraj — kaput görüntünün alt %3'ünü kaplıyor, Maltepe'de %15 — bu yüzden
homografi taşınmıyor ve o klipte kuşbakışı harita ile TTC ölçülmüyor. Tespit ve
takip kameradan bağımsız çalıştığı için karşılaştırılabiliyor.

| Sahne | nesne/kare | medyan güven | >0.5 oranı | kimlik | medyan iz | parçalanma | delikli |
|---|---|---|---|---|---|---|---|
| kuru şehir | 12.4 | 0.63 | 66% | 88 | 21 | 19% | 23% |
| kuru kırsal | 0.7 | 0.87 | 76% | 3 | 44 | 33% | 0% |
| yağmur, açık yol | 8.0 | **0.75** | 80% | 53 | 16 | 19% | **42%** |
| yağmur, sisli | 6.6 | **0.57** | 66% | 65 | **9** | **31%** | 25% |

**Yağmurun kendisi tespiti bozmuyor.** Açık yolda medyan güven 0.75 — kuru şehir
sahnesinden (0.63) daha yüksek. Sebep hava değil sahne: otoyolda araçlar büyük
ve seyrek, şehirde küçük ve iç içe.

**Görüş mesafesi bozuyor.** Kontrollü karşılaştırma son iki satır: aynı kamera,
aynı sürüş, tek değişen sis. Medyan güven **0.75 → 0.57**, parçalanma
**%19 → %31**, medyan iz uzunluğu **16 → 9 kare** — izler yarı yarıya kısalıyor.

**Yağmurun asıl bedeli süreklilikte.** Açık yağmurlu yolda delikli iz oranı
**%42**, kuru şehirde %23. Tespit güveni yüksek olmasına rağmen izler kaybolup
geri geliyor. Muhtemel sebepler: silecek geçişleri, cama düşen damlalar, ıslak
zeminden gelen yansımalar. Bu, TTC hesabını doğrudan etkiler — hız çıkarımı
gözlemler arası boşluğa duyarlı.

**Ders:** "yağmurda çalışır mı" yanlış soru. Yağmur tespiti bozmuyor, **görüş
mesafesi** bozuyor ve **süreklilik** bozuluyor. İkisi farklı katmanları vuruyor.

---

## Bilinen ve çözülmemiş kısıtlar

- **Boylamsal ölçek belirsiz.** Yanal ölçek araç genişliğiyle kalibre edildi
  (kamera yüksekliği 1.43 m çıktı, ön cam montajı için doğru aralık) ama
  boylamsal ölçek görüntüden belirlenemiyor — odak uzaklığı yanal bağıntıda
  sadeleşiyor, derinlik bağıntısında sadeleşmiyor. Haritadaki metre değerleri
  bir katsayı kadar belirsiz. **TTC bundan etkilenmiyor**, çünkü mesafe ve hız
  aynı katsayıyla ölçekleniyor.

- **Haritanın orijini aracın burnu değil**, kalibrasyon dörtgeninin yakın
  kenarı. Aradaki mesafe ölçülmedi. TTC bu yüzden gerçek çarpışmadan biraz
  önce sıfırlanır — güvenli yönde bir hata ama ölçülmemiş bir kaydırma.

- **Koridor kapısı uzakta güvenilmez.** Yanal hata mesafeyle büyüdüğü için
  20 m ötedeki bir nesnenin güzergâhta olup olmadığı kesin değil. Bölüm 2'deki
  tablo bunun büyüklüğünü veriyor.

- **Gece test edilmedi.** Yağmur ve sis ölçüldü (bölüm 6) ama elimizdeki iki
  çekimin ikisi de gündüz. Gece kaydı alınırsa aynı tablo üretilebilir.

- **Yağmur kaydında kuşbakışı ölçülmedi.** O kamera kalibre edilmedi; taşınan
  yalnızca tespit ve takip metrikleri. Homografi çıkarılırsa harita ve TTC de
  karşılaştırılabilir.

- **Yol düzlem varsayılıyor.** Homografi eğim ve tümsekte bozulur. Maltepe
  çekiminde belirgin eğim yok, yani bu kısıt **test edilmedi** — çalışmadığı
  değil, denenmediği için bilinmiyor.

- **Gerçek zamanlı değil.** Yerelde 4.5 FPS, ücretsiz CPU'da ~8 FPS. Tasarım
  baştan çevrimdışı işleme üzerineydi.

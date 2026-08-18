"""Pipeline yapilandirmasi.

YAML dosyasindan okunur, CLI argumanlariyla ezilebilir. Her hafta yeni bir
katman eklendikce buraya yeni bir alt-config dataclass'i eklenecek
(TrackingConfig, DepthConfig, BEVConfig, RiskConfig ...) ve `Config` icine
alan olarak yazilacak - baska bir yeri degistirmek gerekmez.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class VideoConfig:
    """Video giris/cikis ve on-isleme ayarlari."""

    input: str | None = None
    output: str = "outputs/result.mp4"
    #: Kac karede bir isleme alinacak (1 = her kare). CPU'da hizlandirmak icin.
    frame_stride: int = 1
    #: Sadece ilk N kareyi isle (hizli deneme icin). None = tamami.
    max_frames: int | None = None
    #: Isleme oncesi kareyi bu genislige olcekle (en-boy orani korunur). None = dokunma.
    resize_width: int | None = 1280


@dataclass
class DetectionConfig:
    """YOLO tespit katmani ayarlari."""

    model: str = "yolov8n.pt"
    #: Kasitli olarak dusuk. ByteTrack ikinci eslestirme asamasinda dusuk
    #: guvenli tespitleri kayip izleri kurtarmak icin kullanir; detektor
    #: burada agresif filtrelerse o asama bos kalir. Ciktiyi asil temizleyen
    #: esik `tracking.new_track_thresh` (0.25). Takip kapaliyken bu degeri
    #: yukseltmek gerekir - bkz. `--no-track`.
    conf: float = 0.05
    iou: float = 0.5
    imgsz: int = 640
    #: "auto" | "cpu" | "cuda" | "cuda:0"
    device: str = "auto"
    #: COCO sinif id'leri: person, bicycle, car, motorcycle, bus, truck, traffic light
    classes: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 5, 7, 9])
    #: Yari hassasiyet (GPU'da hizlandirir, CPU'da yok sayilir)
    half: bool = False


@dataclass
class CameraConfig:
    """Kaynak kameraya ozgu geometri bilgileri.

    Hafta 4'te homografi kaynak noktalari da buraya eklenecek.
    """

    #: Kaputun/torpidonun basladigi satir, goruntu yuksekliginin orani olarak.
    #: Bu satirin altindaki pikseller yol degil arac ici - derinlik orneklemesi
    #: ve zemine degme noktasi hesabi bu bolgeyi gormemeli.
    #: Maltepe cekimi icin olculen deger: 915/1080 = 0.85. None = kaput yok.
    hood_top: float | None = None

    #: Yol duzlemi uzerinde secilmis dort nokta - homografinin kaynagi.
    #: Sira: yakin-sol, yakin-sag, uzak-sag, uzak-sol (saat yonu).
    #: Piksel degil ORAN olarak saklanir (genislik/yukseklik kesri), boylece
    #: isleme cozunurlugu degistiginde kalibrasyon bozulmaz.
    #: Bu dortgen gercek dunyada bir dikdortgen kabul edilir; kenar
    #: uzunluklari BEVConfig.quad_width_m / quad_depth_m ile verilir.
    road_quad: list[list[float]] | None = None


@dataclass
class DepthConfig:
    """Monokuler derinlik tahmini (Depth Anything V2).

    Model GORELI ters derinlik (disparity) uretir: buyuk deger = yakin.
    Mutlak mesafe tek kameradan olculemez; olcek Hafta 5'te sabit bir
    referansla kalibre edilecek.
    """

    enabled: bool = True
    #: Small / Base / Large - hiz ve dogruluk arasindaki secim
    model: str = "depth-anything/Depth-Anything-V2-Small-hf"
    device: str = "auto"
    half: bool = False
    #: Modele verilecek tensor genisligi (14'un katina yuvarlanir).
    #: Modelin maliyetini belirleyen asil ayar bu: 518 -> 321 ms, 294 -> 109 ms
    #: (CPU olcumu). Nesne basina derinlik zaten kutu icinde medyan alinarak
    #: cikarildigi icin cozunurluk dusurmenin dogruluga bedeli sinirli.
    input_width: int = 518
    #: Derinligi kac karede bir yeniden hesapla (1 = her kare).
    #: Aradaki karelerde son harita yeniden kullanilir. Derinlik sahnede
    #: tespit ve takipten cok daha yavas degisir, bu yuzden 2-3 gorsel olarak
    #: fark ettirmeden GPU yukunu bolüyor - bu makinedeki karti ayakta tutmak
    #: icin onemli (bkz. proje gunlugu, 18 Agustos).
    every_n_frames: int = 1
    #: Ornek bolgesi: kutunun ortadaki %50 genisligi, alt %25 yuksekligi.
    #: Alt orta bolge secilir cunku nesnenin zemine degdigi yer orasidir;
    #: kutunun ust kismi arkadaki sahneyi de kapsar.
    sample_width_ratio: float = 0.5
    sample_height_ratio: float = 0.25
    #: Derinlik haritasini yan panel olarak goster
    show_panel: bool = True
    colormap: str = "INFERNO"


@dataclass
class TrackingConfig:
    """ByteTrack takip katmani ayarlari.

    Esik degerleri Ultralytics'in bytetrack.yaml varsayilanlariyla ayni;
    burada tekrar edilmelerinin sebebi tek bir config dosyasindan yonetilmeleri.
    """

    enabled: bool = True
    #: Birinci asama eslestirme esigi. Yukseltmek izleri temizler, kopmayi artirir.
    track_high_thresh: float = 0.25
    #: Ikinci asama - ByteTrack'in ayirt edici ozelligi. Bu bandin altindaki
    #: tespitler atilmaz, kaybolan izleri kurtarmak icin kullanilir.
    track_low_thresh: float = 0.1
    #: Eslesmeyen tespit bu degerin uzerindeyse yeni iz baslatilir.
    new_track_thresh: float = 0.25
    #: Kaybolan izin kac kare hayatta tutulacagi (okluzyon toleransi).
    track_buffer: int = 30
    #: IoU tabanli eslestirme esigi. DIKKAT - sezgiye aykiri: maliyet matrisi
    #: `1 - IoU` ve esik bir ust sinir (`lap.lapjv(cost_limit=...)`), yani bu
    #: degeri YUKSELTMEK eslestirmeyi GEVSETIR. 0.9 => IoU >= 0.1 kabul edilir.
    #: ByteTrack varsayilani 0.8; scripts/tracking_sweep.py olcumunde 0.9'a
    #: cikarmak parcalanmayi %44'ten %24'e dusurdu (KITTI 10 Hz, kareler arasi
    #: yer degistirme buyuk oldugu icin dar esik izleri koparıyordu).
    match_thresh: float = 0.9
    #: Tespit skorunu eslestirme maliyetine karistir.
    fuse_score: bool = True
    #: Hareket izinin kac saniyelik gecmisi gosterecegi (0 = iz cizme).
    #: Kare degil saniye cinsinden, cunku ayni kare sayisi 10 Hz KITTI'de 3
    #: saniyelik dev bir supurme izi, 30 fps dashcam'de 1 saniyelik kisa bir iz
    #: demek. Saniye sabitlenince gorsel yogunluk kaynaktan bagimsiz kalir.
    trail_seconds: float = 1.5
    #: Bu kareden kisa izler analizde "parcalanmis" sayilir.
    min_track_len: int = 5


@dataclass
class BEVConfig:
    """Kusbakisi (bird's eye view) projeksiyon ayarlari.

    Yontem A - homografi: yol duz bir duzlem kabul edilir ve goruntudeki dort
    nokta zemin duzlemindeki karsiliklariyla eslestirilir. Derinlik modeline
    hic ihtiyac duymaz, hizli ve kararlidir; yol egimliyse bozulur.
    """

    enabled: bool = True

    #: camera.road_quad'in gercek dunyadaki genisligi. Serit genisligi
    #: referans alinir - Turkiye'de tipik olarak 3.5 m.
    quad_width_m: float = 3.5
    #: camera.road_quad'in gercek dunyadaki derinligi.
    #: DIKKAT: bu bir VARSAYIM. Genislik serit isaretinden olculebiliyor ama
    #: derinlik icin goruntude bilinen bir referans yok. Yanlissa haritadaki
    #: mesafeler tek bir katsayiyla olceklenmis olur - siralamayi ve yanal
    #: konumu bozmaz, mutlak mesafeyi bozar. Hafta 5'te kalibre edilecek.
    quad_depth_m: float = 12.0

    #: Haritada gosterilecek alan (referans satirindan ileri, ve iki yana).
    range_ahead_m: float = 35.0
    range_side_m: float = 12.0
    #: Referans satirinin GERISI. Koordinat orijini kalibrasyon dortgeninin
    #: yakin kenari; ego arac bunun birkac metre gerisinde ve yakindaki
    #: araclarin zemine degme noktasi da bu bolgeye duser. Gosterilmezse
    #: haritanin en dolu kismi kaybolur.
    range_behind_m: float = 7.0
    #: Harita cozunurlugu.
    px_per_m: float = 12.0

    #: Zemin cizgilerinin araligi (metre).
    grid_step_m: float = 5.0
    #: Bu mesafenin otesindeki nesneler haritaya cizilmez - homografi
    #: uzakta hizla bozulur ve orada uretilen konum guvenilmez.
    max_range_m: float = 40.0


@dataclass
class SegmentationConfig:
    """Yol segmentasyonu (SegFormer / Cityscapes) ayarlari."""

    enabled: bool = True
    model: str = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
    device: str = "auto"
    half: bool = False
    #: Modele verilecek genislik (32'nin katina yuvarlanir).
    #: 768 -> 234 ms, 1024 -> 464 ms (CPU olcumu, kaput kesilmis kare).
    input_width: int = 768
    #: Yol sahnede araclardan cok daha yavas degisir; her karede yeniden
    #: hesaplamak gereksiz. Derinlik katmanindaki ayni takas.
    every_n_frames: int = 5
    #: Maskedeki kucuk delikleri kapatan morfoloji cekirdegi (0/1 = kapali).
    close_kernel: int = 5
    #: Serit boyasini haritaya isle.
    show_lane_paint: bool = True


@dataclass
class RiskConfig:
    """Goreli hiz ve carpismaya kalan sure (TTC) ayarlari."""

    enabled: bool = True

    #: Hiz tahmini icin gereken en az gozlem sayisi. Bunun altinda sayi
    #: uretilmez - iki noktadan cikarilan hiz gurultunun kendisidir.
    min_observations: int = 5
    #: Hiz, son bu kadar saniyelik gecmise dogru fit edilerek cikarilir.
    history_seconds: float = 1.2

    #: Bu esigin altindaki TTC "kritik", ikincisinin altindaki "uyari".
    ttc_critical_s: float = 2.0
    ttc_warning_s: float = 4.0
    #: Bundan buyuk TTC gosterilmez - anlamli bir uyari degil.
    ttc_max_s: float = 10.0

    #: Yaklasma hizi bu degerin altindaysa TTC hesaplanmaz. Sifira yakin
    #: hizda TTC sonsuza gider ve kucuk bir olcum gurultusu devasa ya da
    #: negatif sayilar uretir.
    min_closing_speed: float = 0.5

    #: Ego'nun guzergah koridorunun yari genisligi. Yalnizca bu koridordaki
    #: nesneler icin TTC uretilir.
    #: GEREKCESI: ego arac hareket ettigi icin park halindeki her nesne de
    #: "yaklasiyor" gorunur ve gecerli bir TTC uretir. Ilk olcumde 104 izin
    #: 57'si kritik cikti - yol kenarindaki araclari carpisma riski sayan bir
    #: uyari sistemi ise yaramaz. Yanal kapi bunu eliyor.
    path_half_width_m: float = 1.7
    #: Bu mesafenin altindaki nesne icin TTC uretilmez. Referans satirinin
    #: dibindeki nesne yanimizdan geciyor olabilir, onumuze giriyor degil.
    min_distance_m: float = 2.0

    #: Hiz fitinin izin verilen en buyuk RMS artigi, mesafenin orani olarak.
    #: Gercekten yaklasan bir nesnenin mesafe-zaman egrisi duz bir dogrudur;
    #: artik buyukse olculen sey hareket degil gurultudur.
    #: GEREKCESI: kismen ortulu bir aracin zemine degme noktasi gorunmez;
    #: kutunun gorunen alt kenari gercek temas noktasinin ustunde kalir ve
    #: homografi araci oldugundan uzaga koyar. Ortulme kare kare degistigi
    #: icin mesafe ziplar ve devasa bir sahte yaklasma hizi uretir. Olculdu:
    #: Duster'in arkasindaki bir arac 22 m'de 14 m/s yaklasiyor gorundu.
    #: BEDELI: 0.04 esigi o sahte izi eliyor ama gercek uyarinin gozlemlerini
    #: de 39'dan 21'e dusuruyor. Uyari yine tum yaklasma boyunca cikiyor,
    #: sadece daha seyrek. Sahte kritik uyari uretmemek daha onemli sayildi.
    max_fit_residual_ratio: float = 0.04

    #: Risk degerlendirmesi icin gereken en dusuk tespit guveni.
    #: `detection.conf` ByteTrack'in ikinci eslestirme turu icin bilincli
    #: olarak 0.05'te tutuluyor; o zayif kutular takibi ayakta tutmakta ise
    #: yariyor ama kare kare titriyor. Titreyen kutu -> titreyen zemin konumu
    #: -> uydurma yaklasma hizi. Olculdu: 0.16 guvenli bir tespit 24.7 m'de
    #: 15 m/s yaklasiyor gorunup kritik uyari uretti.
    min_confidence: float = 0.35


@dataclass
class VisualizeConfig:
    """Cizim ayarlari."""

    show_conf: bool = True
    show_class_name: bool = True
    show_hud: bool = True
    show_trails: bool = True
    line_thickness: int = 2
    font_scale: float = 0.5


@dataclass
class Config:
    """Tum pipeline yapilandirmasi."""

    video: VideoConfig = field(default_factory=VideoConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    bev: BEVConfig = field(default_factory=BEVConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    visualize: VisualizeConfig = field(default_factory=VisualizeConfig)

    def validate(self) -> list[str]:
        """Bolumler arasi tutarsizliklari bulur ve uyari metinleri dondurur.

        Tek bir bolume bakarak gorulemeyen hatalari yakalar - bunlarin en
        onemlisi asagidaki tespit/takip esik catismasidir.
        """
        warnings_found: list[str] = []

        if self.tracking.enabled and self.detection.conf > self.tracking.track_low_thresh:
            warnings_found.append(
                f"detection.conf ({self.detection.conf}) > tracking.track_low_thresh "
                f"({self.tracking.track_low_thresh}): ByteTrack'in dusuk guvenli ikinci "
                f"eslestirme asamasi bos kalir ve algoritma siradan IoU takibine duser. "
                f"Takip acikken detection.conf degerini track_low_thresh'in altina cekin."
            )

        if self.tracking.enabled and self.tracking.track_low_thresh >= self.tracking.track_high_thresh:
            warnings_found.append(
                f"tracking.track_low_thresh ({self.tracking.track_low_thresh}) >= "
                f"track_high_thresh ({self.tracking.track_high_thresh}): ikinci asama devre disi kalir."
            )

        if self.camera.hood_top is not None and not 0.0 < self.camera.hood_top <= 1.0:
            warnings_found.append(
                f"camera.hood_top ({self.camera.hood_top}) 0-1 arasi bir oran olmali "
                f"(goruntu yuksekliginin kesri). Piksel degeri degil."
            )

        if self.depth.enabled and self.camera.hood_top is None:
            warnings_found.append(
                "camera.hood_top ayarlanmamis. Kaput goren bir cekimde kutunun alt "
                "kenari arac icine dusebilir; o bolgenin derinligi yolu degil kaputu "
                "olcer. Kaput gorunuyorsa oranini girin (Maltepe icin 0.85)."
            )

        if self.bev.enabled and self.camera.road_quad is None:
            warnings_found.append(
                "bev.enabled ama camera.road_quad tanimli degil. Homografi icin yol "
                "duzleminde dort nokta gerekiyor; 'python scripts/calibrate_bev.py' "
                "ile secip config'e yazin."
            )
        elif self.camera.road_quad is not None and len(self.camera.road_quad) != 4:
            warnings_found.append(
                f"camera.road_quad tam olarak 4 nokta olmali, {len(self.camera.road_quad)} verildi."
            )

        if self.video.frame_stride > 1 and self.tracking.enabled:
            warnings_found.append(
                f"video.frame_stride={self.video.frame_stride} ile kareler arasi hareket buyur; "
                f"takip eslestirmesi zorlanir. Kimlik atlamasi artarsa track_buffer'i dusurup "
                f"match_thresh'i gevsetmeyi deneyin."
            )

        return warnings_found

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        """YAML dosyasindan config uretir. `path` None ise varsayilanlar kullanilir."""
        if path is None:
            return cls()

        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Config dosyasi bulunamadi: {path}")

        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Config dosyasinin koku bir sozluk olmali: {path}")

        cfg = cls()
        for section_name, values in raw.items():
            section = getattr(cfg, section_name, None)
            if not is_dataclass(section):
                warnings.warn(f"Config'de bilinmeyen bolum yok sayildi: {section_name}", stacklevel=2)
                continue
            if values is None:
                continue
            if not isinstance(values, dict):
                raise ValueError(f"'{section_name}' bolumu bir sozluk olmali, {type(values).__name__} geldi")
            _apply(section, values, section_name)
        return cfg

    def override(self, **kwargs: Any) -> Config:
        """`bolum.alan=deger` bicimindeki duz anahtarlarla config'i gunceller.

        None degerler yok sayilir; boylece CLI'da "arguman verilmedi" durumu
        ayrica kontrol edilmek zorunda kalmaz.
        """
        for dotted, value in kwargs.items():
            if value is None:
                continue
            section_name, _, key = dotted.partition(".")
            section = getattr(self, section_name, None)
            if not key or not is_dataclass(section):
                raise KeyError(f"Gecersiz override anahtari: {dotted}")
            if not hasattr(section, key):
                raise KeyError(f"Bilinmeyen config alani: {dotted}")
            setattr(section, key, value)
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def dump(self, path: str | Path) -> None:
        """Calistirilan config'i diske yazar (tekrar uretilebilirlik icin)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )


def _apply(section: Any, values: dict[str, Any], section_name: str) -> None:
    """Bir alt-config dataclass'inin alanlarini sozlukten gunceller."""
    known = {f.name for f in fields(section)}
    for key, value in values.items():
        if key not in known:
            warnings.warn(f"Config'de bilinmeyen anahtar yok sayildi: {section_name}.{key}", stacklevel=3)
            continue
        setattr(section, key, value)

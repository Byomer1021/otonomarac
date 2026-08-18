"""Gradio arayuzu - Hugging Face Spaces icin (Hafta 7).

Ucretsiz katmanda GPU yok ve gercek zamanli calisma beklenmiyor; arayuz
yuklenen videoyu cevrimdisi isleyip sonucu donduruyor. Ayarlarin gerekcesi
`configs/spaces.yaml` icinde.

Yerel calistirma:
    python app.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from perception.config import Config  # noqa: E402
from perception.pipeline import PerceptionPipeline  # noqa: E402
from perception.video_io import VideoReader, VideoWriter  # noqa: E402

ROOT = Path(__file__).resolve().parent
SPACES_CONFIG = ROOT / "configs" / "spaces.yaml"
EXAMPLE = ROOT / "examples" / "maltepe.mp4"

#: Ucretsiz CPU katmaninda 15 saniyelik klip yaklasik bir dakikada bitiyor.
#: Daha uzunu kullaniciyi tarayici sekmesinde bekletir.
MAX_SECONDS = 15.0

#: Ornek videonun kalibrasyonu. Kendi cekimimiz oldugu icin olculdu;
#: yuklenen videolar icin boyle bir sey YOK - bkz. `_calibration_note`.
EXAMPLE_CAMERA = {
    "hood_top": 0.85,
    "road_quad": [[0.4138, 0.8389], [0.6539, 0.8389], [0.5786, 0.7278], [0.4934, 0.7278]],
}
EXAMPLE_BEV = {"quad_width_m": 3.84, "quad_depth_m": 12.0}


def _trim(source: Path, seconds: float, target: Path) -> Path:
    """Videoyu ilk `seconds` saniyeye kirpar; ffmpeg yoksa oldugu gibi birakir."""
    if shutil.which("ffmpeg") is None:
        return source
    try:
        subprocess.run(
            ["ffmpeg", "-v", "error", "-t", str(seconds), "-i", str(source),
             "-c", "copy", "-y", str(target)],
            check=True, capture_output=True, timeout=120,
        )
        return target if target.exists() and target.stat().st_size > 0 else source
    except (subprocess.SubprocessError, OSError):
        return source


def _build_config(is_example: bool, use_depth: bool, use_segmentation: bool) -> Config:
    config = Config.load(SPACES_CONFIG)
    config.depth = replace(config.depth, enabled=use_depth)
    config.segmentation = replace(config.segmentation, enabled=use_segmentation)

    if is_example:
        # Kusbakisi harita yalnizca kalibrasyonu olcup bildigimiz videoda acilir.
        config.camera = replace(config.camera, **EXAMPLE_CAMERA)
        config.bev = replace(config.bev, enabled=True, **EXAMPLE_BEV)
    return config


def _calibration_note(is_example: bool) -> str:
    if is_example:
        return (
            "**Kuşbakışı harita açık.** Bu örnek videonun kamerası kalibre edildi: "
            "yol düzleminde dört nokta seçilip ölçek, kalibrasyonda kullanılmayan "
            "bir referansla (araç genişliği) doğrulandı."
        )
    return (
        "**Kuşbakışı harita kapalı.** Homografi kameraya ve montaja özgü dört nokta "
        "ister; yüklediğin video için bu bilgi yok. Uydurma bir dörtgenle makul "
        "görünen ama yanlış mesafeler üretmektense harita kapatıldı. Kendi videon "
        "için kalibrasyon: `python scripts/calibrate_bev.py <video> --verify`.\n\n"
        "Tespit, takip ve derinlik kameradan bağımsız çalışır."
    )


def _encode_for_web(source: Path, fps: float) -> Path:
    """Ciktiyi H.264'e cevirir.

    `VideoWriter` mp4v (MPEG-4 Part 2) yaziyor: dosya sisiyor - 10 saniyelik
    klip 18 MB - ve tarayicilarin cogu bu codec'i oynatmiyor, yani Gradio'nun
    oynaticisinda bos bir kutu gorunurdu. ffmpeg yoksa dosya oldugu gibi
    birakiliyor; buyuk ama en azindan indirilebilir.
    """
    if shutil.which("ffmpeg") is None:
        return source

    target = source.with_name("sonuc_web.mp4")
    try:
        subprocess.run(
            ["ffmpeg", "-v", "error", "-r", str(fps), "-i", str(source),
             "-c:v", "libx264", "-crf", "26", "-preset", "veryfast",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-y", str(target)],
            check=True, capture_output=True, timeout=300,
        )
    except (subprocess.SubprocessError, OSError):
        return source
    return target if target.exists() and target.stat().st_size > 0 else source


def process(video, use_depth, use_segmentation, progress=gr.Progress()):
    """Videoyu isler ve (cikti yolu, rapor) dondurur."""
    if not video:
        return None, "Once bir video yukle ya da ornegi sec."

    source = Path(video)
    is_example = source.resolve() == EXAMPLE.resolve()
    workdir = Path(tempfile.mkdtemp(prefix="otonomarac_"))
    clipped = _trim(source, MAX_SECONDS, workdir / "clip.mp4")
    output = workdir / "sonuc.mp4"

    config = _build_config(is_example, use_depth, use_segmentation)
    started = time.perf_counter()

    try:
        progress(0.02, desc="Modeller yukleniyor")
        pipeline = PerceptionPipeline(config)

        with VideoReader(
            clipped,
            frame_stride=config.video.frame_stride,
            resize_width=config.video.resize_width,
        ) as reader:
            meta = reader.output_meta
            if pipeline.tracker is not None:
                pipeline.tracker.configure_for_fps(meta.fps)
            total = meta.frame_count or 1

            with VideoWriter(output, fps=meta.fps) as writer:
                for index, frame in enumerate(reader):
                    writer.write(pipeline.render(pipeline.process_frame(frame)))
                    if index % 5 == 0:
                        progress(min(0.98, index / total), desc=f"Kare {index}/{total}")
                written = writer.frames_written

        progress(0.99, desc="Video kodlaniyor")
        output = _encode_for_web(output, meta.fps)
    except Exception as exc:  # kullaniciya ham traceback gosterme
        return None, f"Islem basarisiz oldu: {exc}"

    elapsed = time.perf_counter() - started
    return str(output), _report(pipeline, config, elapsed, is_example, written)


def _report(
    pipeline: PerceptionPipeline,
    config: Config,
    elapsed: float,
    is_example: bool,
    frames: int,
) -> str:
    stages = pipeline.profiler.summary()

    lines = [_calibration_note(is_example), "", "### Performans", ""]
    lines.append(f"{frames} kare, {elapsed:.1f} saniye "
                 f"({frames / elapsed:.1f} kare/sn islendi)\n")
    lines.append("| Aşama | ms/kare |")
    lines.append("|---|---|")
    for name, stat in sorted(stages.items(), key=lambda kv: -kv[1]["total_s"]):
        lines.append(f"| {name} | {stat['mean_ms']:.1f} |")

    if pipeline.tracker is not None:
        lines += ["", "### Takip", "```",
                  pipeline.tracker.stats.format_report(config.tracking.min_track_len), "```"]
    if pipeline.risk is not None:
        lines += ["", "### Risk", "```", pipeline.risk.summary(), "```"]
    return "\n".join(lines)


DESCRIPTION = f"""
Tek bir öne bakan kameradan sürüş sahnesini çözümleyen algı yığını:
**tespit → takip → derinlik → kuşbakışı projeksiyon → çarpışmaya kalan süre**,
üzerine sürülebilir alan segmentasyonu.

Ücretsiz CPU katmanında çalışıyor, gerçek zamanlı değil — video yüklenir,
çevrimdışı işlenir, sonuç döner. İlk **{MAX_SECONDS:.0f} saniye** işlenir.
Model ağırlıkları ilk çalıştırmada indiğinden o çalıştırma daha uzun sürer.

Kod ve mühendislik günlüğü: [github.com/Byomer1021/otonomarac](https://github.com/Byomer1021/otonomarac)
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Otonomaraç — Sürüş Algısı") as demo:
        gr.Markdown("# Otonomaraç — Tek Kameradan Sürüş Algısı")
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Column(scale=1):
                video_in = gr.Video(label="Sürüş videosu", sources=["upload"])
                depth_cb = gr.Checkbox(value=True, label="Derinlik (nesne başına göreli mesafe)")
                seg_cb = gr.Checkbox(value=True, label="Sürülebilir alan segmentasyonu")
                run_btn = gr.Button("İşle", variant="primary")
                if EXAMPLE.exists():
                    gr.Examples(
                        examples=[[str(EXAMPLE), True, True]],
                        inputs=[video_in, depth_cb, seg_cb],
                        label="Örnek: Maltepe, İstanbul (kalibre edilmiş)",
                    )
            with gr.Column(scale=2):
                video_out = gr.Video(label="Sonuç")
                report_out = gr.Markdown()

        run_btn.click(
            process,
            inputs=[video_in, depth_cb, seg_cb],
            outputs=[video_out, report_out],
        )
    return demo


if __name__ == "__main__":
    build_ui().queue().launch()

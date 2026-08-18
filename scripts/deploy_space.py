"""Projeyi Hugging Face Spaces'e yukler (Hafta 7).

Space, deponun kendisi degil bir alt kumesi: kutuphane kaynagi, arayuz, config
ve ornek klip. Veri, cikti ve gelistirme betikleri gitmez.

Onkosul - bir kez yapilir:

    hf auth login          # https://huggingface.co/settings/tokens (Write yetkisi)

Sonra:

    python scripts/deploy_space.py                 # <kullanici>/otonomarac
    python scripts/deploy_space.py --name demo     # <kullanici>/demo
    python scripts/deploy_space.py --private
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Space'e kopyalanacaklar. Geri kalan her sey disarida kalir - ozellikle
#: data/ (6.5 GB kaynak video) ve outputs/.
PAYLOAD = ["app.py", "src", "configs", "examples"]

#: Spaces yalnizca requirements.txt okur, requirements-app.txt'yi gormez.
#: Depodaki requirements.txt torch icermiyor - yerelde platforma gore ayri
#: kuruluyor. Space'te bu bosluk birakilirsa pip, ultralytics uzerinden
#: VARSAYILAN CUDA torch'unu ceker: CPU-only bir makineye 2.8 GB gereksiz
#: indirme ve muhtemelen basarisiz bir derleme. CPU tekerlegi acikca isteniyor.
EXTRA_REQUIREMENTS = [
    "--extra-index-url https://download.pytorch.org/whl/cpu",
    "torch",
    "torchvision",
    "gradio>=4.44",
]


def _check_login():
    from huggingface_hub import HfApi

    try:
        return HfApi().whoami()["name"]
    except Exception:
        raise SystemExit(
            "Hugging Face girisi yapilmamis.\n\n"
            "  1. https://huggingface.co/settings/tokens adresinden "
            "'Write' yetkili bir token olustur\n"
            "  2. Terminalde:  hf auth login\n"
            "  3. Token'i yapistir, sonra bu betigi tekrar calistir"
        )


def _stage(target: Path) -> None:
    """Space'e gidecek dosyalari gecici bir klasorde toplar."""
    for item in PAYLOAD:
        source = ROOT / item
        if not source.exists():
            raise SystemExit(f"Eksik dosya: {source}")
        if source.is_dir():
            shutil.copytree(
                source, target / item,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
        else:
            shutil.copy2(source, target / item)

    # Spaces frontmatter'i README.md olarak gider. sdk_version elle degil
    # kurulu gradio surumunden yaziliyor: elle tutulan bir surum numarasi
    # kacinilmaz olarak test edilenden ayrisir ve Space baska bir gradio ile
    # ayaga kalkar.
    readme = (ROOT / "docs" / "spaces-README.md").read_text(encoding="utf-8")
    readme = re.sub(
        r"^sdk_version:.*$", f"sdk_version: {_gradio_version()}",
        readme, count=1, flags=re.MULTILINE,
    )
    (target / "README.md").write_text(readme, encoding="utf-8")

    # Depodaki dosyanin bas yorumu "torch burada yok" diyor; Space surumunde
    # torch VAR, o yuzden yorumlar atilip dosya yeniden uretiliyor.
    base = [
        line
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    (target / "requirements.txt").write_text(
        "# Hugging Face Spaces icin uretildi - scripts/deploy_space.py\n"
        + "\n".join(base + [""] + EXTRA_REQUIREMENTS)
        + "\n",
        encoding="utf-8",
    )


def _gradio_version() -> str:
    """Yerelde kurulu gradio surumu - Space ayni surumle ayaga kalksin diye."""
    try:
        from importlib.metadata import version

        return version("gradio")
    except Exception:
        return "4.44.0"

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="otonomarac", help="Space adi")
    parser.add_argument("--private", action="store_true", help="Gizli Space olustur")
    parser.add_argument("--dry-run", action="store_true", help="Yalnizca hazirla, yukleme")
    args = parser.parse_args(argv)

    from huggingface_hub import HfApi

    user = _check_login()
    repo_id = f"{user}/{args.name}"
    print(f"kullanici : {user}")
    print(f"Space     : {repo_id}")

    staging = Path(tempfile.mkdtemp(prefix="space_"))
    _stage(staging)

    total = sum(f.stat().st_size for f in staging.rglob("*") if f.is_file())
    files = sum(1 for f in staging.rglob("*") if f.is_file())
    print(f"yuklenecek: {files} dosya, {total / 1024 / 1024:.1f} MB")

    if args.dry_run:
        print(f"\n--dry-run: yukleme yapilmadi. Hazirlanan klasor:\n  {staging}")
        return 0

    api = HfApi()
    api.create_repo(
        repo_id=repo_id, repo_type="space", space_sdk="gradio",
        private=args.private, exist_ok=True,
    )
    api.upload_folder(
        folder_path=str(staging), repo_id=repo_id, repo_type="space",
        commit_message="Deploy otonomarac perception demo",
    )
    shutil.rmtree(staging, ignore_errors=True)

    url = f"https://huggingface.co/spaces/{repo_id}"
    print(f"\nYuklendi: {url}")
    print("Ilk acilisda modeller inecegi icin birkac dakika surebilir.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

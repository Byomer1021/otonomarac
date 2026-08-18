"""Projeyi Hugging Face Spaces'e yukler (Hafta 7).

Space, deponun kendisi degil bir alt kumesi: kutuphane kaynagi, arayuz, config
ve ornek klip. Veri (6.5 GB kaynak video), ciktilar ve gelistirme betikleri
disarida kalir.

Dosyalar depodan DOGRUDAN yukleniyor, once gecici bir klasore kopyalanmiyor.
Ilk surum kopyaliyordu ve bu makinede calismadi: antivirus %TEMP% altina .py
yazilmasini engelliyor ve staging alti denemenin besinde PermissionError ile
dusuyordu (Hafta 1'de `pip install -e .` de ayni sebeple kirilmisti). Kopyalama
zaten gereksizdi - `upload_folder` desen filtresi kabul ediyor.

Space'e ozel iki dosya (README.md ve requirements.txt) diske hic yazilmadan,
bellekten yukleniyor.

Onkosul - bir kez yapilir:

    hf auth login          # https://huggingface.co/settings/tokens (Write yetkisi)

Sonra:

    python scripts/deploy_space.py
    python scripts/deploy_space.py --name demo --private
    python scripts/deploy_space.py --dry-run     # Hub'a dokunmadan listele
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Space'e gidecek dosya ve klasorler. Hem yukleme desenleri hem de
#: --dry-run listesi bundan turetiliyor, ikisi ayrisamasin diye.
PAYLOAD_FILES = ["app.py"]
PAYLOAD_DIRS = ["src", "configs", "examples"]

#: huggingface_hub desenleri fnmatch ile eslestiriyor; orada `*` bolu
#: isaretini de gecer, yani `src/*` alt klasorleri de kapsar.
ALLOW = PAYLOAD_FILES + [f"{d}/*" for d in PAYLOAD_DIRS]
IGNORE = ["*__pycache__*", "*.pyc"]

#: Spaces yalnizca requirements.txt okur.
#: Depodaki requirements.txt torch icermiyor - yerelde platforma gore ayri
#: kuruluyor. Space'te bu bosluk birakilirsa pip, ultralytics uzerinden
#: VARSAYILAN CUDA torch'unu ceker: CPU-only bir konteynere birkac gigabayt.
EXTRA_REQUIREMENTS = [
    "--extra-index-url https://download.pytorch.org/whl/cpu",
    "torch",
    "torchvision",
    "gradio>=4.44",
]


def _login_name() -> str:
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


def _gradio_version() -> str:
    """Yerelde kurulu gradio surumu - Space ayni surumle ayaga kalksin diye."""
    try:
        from importlib.metadata import version

        return version("gradio")
    except Exception:
        return "4.44.0"


def space_readme() -> str:
    """Space frontmatter'i; sdk_version kurulu gradio'dan yaziliyor.

    Elle tutulan bir surum numarasi kacinilmaz olarak test edilenden ayrisir ve
    Space baska bir gradio ile ayaga kalkar.
    """
    text = (ROOT / "docs" / "spaces-README.md").read_text(encoding="utf-8")
    return re.sub(
        r"^sdk_version:.*$", f"sdk_version: {_gradio_version()}",
        text, count=1, flags=re.MULTILINE,
    )


def space_requirements() -> str:
    """Space'in requirements.txt'si - depodakinden uretiliyor.

    Depodaki dosyanin bas yorumu "torch burada yok" diyor; Space surumunde
    torch VAR, o yuzden yorumlar atilip dosya yeniden kuruluyor.
    """
    base = [
        line
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    header = "# Hugging Face Spaces icin uretildi - scripts/deploy_space.py"
    return "\n".join([header] + base + [""] + EXTRA_REQUIREMENTS) + "\n"


def _matching_files() -> list[Path]:
    """Yuklenecek dosyalarin listesi - --dry-run bunu gosterir."""
    found = [ROOT / name for name in PAYLOAD_FILES if (ROOT / name).is_file()]
    for folder in PAYLOAD_DIRS:
        found += [
            path
            for path in sorted((ROOT / folder).rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        ]
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--name", default="otonomarac", help="Space adi")
    parser.add_argument("--private", action="store_true", help="Gizli Space olustur")
    parser.add_argument(
        "--hardware", default="cpu-basic",
        help="Space donanimi. cpu-basic PRO abonelik ister; ucretsiz katmanda zero-a10g",
    )
    parser.add_argument("--dry-run", action="store_true", help="Yalnizca listele, yukleme")
    args = parser.parse_args(argv)

    files = _matching_files()
    total = sum(f.stat().st_size for f in files)

    if args.dry_run:
        print(f"Yuklenecek {len(files) + 2} dosya, {(total) / 1024 / 1024:.1f} MB\n")
        for f in files:
            print(f"  {f.relative_to(ROOT)}")
        print("  README.md          (uretilecek)")
        print("  requirements.txt   (uretilecek)")
        print(f"\nsdk_version: {_gradio_version()}")
        print("\n--dry-run: Hub'a dokunulmadi.")
        return 0

    from huggingface_hub import CommitOperationAdd, HfApi

    user = _login_name()
    repo_id = f"{user}/{args.name}"
    print(f"kullanici : {user}")
    print(f"Space     : {repo_id}")
    print(f"yuklenecek: {len(files) + 2} dosya, {total / 1024 / 1024:.1f} MB")

    api = HfApi()
    try:
        api.create_repo(
            repo_id=repo_id, repo_type="space", space_sdk="gradio",
            space_hardware=args.hardware, private=args.private, exist_ok=True,
        )
    except Exception as exc:
        # Hugging Face 2026'da ucretsiz katmani daraltti: statik Space'ler
        # herkese acik ama Gradio/Docker Space'i CPU Basic'te barindirmak PRO
        # istiyor. Ham 402 yerine ne yapilacagini soyle.
        if "402" in str(exc):
            raise SystemExit(
                "Hugging Face bu Space'i olusturmayi reddetti (402).\n\n"
                "Gradio Space'ini CPU Basic'te barindirmak PRO abonelik istiyor;\n"
                "ucretsiz katmanda yalnizca statik Space'ler ve ZeroGPU var.\n\n"
                "  - PRO: https://huggingface.co/pro\n"
                "  - Ucretsiz GPU: --hardware zero-a10g  (app.py'de @spaces.GPU gerekir)"
            )
        raise

    # Kaynak dosyalar dogrudan depodan.
    api.upload_folder(
        folder_path=str(ROOT), repo_id=repo_id, repo_type="space",
        allow_patterns=ALLOW, ignore_patterns=IGNORE,
        commit_message="Deploy otonomarac perception demo",
    )
    # Space'e ozel iki dosya bellekten - diske yazilmiyor.
    api.create_commit(
        repo_id=repo_id, repo_type="space",
        operations=[
            CommitOperationAdd("README.md", space_readme().encode("utf-8")),
            CommitOperationAdd("requirements.txt", space_requirements().encode("utf-8")),
        ],
        commit_message="Space metadata and requirements",
    )

    url = f"https://huggingface.co/spaces/{repo_id}"
    print(f"\nYuklendi: {url}")
    print("Ilk acilista torch, ultralytics ve uc model agirligi inecegi icin")
    print("Space'in ayaga kalkmasi birkac dakika surebilir.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

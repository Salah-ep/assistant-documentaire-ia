"""
Genere deux documents de test dans data/ :

  - rapport_numerique.pdf : un PDF "natif" avec une vraie couche de
    texte (genere directement avec reportlab).
  - rapport_scanne.pdf : le MEME contenu, mais rendu en image puis
    reinsere dans un PDF sans aucune couche de texte - pour simuler un
    document scanne et exercer reellement le chemin OCR du pipeline.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

TEXTE_RAPPORT = [
    "Rapport interne - Maintenance des capteurs de pression",
    "",
    "Le site de production a enregistre une hausse des alertes de",
    "pression sur la ligne B au cours du mois d'octobre 2026.",
    "",
    "Apres analyse, la cause principale identifiee est l'encrassement",
    "progressif des capteurs, du a un defaut de filtration en amont.",
    "",
    "Une procedure de nettoyage preventif est recommandee toutes les",
    "deux semaines pour les capteurs installes sur la ligne B.",
    "",
    "Le cout estime de cette maintenance preventive est inferieur au",
    "cout d'un arret de production impreVu lie a une panne de capteur.",
]


def generate_native_pdf(path: Path) -> None:
    c = canvas.Canvas(str(path), pagesize=A4)
    width, height = A4
    text_obj = c.beginText(50, height - 60)
    text_obj.setFont("Helvetica", 12)
    for line in TEXTE_RAPPORT:
        text_obj.textLine(line)
    c.drawText(text_obj)
    c.save()


def generate_scanned_pdf(path: Path) -> None:
    """Rend le meme texte dans une image (comme un document scanne),
    puis l'enregistre directement en PDF image (sans couche de texte)."""
    img_width, img_height = 1240, 1754  # ~ A4 a 150 dpi
    img = Image.new("RGB", (img_width, img_height), color="white")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
    except OSError:
        font = ImageFont.load_default()

    y = 80
    for line in TEXTE_RAPPORT:
        draw.text((60, y), line, fill="black", font=font)
        y += 45

    img.save(path, "PDF", resolution=150.0)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    native_path = DATA_DIR / "rapport_numerique.pdf"
    scanned_path = DATA_DIR / "rapport_scanne.pdf"

    generate_native_pdf(native_path)
    generate_scanned_pdf(scanned_path)

    print(f"Genere : {native_path}")
    print(f"Genere : {scanned_path}")


if __name__ == "__main__":
    main()

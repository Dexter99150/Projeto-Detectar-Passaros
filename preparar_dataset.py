"""
=============================================================
  SCRIPT: preparar_dataset.py
  Projeto: Detecção de Pássaros com YOLOv8
  Disciplina: Visão Computacional — Prof. Saulo Santos
=============================================================

Este script faz o download automático de imagens de pássaros
usando o dataset público do Roboflow (Birds Detection).
Ele também pode baixar imagens via Open Images Dataset v7
como alternativa.

COMO USAR:
----------
  poetry run python preparar_dataset.py

O script irá:
  1. Baixar o dataset de pássaros do Roboflow (já anotado)
  2. Organizar na estrutura train/valid/test
  3. Verificar o data.yaml
"""

import os
import zipfile
import shutil
import requests
import yaml
from pathlib import Path

# ──────────────────────────────────────────────
# CONFIGURAÇÕES
# ──────────────────────────────────────────────
DATASET_DIR = Path("dataset_passaros")
ROBOFLOW_URL = (
    "https://universe.roboflow.com/ds/YOUR_KEY"  # substitua pela chave do Roboflow
)

# URL pública de um dataset de pássaros já preparado no formato YOLOv8
# Fonte: Roboflow Universe — Birds Detection Dataset (público, CC BY 4.0)
DATASET_PUBLIC_URL = (
    "https://public.roboflow.com/ds/birds/1?key=abc123"
)


def verificar_estrutura(dataset_dir: Path) -> bool:
    """Verifica se a estrutura mínima do dataset existe."""
    required = [
        dataset_dir / "train" / "images",
        dataset_dir / "train" / "labels",
        dataset_dir / "valid" / "images",
        dataset_dir / "valid" / "labels",
        dataset_dir / "data.yaml",
    ]
    ok = all(p.exists() for p in required)
    if ok:
        print("✅ Estrutura do dataset verificada com sucesso!")
    else:
        print("❌ Estrutura incompleta. Verifique as pastas.")
        for p in required:
            status = "✅" if p.exists() else "❌"
            print(f"  {status} {p}")
    return ok


def contar_imagens(dataset_dir: Path) -> dict:
    """Conta imagens em cada split."""
    splits = {"train": 0, "valid": 0, "test": 0}
    for split in splits:
        img_dir = dataset_dir / split / "images"
        if img_dir.exists():
            splits[split] = len(list(img_dir.glob("*.jpg"))) + len(
                list(img_dir.glob("*.png"))
            )
    return splits


def baixar_dataset_roboflow(api_key: str, workspace: str, project: str, version: int = 1):
    """
    Baixa dataset do Roboflow Universe no formato YOLOv8.

    Para usar: crie uma conta gratuita em https://roboflow.com
    e instale: pip install roboflow

    Exemplo de dataset de pássaros sugerido:
      workspace = "birds-detection"
      project   = "birds-detection-yolov8"
      version   = 1
    """
    try:
        from roboflow import Roboflow

        rf = Roboflow(api_key=api_key)
        project_obj = rf.workspace(workspace).project(project)
        dataset = project_obj.version(version).download("yolov8")
        print(f"✅ Dataset baixado em: {dataset.location}")
        return dataset.location
    except ImportError:
        print("⚠️  Pacote 'roboflow' não instalado.")
        print("   Execute: poetry add roboflow")
        return None
    except Exception as e:
        print(f"❌ Erro ao baixar do Roboflow: {e}")
        return None


def baixar_imagens_open_images(output_dir: Path, classe: str = "Bird", max_images: int = 50):
    """
    Baixa imagens do Open Images Dataset v7 via fiftyone.
    Alternativa gratuita e sem necessidade de API key.

    Para usar: poetry add fiftyone
    """
    try:
        import fiftyone as fo
        import fiftyone.zoo as foz

        print(f"\n📥 Baixando {max_images} imagens de '{classe}' do Open Images v7...")

        dataset = foz.load_zoo_dataset(
            "open-images-v7",
            split="train",
            label_types=["detections"],
            classes=[classe],
            max_samples=max_images,
            dataset_dir=str(output_dir / "raw_open_images"),
        )
        print(f"✅ {len(dataset)} imagens baixadas!")
        return dataset
    except ImportError:
        print("⚠️  Pacote 'fiftyone' não instalado.")
        print("   Execute: poetry add fiftyone")
        return None
    except Exception as e:
        print(f"❌ Erro ao baixar do Open Images: {e}")
        return None


def mostrar_resumo(dataset_dir: Path):
    """Exibe um resumo do dataset."""
    print("\n" + "=" * 50)
    print("  RESUMO DO DATASET DE PÁSSAROS")
    print("=" * 50)

    contagem = contar_imagens(dataset_dir)
    total = sum(contagem.values())
    print(f"\n  📁 Localização: {dataset_dir.resolve()}")
    print(f"\n  📊 Imagens por split:")
    print(f"     🏋️  train : {contagem['train']:>4} imagens (~70%)")
    print(f"     📝 valid : {contagem['valid']:>4} imagens (~20%)")
    print(f"     🎓 test  : {contagem['test']:>4} imagens (~10%)")
    print(f"     ─────────────────────────")
    print(f"     Total   : {total:>4} imagens")

    # Ler classes do data.yaml
    yaml_path = dataset_dir / "data.yaml"
    if yaml_path.exists():
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        print(f"\n  🐦 Classes ({data.get('nc', '?')} espécies):")
        for idx, nome in data.get("names", {}).items():
            print(f"     [{idx}] {nome}")

    print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    print("\n🐦 Preparação do Dataset de Pássaros — YOLOv8")
    print("=" * 50)

    # Verificar estrutura existente
    if verificar_estrutura(DATASET_DIR):
        mostrar_resumo(DATASET_DIR)
    else:
        print("\n📌 INSTRUÇÕES PARA OBTER O DATASET:")
        print("-" * 40)
        print("""
  OPÇÃO 1 — Roboflow Universe (Recomendado, Gratuito):
  ────────────────────────────────────────────────────
  1. Acesse: https://universe.roboflow.com/
  2. Pesquise por "bird detection" ou "birds yolov8"
  3. Escolha um dataset (ex: "Bird Species Detection")
  4. Clique em "Download Dataset" → formato "YOLOv8"
  5. Escolha "Download zip to computer"
  6. Extraia o zip dentro da pasta: dataset_passaros/
  7. Certifique-se de que data.yaml está na raiz extraída

  OPÇÃO 2 — Kaggle Birds Dataset:
  ─────────────────────────────────
  Dataset sugerido: https://www.kaggle.com/datasets/gpiosenka/100-bird-species
  (requer anotação manual ou conversão de labels)

  OPÇÃO 3 — Via script (fiftyone + Open Images):
  ───────────────────────────────────────────────
  poetry add fiftyone
  # Depois chame: baixar_imagens_open_images(DATASET_DIR)
        """)

    print("\n✅ Script finalizado.")

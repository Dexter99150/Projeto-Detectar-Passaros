"""
=============================================================
  SCRIPT: inferencia_passaros.py
  Projeto: Detecção de Pássaros com YOLOv8
  Disciplina: Visão Computacional — Prof. Saulo Santos
=============================================================

Script para usar o modelo treinado e detectar pássaros
em imagens, vídeos ou câmera ao vivo.

COMO USAR:
----------
  # Detectar em uma imagem:
  poetry run python inferencia_passaros.py --source imagem.jpg

  # Detectar em uma pasta de imagens:
  poetry run python inferencia_passaros.py --source pasta/imagens/

  # Detectar em tempo real pela webcam:
  poetry run python inferencia_passaros.py --source 0

  # Especificar outro modelo:
  poetry run python inferencia_passaros.py --source imagem.jpg --model meu_modelo.pt
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


# ──────────────────────────────────────────────
# CONFIGURAÇÕES PADRÃO
# ──────────────────────────────────────────────
MODELO_PADRAO = "modelo_final_passaros.pt"
CONFIANCA     = 0.25    # limiar mínimo de confiança (25%)
ESPESSURA_BOX = 2       # espessura das bordas das caixas
PASTA_SAIDA   = "resultados_predicao"


def detectar(source: str, model_path: str, conf: float, salvar: bool = True):
    """
    Executa detecção de pássaros em uma fonte de imagem/vídeo.

    Args:
        source    : caminho da imagem, pasta, vídeo ou '0' para webcam
        model_path: caminho para o arquivo .pt do modelo treinado
        conf      : confiança mínima para aceitar uma detecção
        salvar    : se True, salva as imagens com as caixas desenhadas
    """
    model_file = Path(model_path)

    # Verificar se o modelo existe
    if not model_file.exists():
        print(f"\n❌ Modelo não encontrado: {model_file}")
        print("   Execute o treinamento primeiro no notebook ou com:")
        print("   poetry run yolo detect train data=dataset_passaros/data.yaml ...")
        return

    print(f"\n🐦 Detecção de Pássaros — YOLOv8")
    print(f"   Modelo    : {model_file}")
    print(f"   Fonte     : {source}")
    print(f"   Confiança : {conf:.0%}")
    print("─" * 45)

    # Carregar modelo
    model = YOLO(str(model_file))

    # Executar predição
    results = model.predict(
        source=source,
        conf=conf,
        save=salvar,
        save_dir=PASTA_SAIDA if salvar else None,
        line_width=ESPESSURA_BOX,
        show=False,          # True para abrir janela (requer display)
        verbose=False,
    )

    # Exibir resultados no terminal
    total_passaros = 0
    for i, r in enumerate(results):
        n = len(r.boxes)
        total_passaros += n
        if n > 0:
            print(f"\n🖼️  Imagem {i+1}: {n} pássaro(s) detectado(s)")
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf_v = float(box.conf[0])
                nome   = model.names[cls_id]
                coords = box.xyxy[0].tolist()
                print(f"   → {nome:12s} | Confiança: {conf_v:.1%} | "
                      f"Caixa: [{coords[0]:.0f}, {coords[1]:.0f}, "
                      f"{coords[2]:.0f}, {coords[3]:.0f}]")
        else:
            print(f"\n🖼️  Imagem {i+1}: nenhum pássaro detectado.")

    print("\n" + "═" * 45)
    print(f"  Total de pássaros detectados: {total_passaros}")
    if salvar:
        print(f"  Imagens salvas em: {PASTA_SAIDA}/")
    print("═" * 45 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Detecção de Pássaros com YOLOv8",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python inferencia_passaros.py --source foto.jpg
  python inferencia_passaros.py --source pasta_fotos/
  python inferencia_passaros.py --source 0          # webcam
  python inferencia_passaros.py --source foto.jpg --conf 0.5
        """
    )
    parser.add_argument(
        "--source",
        type=str,
        default="dataset_passaros/test/images",
        help="Imagem, pasta, vídeo ou '0' para webcam (padrão: test/images)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODELO_PADRAO,
        help=f"Caminho do modelo .pt (padrão: {MODELO_PADRAO})",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=CONFIANCA,
        help=f"Confiança mínima (padrão: {CONFIANCA})",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Não salvar imagens com as detecções",
    )

    args = parser.parse_args()
    detectar(
        source=args.source,
        model_path=args.model,
        conf=args.conf,
        salvar=not args.no_save,
    )


if __name__ == "__main__":
    main()

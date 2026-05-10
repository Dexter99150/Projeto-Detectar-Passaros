"""
=============================================================
  SCRIPT: detectar_video_passaros.py
  Projeto: Detecção de Pássaros com YOLOv8
  Disciplina: Visão Computacional — Prof. Saulo Santos
=============================================================

Processa um vídeo MP4 frame a frame, detecta pássaros e
desenha um CÍRCULO colorido ao redor de cada um, junto com
o nome da espécie e a confiança da detecção.

COMO USAR:
----------
  # Com um vídeo MP4:
  poetry run python detectar_video_passaros.py --source video_passaros.mp4

  # Com a webcam ao vivo:
  poetry run python detectar_video_passaros.py --source 0

  # Ajustar confiança mínima:
  poetry run python detectar_video_passaros.py --source video.mp4 --conf 0.4

  # Especificar outro modelo:
  poetry run python detectar_video_passaros.py --source video.mp4 --model outro_modelo.pt

O vídeo processado será salvo como: passaros_detectados.mp4
"""

import cv2
import argparse
import math
from pathlib import Path
from ultralytics import YOLO


# ──────────────────────────────────────────────────────────────
# CONFIGURAÇÕES VISUAIS — edite para mudar a aparência
# ──────────────────────────────────────────────────────────────

# Cores por espécie (BGR — OpenCV usa BGR, não RGB!)
CORES_ESPECIES = {
    "Estornino":     (0,   200, 255),   # laranja
    "canario":      (255, 180,   0),   # azul claro
    "cernicalo":     (0,   255, 100),   # verde
    "curruca":   (0,    80, 255),   # vermelho
    "golondrina": (255,   0, 200),   # roxo
}
COR_PADRAO = (0, 255, 255)   # amarelo (para espécies não mapeadas)

ESPESSURA_CIRCULO  = 3    # espessura da borda do círculo
ESPESSURA_TEXTO    = 2    # espessura do texto
TAMANHO_FONTE      = 0.65 # tamanho da fonte
# ──────────────────────────────────────────────────────────────


def obter_cor(nome_especie: str) -> tuple:
    """Retorna a cor BGR para uma espécie. Usa cor padrão se não mapeada."""
    return CORES_ESPECIES.get(nome_especie.lower(), COR_PADRAO)


def desenhar_deteccao(frame, box, nome_especie: str, confianca: float):
    """
    Desenha um CÍRCULO + label sobre um pássaro detectado.

    O círculo é calculado a partir da bounding box do YOLO:
    - Centro = ponto médio da bounding box
    - Raio   = metade da diagonal da bounding box (cobre todo o pássaro)
    """
    # Coordenadas da bounding box
    x1, y1, x2, y2 = int(box.xyxy[0][0]), int(box.xyxy[0][1]), \
                     int(box.xyxy[0][2]), int(box.xyxy[0][3])

    # Centro e raio do círculo
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    largura  = x2 - x1
    altura   = y2 - y1
    raio = int(math.sqrt(largura**2 + altura**2) / 2)

    cor = obter_cor(nome_especie)

    # ── Desenhar círculo ──────────────────────────────────────────────────────
    cv2.circle(frame, (cx, cy), raio, cor, ESPESSURA_CIRCULO)

    # Círculo interno menor (ponto central) para destacar o centro
    cv2.circle(frame, (cx, cy), 4, cor, -1)

    # ── Montar texto do label ─────────────────────────────────────────────────
    label = f"{nome_especie} {confianca:.0%}"

    # Medir tamanho do texto para desenhar fundo
    (tw, th), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, TAMANHO_FONTE, ESPESSURA_TEXTO
    )

    # Posição do label (acima do círculo, centralizado)
    tx = cx - tw // 2
    ty = cy - raio - 8

    # Fundo escuro do texto (legibilidade)
    cv2.rectangle(frame, (tx - 4, ty - th - 4), (tx + tw + 4, ty + 4),
                  (0, 0, 0), -1)

    # Texto colorido
    cv2.putText(frame, label, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, TAMANHO_FONTE, cor,
                ESPESSURA_TEXTO, cv2.LINE_AA)

    return frame


def processar_video(source, model_path: str, conf: float, salvar: bool = True):
    """
    Lê um vídeo frame a frame, aplica detecção e desenha círculos.

    Args:
        source    : caminho do .mp4 ou 0 para webcam
        model_path: caminho do modelo .pt treinado
        conf      : confiança mínima para exibir detecção
        salvar    : se True, salva o vídeo processado
    """
    # ── Verificar modelo ──────────────────────────────────────────────────────
    model_file = Path(model_path)
    if not model_file.exists():
        print(f"\n❌ Modelo não encontrado: {model_file}")
        print("   Treine o modelo primeiro e copie o best.pt:")
        print("   cp runs/detect/passaros_yolov8n/weights/best.pt modelo_final_passaros.pt")
        return

    print(f"\n🐦 Detecção de Pássaros em Vídeo — YOLOv8")
    print(f"   Modelo    : {model_file}")
    print(f"   Fonte     : {source}")
    print(f"   Confiança : {conf:.0%}")
    print("─" * 50)

    # ── Carregar modelo ───────────────────────────────────────────────────────
    model = YOLO(str(model_file))
    print(f"✅ Modelo carregado. Classes conhecidas: {list(model.names.values())}")

    # ── Abrir vídeo com OpenCV ────────────────────────────────────────────────
    is_webcam = str(source) == "0"
    cap = cv2.VideoCapture(0 if is_webcam else str(source))

    if not cap.isOpened():
        print(f"❌ Não foi possível abrir a fonte: {source}")
        return

    # Propriedades do vídeo original
    fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"📹 Vídeo: {width}x{height} @ {fps}fps | Total frames: {total if not is_webcam else '∞ (webcam)'}")

    # ── Configurar gravação do vídeo de saída ─────────────────────────────────
    writer = None
    output_path = "passaros_detectados.mp4"
    if salvar:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"💾 Vídeo processado será salvo em: {output_path}")

    # ── Loop principal: frame a frame ─────────────────────────────────────────
    frame_num = 0
    total_deteccoes = 0

    print("\n▶️  Processando... (pressione Q para sair se janela aberta)\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        # Rodar YOLO no frame atual
        results = model(frame, conf=conf, verbose=False)

        deteccoes_frame = 0
        for result in results:
            for box in result.boxes:
                cls_id      = int(box.cls[0])
                confianca   = float(box.conf[0])
                nome        = model.names[cls_id]

                # Desenhar círculo + label no frame
                frame = desenhar_deteccao(frame, box, nome, confianca)
                deteccoes_frame += 1
                total_deteccoes += 1

        # Contador no canto superior esquerdo do vídeo
        cv2.putText(frame,
                    f"Passaros: {deteccoes_frame} | Frame: {frame_num}/{total if not is_webcam else '?'}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Salvar frame no vídeo de saída
        if writer:
            writer.write(frame)

        # Mostrar janela (funciona no Linux com display)
        cv2.imshow("Deteccao de Passaros - YOLOv8 | Q para sair", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n⏹️  Interrompido pelo usuário.")
            break

        # Log de progresso a cada 30 frames
        if frame_num % 30 == 0:
            pct = f"{frame_num/total*100:.1f}%" if total > 0 else f"frame {frame_num}"
            print(f"   [{pct}] Frame {frame_num} | Detecções acumuladas: {total_deteccoes}")

    # ── Encerrar ──────────────────────────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print("\n" + "═" * 50)
    print(f"  ✅ Processamento concluído!")
    print(f"  📊 Frames processados : {frame_num}")
    print(f"  🐦 Total de detecções : {total_deteccoes}")
    if salvar and Path(output_path).exists():
        tamanho_mb = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"  💾 Vídeo salvo        : {output_path} ({tamanho_mb:.1f} MB)")
    print("═" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Detecção de Pássaros em Vídeo com YOLOv8 — círculos coloridos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python detectar_video_passaros.py --source video_passaros.mp4
  python detectar_video_passaros.py --source 0                    # webcam
  python detectar_video_passaros.py --source video.mp4 --conf 0.4
        """
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Caminho do vídeo .mp4 ou '0' para webcam",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="modelo_final_passaros.pt",
        help="Caminho do modelo .pt treinado (padrão: modelo_final_passaros.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Confiança mínima para exibir detecção (padrão: 0.35)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Não salvar o vídeo de saída",
    )

    args = parser.parse_args()
    processar_video(
        source=args.source,
        model_path=args.model,
        conf=args.conf,
        salvar=not args.no_save,
    )


if __name__ == "__main__":
    main()

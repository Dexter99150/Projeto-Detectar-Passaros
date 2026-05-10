# 🐦 Detecção de Pássaros com YOLOv8

**Disciplina:** Visão Computacional  
**Professor:** Saulo Santos  
**Instituição:** UNIARA  
**Feito Por:** João Pedro Marucci Pagliuso RA: 05222-040  

---

## 📌 Sobre o Projeto

Este projeto implementa um modelo de **detecção de objetos** usando a arquitetura **YOLOv8** (You Only Look Once) para identificar e classificar espécies de pássaros em imagens e vídeos.

O modelo é treinado via **Transfer Learning** a partir do `yolov8n.pt` (pré-treinado no dataset COCO).

### Espécies detectadas:
| ID | Espécie |
|----|---------|
| 0  | Estornino  |
| 1  | Canario   |
| 2  | Cernicalo  |
| 3  | Curruca |
| 4  | Golondrina |
| 5  | Gorrion |
| 6  | Vencejo |
| 7  | Verdecillo |


## Passo a Passo

**1. Instalar dependências**
```bash
poetry install
```
> **⚠️ Sem GPU?** Se o PyTorch travar tentando baixar drivers NVIDIA, instale a versão CPU:
> ```bash
> poetry source add --priority=explicit pytorch_cpu \
>     https://download.pytorch.org/whl/cpu
> poetry add --source pytorch_cpu \
>     torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
> poetry add ultralytics
> ```

**2. Registrar o kernel no Jupyter**
```bash
poetry run python -m ipykernel install --user --name deteccao-passaros --display-name "deteccao-passaros (poetry)"
```

**3. Treinar o modelo** (O projeto já possui um modelo treinado, pode pular essa etapa)
```bash
poetry run yolo detect train data=$(pwd)/dataset_passaros/data.yaml model=yolov8n.pt epochs=50 imgsz=640
```

**4. Copiar o modelo treinado**
```bash
cp runs/detect/passaros_yolov8n/weights/best.pt modelo_final_passaros.pt
```

**5. Rodar a detecção no vídeo**
```bash
poetry run python detectar_video_passaros.py --source video_passaros.mp4
```

O vídeo processado será salvo automaticamente como `passaros_detectados.mp4`.

---

## ⚠️ Avisos

- O vídeo deve se chamar exatamente **`video_passaros.mp4`** e estar na raiz do projeto
- O formato suportado é **MP4 com codec H264** — se o vídeo não abrir, converta com:
```bash
ffmpeg -i seu_video.mp4 -vcodec libx264 video_passaros.mp4
```
- Já existe um vídeo de exemplo na pasta do projeto para testar imediatamente
- Projeto feito para rodar em linux

---

## Webcam (opcional)

```bash
poetry run python detectar_video_passaros.py --source 0
```

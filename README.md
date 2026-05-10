# 🐦 Detecção de Pássaros com YOLOv8

**Disciplina:** Visão Computacional  
**Professor:** Saulo Santos  
**Instituição:** UNIARA / UNICEP  

---

## 📌 Sobre o Projeto

Este projeto implementa um modelo de **detecção de objetos** usando a arquitetura **YOLOv8** (You Only Look Once) para identificar e classificar espécies de pássaros em imagens.

O modelo é treinado via **Transfer Learning** a partir do `yolov8n.pt` (pré-treinado no dataset COCO) e especializado para detectar pássaros brasileiros comuns.

### Espécies detectadas:
| ID | Espécie |
|----|---------|
| 0  | Pardal  |
| 1  | Pomba   |
| 2  | Tucano  |
| 3  | Papagaio |
| 4  | Beija-flor |

---

## 🗂️ Estrutura do Projeto

```
deteccao-passaros/
├── notebooks/
│   └── roteiro_alunos_yolo_passaros.ipynb  ← Notebook principal
├── dataset_passaros/
│   ├── data.yaml              ← Configuração do dataset
│   ├── train/
│   │   ├── images/            ← ~70% das imagens de treino
│   │   └── labels/            ← Anotações .txt (formato YOLO)
│   ├── valid/
│   │   ├── images/            ← ~20% para validação
│   │   └── labels/
│   └── test/
│       ├── images/            ← ~10% para teste final
│       └── labels/
├── preparar_dataset.py        ← Script para baixar o dataset
├── inferencia_passaros.py     ← Script de inferência standalone
├── pyproject.toml             ← Dependências (Poetry)
└── README.md
```

---

## ⚙️ Configuração do Ambiente

### Pré-requisitos
- Python 3.10+
- [Poetry](https://python-poetry.org/docs/#installation)
- Linux (Ubuntu 22.04+ recomendado)

### Instalação

```bash
# 1. Clonar o repositório
git clone https://github.com/SEU_USUARIO/deteccao-passaros.git
cd deteccao-passaros

# 2. Instalar dependências com Poetry
poetry install

# 3. Registrar kernel para o Jupyter
poetry run python -m ipykernel install --user \
    --name deteccao-passaros \
    --display-name "deteccao-passaros (poetry)"
```

> **⚠️ Sem GPU?** Se o PyTorch travar tentando baixar drivers NVIDIA, instale a versão CPU:
> ```bash
> poetry source add --priority=explicit pytorch_cpu \
>     https://download.pytorch.org/whl/cpu
> poetry add --source pytorch_cpu \
>     torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
> poetry add ultralytics
> ```

---

## 📥 Obtendo o Dataset

### Opção 1 — Roboflow Universe (Recomendado)

1. Acesse [https://universe.roboflow.com/](https://universe.roboflow.com/)
2. Pesquise por **"bird detection yolov8"**
3. Escolha um dataset com boas avaliações (ex: "Bird Species Detection")
4. Clique em **"Download Dataset"** → formato **"YOLOv8"**
5. Marque **"Download zip to computer"** e extraia dentro de `dataset_passaros/`
6. Certifique-se de que `data.yaml` está na raiz de `dataset_passaros/`

> **Dataset sugerido:**  
> [Birds Detection Dataset - Roboflow Universe](https://universe.roboflow.com/roboflow-100/bird-species)

### Opção 2 — Via script

```bash
# Ver instruções automáticas
poetry run python preparar_dataset.py
```

---

## 🚀 Treinamento

### Via Jupyter Notebook (recomendado)

```bash
# Abrir o VS Code na pasta do projeto
code .

# Abrir o notebook:
# notebooks/roteiro_alunos_yolo_passaros.ipynb
# Selecionar kernel: "deteccao-passaros (poetry)"
# Executar células em ordem
```

### Via Terminal

```bash
# Treinar o modelo
poetry run yolo detect train \
    data=dataset_passaros/data.yaml \
    model=yolov8n.pt \
    epochs=50 \
    imgsz=640 \
    batch=8 \
    name=passaros_yolov8n

# Para teste rápido (sem GPU):
poetry run yolo detect train \
    data=dataset_passaros/data.yaml \
    model=yolov8n.pt \
    epochs=5 \
    imgsz=416 \
    batch=2
```

Os resultados serão salvos em `runs/detect/passaros_yolov8n/`.

---

## 🎥 Detecção em Vídeo (com Círculos)

Assim como no exemplo do professor com peixes, você pode processar um vídeo MP4 e ver os pássaros sendo identificados em tempo real com **círculos coloridos** por espécie:

```bash
# Detectar pássaros em um vídeo MP4
poetry run python detectar_video_passaros.py --source video_passaros.mp4

# Webcam ao vivo
poetry run python detectar_video_passaros.py --source 0
```

O vídeo processado é salvo automaticamente como `passaros_detectados.mp4`.

| Espécie | Cor do círculo |
|---------|----------------|
| Pardal | 🟠 Laranja |
| Pomba | 🔵 Azul |
| Tucano | 🟢 Verde |
| Papagaio | 🔴 Vermelho |
| Beija-flor | 🟣 Roxo |

---

## 🔍 Inferência (Uso do Modelo)

```bash
# Detectar pássaros em uma imagem
poetry run python inferencia_passaros.py --source minha_foto.jpg

# Detectar em uma pasta inteira
poetry run python inferencia_passaros.py --source pasta_fotos/

# Webcam em tempo real
poetry run python inferencia_passaros.py --source 0

# Ajustar confiança mínima
poetry run python inferencia_passaros.py --source foto.jpg --conf 0.5
```

Ou via CLI do YOLO:
```bash
poetry run yolo detect predict \
    model=runs/detect/passaros_yolov8n/weights/best.pt \
    source=minha_foto.jpg \
    conf=0.25
```

---

## 📊 Interpretando os Resultados

Após o treinamento, acesse `runs/detect/passaros_yolov8n/`:

| Arquivo | Descrição |
|---------|-----------|
| `results.png` | Gráficos de loss e métricas por época |
| `confusion_matrix.png` | Matriz de confusão entre espécies |
| `val_batch0_pred.jpg` | Predições visuais no conjunto de validação |
| `weights/best.pt` | **Melhor modelo** (use este para inferência) |
| `weights/last.pt` | Último modelo salvo |

### Métricas-chave:
- **mAP@50 > 0.7** → modelo com bom desempenho ✅
- **mAP@50 > 0.5** → modelo razoável, precisa de mais dados ⚠️
- **mAP@50 < 0.3** → modelo insuficiente, revisar dataset ❌

---

## 🔗 Referências

- [Documentação YOLOv8 - Ultralytics](https://docs.ultralytics.com/)
- [Roboflow Universe — Datasets](https://universe.roboflow.com/)
- [Repositório do Professor](https://github.com/Prof-Saulo-Santos/curso-graduacao-visao-computacional-YOLO)
- [Transfer Learning com YOLOv8](https://docs.ultralytics.com/modes/train/)

---

## 📄 Licença

Projeto acadêmico para fins educacionais.

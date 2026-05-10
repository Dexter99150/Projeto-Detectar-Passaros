# Configuração do Ambiente — Detecção de Pássaros 🐦

Este documento é o guia passo a passo para configurar o ambiente de desenvolvimento
no Linux (Ubuntu) seguindo o padrão do Prof. Saulo Santos.

---

## 1. Instalar o Poetry (caso não tenha)

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Adicione ao PATH (se necessário):
```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

Verificar instalação:
```bash
poetry --version
```

---

## 2. Clonar o Projeto e Instalar Dependências

```bash
# Navegar até a pasta de projetos da disciplina
cd ~/projetos/visao/

# Clonar este repositório
git clone https://github.com/SEU_USUARIO/deteccao-passaros.git
cd deteccao-passaros

# Instalar dependências via Poetry
poetry install
```

---

## 3. Registrar o Kernel no Jupyter

```bash
poetry run python -m ipykernel install \
    --user \
    --name deteccao-passaros \
    --display-name "deteccao-passaros (poetry)"
```

Depois, abra o VS Code (Ctrl+Shift+P → Developer: Reload Window)  
e selecione o kernel **"deteccao-passaros (poetry)"** no notebook.

---

## 4. Solução para Problemas sem GPU

Se o Poetry travar tentando baixar drivers NVIDIA gigantescos:

```bash
# Adicionar fonte de PyTorch somente-CPU
poetry source add --priority=explicit pytorch_cpu \
    https://download.pytorch.org/whl/cpu

# Instalar versão CPU do torch
poetry add --source pytorch_cpu \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1

# Instalar ultralytics separado
poetry add ultralytics
```

---

## 5. Verificar se Tudo Está Funcionando

```bash
poetry run python -c "
import torch
from ultralytics import YOLO
print('torch    :', torch.__version__)
print('CUDA     :', torch.cuda.is_available())
model = YOLO('yolov8n.pt')
print('YOLOv8   : OK ✅')
"
```

---

## 6. Abrir o Notebook Principal

```bash
# No terminal (VS Code abre automaticamente)
code notebooks/roteiro_alunos_yolo_passaros.ipynb
```

Ou via Jupyter Lab:
```bash
poetry run jupyter lab
```

# Detecção de ovos férteis e inférteis

Projeto de visão computacional que usa o **YOLOv8** para localizar ovos em imagens e classificá-los como `fertile` (fértil) ou `infertile` (infértil). O treinamento usa o modelo pré-treinado `yolov8n.pt` e um conjunto de dados exportado do Roboflow em formato YOLO.

> **Status da validação:** treinamento, validação e inferência foram executados com sucesso. A avaliação final no conjunto `test` ainda não é confiável: seus 20 arquivos de rótulo estão vazios. O Ultralytics avisa que não há rótulos e retorna métricas iguais a zero. Anote esse conjunto antes de usar métricas de teste como resultado do projeto.

## Conteúdo

- `Visão-Computacional-final.ipynb`: notebook com experimentos de treinamento, avaliação e predição.
- `pipeline_ovos.py`: pipeline para treinar, avaliar e fazer predições.
- `validate_smoke.py`: teste rápido que executa uma época de treinamento e uma predição.
- `data.yaml`: configuração do conjunto de dados e nomes das classes.
- `train/`, `valid/`, `test/`: imagens e anotações do conjunto de dados.
- `yolov8n.pt`: pesos pré-treinados usados como ponto de partida.
- `runs/detect/train_final/weights/best.pt`: pesos do modelo treinado disponíveis neste projeto.
- `novas_imagens_teste/`: imagens para experimentar predições.

## Requisitos

- Python 3.12 (versão usada na validação deste projeto; outras versões não foram verificadas).
- Dependência principal: [Ultralytics](https://pypi.org/project/ultralytics/), que instala PyTorch e as dependências necessárias.
- Para treinamento, recomenda-se uma GPU compatível com PyTorch. O projeto também foi executado em CPU, mas o treinamento pode levar mais tempo.

Ambiente usado para as verificações deste README: Python 3.12.10, Ultralytics 8.4.171, PyTorch 2.4.1 (CPU).

## Instalação

Na pasta do projeto, crie e ative um ambiente virtual e instale o Ultralytics:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install ultralytics
```

## Como executar

### Teste rápido

```powershell
python validate_smoke.py
```

O teste treina por uma época e faz uma predição em `ovo.jpeg`. Ele não substitui uma avaliação completa do modelo.

### Treinamento e predição

```powershell
python pipeline_ovos.py
```

Por padrão, o pipeline treina por 50 épocas, salva os resultados em `runs/detect/train_final/`, avalia no conjunto `test` e tenta prever as imagens `ovo.jpeg`, `ovo2.jpg`, `ovo3.jpg` e `ovo4.jpg`. A avaliação em `test` continuará sem métricas válidas enquanto os rótulos desse conjunto estiverem vazios.

### Predição com os pesos treinados

```python
from ultralytics import YOLO

model = YOLO("runs/detect/train_final/weights/best.pt")
model.predict(source="novas_imagens_teste", conf=0.25, save=True)
```

As imagens anotadas são salvas pelo Ultralytics na pasta de resultados de predição.

## Conjunto de dados e avaliação

O arquivo `data.yaml` define duas classes: `fertile` e `infertile`. A configuração atual aponta para:

| Divisão | Imagens | Situação das anotações |
|---|---:|---|
| `train` | 139 | Anotações presentes |
| `valid` | 40 | Anotações presentes |
| `test` | 20 | Arquivos de rótulo vazios |

O checkpoint `runs/detect/train_final/weights/best.pt` foi carregado e executado sobre `valid`: nessa execução, foram obtidos precision **0,989**, recall **1,000**, mAP50 **0,995** e mAP50-95 **0,9935**. Esses valores são apenas uma verificação do conjunto de validação disponível; não substituem uma avaliação de teste independente.

O `pipeline_ovos.py` avalia em `test` e grava as métricas em `metricas_treinamento.csv`. Como as anotações de `test` estão vazias, os valores atuais desse arquivo (todos zero) não medem a qualidade do modelo. Adicione as anotações verdadeiras em `test/labels/` antes de interpretar ou publicar métricas de teste.

## Origem dos dados

O conjunto foi exportado do [Roboflow Universe](https://universe.roboflow.com/skripsi-gqi1u/dataset-new-c9tqt) e o arquivo de informações incluído no projeto indica licença **CC BY 4.0**. Consulte os termos e forneça a atribuição exigida ao redistribuir o conjunto.

## Observações

- Execute os comandos a partir da pasta do projeto para que os caminhos relativos sejam encontrados.
- O notebook contém células experimentais e caminhos absolutos específicos da máquina original; atualize esses caminhos antes de executá-las em outro computador. Para uso reproduzível, prefira os scripts acima.
- O conjunto é pequeno. Resultados altos em `valid` não garantem desempenho em imagens de outros ambientes; teste com dados representativos antes de qualquer uso prático.
- A pasta `runs/` contém resultados históricos de treinamento e ocupa aproximadamente 345 MB neste projeto. Antes de publicar, selecione os arquivos que deseja compartilhar; checkpoints e resultados repetidos aumentam o tamanho do repositório.

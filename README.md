# Vision Transformer (ViT) no CIFAR-10

Implementação de um Vision Transformer do zero usando PyTorch para classificação de imagens no dataset CIFAR-10.

## Visão Geral do Modelo

O Vision Transformer (ViT) é uma arquitetura transformer pura aplicada a tarefas de classificação de imagens. Em vez de usar camadas convolucionais, o ViT divide imagens em patches de tamanho fixo, aplica embeddings lineares e processa a sequência resultante através de blocos transformer encoder padrão.

**Componentes da Arquitetura:**

- **Patch Embedding**: Converte imagens 32x32 em sequências de patches 4x4 (64 patches no total)
- **Position Embedding**: Codificações posicionais aprendíveis adicionadas aos embeddings dos patches
- **Class Token**: Token de classificação aprendível adicionado à sequência
- **Transformer Encoder**: 6 camadas com 8 cabeças de atenção cada
- **MLP Head**: Classificador linear para predição de 10 classes

**Configuração do Modelo:**

```
Dimensão do Embedding: 256
Número de Heads: 8
Número de Camadas: 6
Dimensão do MLP: 512
Dropout: 0.1
Patch Size: 4x4
```

## Dataset

**CIFAR-10** é um dataset de visão computacional composto por 60.000 imagens coloridas 32x32 em 10 classes:

- Treinamento: 50.000 imagens
- Teste: 10.000 imagens

**Classes:** avião, automóvel, pássaro, gato, cervo, cachorro, sapo, cavalo, navio, caminhão

## Instalação

### Requisitos

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (opcional, para GPUs NVIDIA)
- macOS com Apple Silicon (opcional, para aceleração MPS)

### Configuração

1. Clone o repositório:

```bash
git clone https://github.com/Gabriel-Wamat/vision_transformer-.git
cd vision_transformer-
```

2. Crie um ambiente virtual:

```bash
python -m venv venv
source venv/bin/activate  # No macOS/Linux
# ou
venv\Scripts\activate  # No Windows
```

3. Instale as dependências:

```bash
pip install -r requirements.txt
```

### Dependências Principais

- **torch** (>=2.0.0): Framework de deep learning
- **torchvision** (>=0.15.0): Utilitários de visão computacional e datasets
- **numpy**: Computação numérica
- **matplotlib**: Visualizações
- **seaborn**: Visualizações estatísticas
- **scikit-learn**: Métricas e redução de dimensionalidade
- **umap-learn**: Visualização UMAP

## Uso

### Treinamento

1. Abra o Jupyter Notebook:

```bash
jupyter notebook
```

2. Abra o arquivo: `ViT_CIFAR_10.ipynb`

3. Execute as células sequencialmente

O notebook inclui:

- **Preparação dos Dados**: Download e pré-processamento automático do CIFAR-10
- **Arquitetura do Modelo**: Implementação completa do ViT
- **Treinamento**: Loop de treinamento com 10 épocas
- **Avaliação**: Métricas de accuracy e loss
- **Visualizações**: Predições e gráficos de accuracy

### Suporte Multi-Plataforma

O código detecta automaticamente o hardware disponível:

- **CUDA** (GPUs NVIDIA): Usado automaticamente se disponível
- **MPS** (Apple Silicon M1/M2/M3): Usado se CUDA não estiver disponível
- **CPU**: Fallback se nem CUDA nem MPS estiverem disponíveis

### Configuração de Treinamento

```python
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 3e-4
PATCH_SIZE = 4
EMBED_DIM = 256
NUM_HEADS = 8
DEPTH = 6
MLP_DIM = 512
DROP_RATE = 0.1
```

## Detalhes Técnicos

### Patch Embedding

As imagens são divididas em patches não sobrepostos de 4x4:

```
Entrada: (batch_size, 3, 32, 32)
Patches: (batch_size, 64, 256)
```

### Multi-Head Self-Attention

Atenção scaled dot-product com 8 cabeças paralelas:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
```

### Transformer Encoder

6 camadas de transformer encoder, cada uma contendo:
- Multi-Head Self-Attention
- MLP (Feed-Forward Network)
- Layer Normalization
- Residual Connections

## Estrutura do Projeto

```
vision_transformer-/
├── README.md
├── requirements.txt
└── ViT_CIFAR_10.ipynb
```

## Referências

- Dosovitskiy, A., et al. (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." ICLR 2021.
- Vaswani, A., et al. (2017). "Attention is All You Need." NeurIPS 2017.

## Requisitos de Hardware

**Mínimo:**
- 8GB RAM
- CPU moderno

**Recomendado:**
- 16GB RAM
- GPU NVIDIA com 4GB+ VRAM (CUDA)
- ou Apple M1/M2/M3 (MPS)

## Licença

Este projeto é fornecido para fins educacionais.

## Agradecimentos

Implementação baseada no paper original do Vision Transformer por Dosovitskiy et al. (2020).

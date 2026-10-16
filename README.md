# CodeFix AI — Engenharia de IA Generativa

Membros:
David de Medeiros Pacheco Junior- RM551462 
Kaique Maia Reis Silva- RM552112
Orlando Akio Morii Cardoso- RM98067
Pedro Henrique Alves Guariente- RM550301 

**Projeto:** análise e sugestões de correção para erros em código Python por uma LLM local.

**LLM base:** [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) (modelo existente, adaptado; não foi treinado do zero).  
**Treinamento:** fine-tuning supervisionado com QLoRA (LoRA + quantização NF4 de 4 bits), na GPU NVIDIA GeForce RTX 3050 Laptop (4 GB).  
**Interface:** Streamlit. **Ambiente testado:** Windows 11, Python 3.11, PyTorch 2.11.0+cu128, CUDA do PyTorch 12.8.

## O problema

Dado um trecho de código Python incorreto, a IA deve responder em português nos campos **Categoria**, **Causa** e **Correção**. Foram trabalhadas 6 classes: `SyntaxError`, `NameError`, `TypeError`, `IndexError`, `KeyError` e `ZeroDivisionError`.

O modelo **não executa** o código do usuário: apenas faz inferência sobre o texto.

## Dados

- `data/train.jsonl`: 48 exemplos
- `data/val.jsonl`: 12 exemplos — utilizados para comparar hiperparâmetros
- `data/test.jsonl`: 12 exemplos — reservados para a avaliação final

O conjunto é **pequeno e didático**. As classes estão balanceadas, mas muitos exemplos têm construções parecidas. Isso não demonstra capacidade geral de depuração de Python.

## Configuração da GPU

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python scripts/gpu_info.py
```

O computador utilizado apresentou `cuda_disponivel: true`, capacidade CUDA 8.6 e realizou cálculo com tensores CUDA. O comando `nvidia-smi` indicou suporte CUDA 13.3 no driver; a versão de runtime utilizada pelo PyTorch foi 12.8. Não foi necessário instalar o CUDA Toolkit completo separadamente.

**Reprodução:** a versão Windows do `scripts/train.py` utiliza `WindowsSFTTrainer`, sobrescrevendo `compute_loss` com `Trainer.compute_loss` para evitar dependência do kernel Triton na versão do TRL utilizada. Confirme que está versionando esse arquivo corrigido (não o script anterior à correção).

## Executar modelo base e treinar

```powershell
python scripts/testar_modelo.py
python scripts/train.py --nome smoke --r 8 --lr 0.0001 --epochs 1 --max-length 256 --max-steps 4 --sem-avaliacao
python scripts/train.py --nome e1 --r 8 --lr 0.0001 --epochs 2 --max-length 256
python scripts/train.py --nome e2 --r 16 --lr 0.0002 --epochs 2 --max-length 256
python scripts/train.py --nome e3 --r 16 --lr 0.0001 --epochs 3 --max-length 256
```

Todas as execuções de treinamento acima foram concluídas no equipamento do projeto. Os valores registrados a seguir vêm das saídas do terminal compartilhadas pelo grupo.

## Comparação de parâmetros

| Experimento | Rank LoRA | LR | Épocas | Steps | Tempo | Train loss | Val loss (final) |
|---|---:|---:|---:|---:|---:|---:|---:|
| E1 | 8 | 0.0001 | 2 | 24 | 37.57 s | 1.703 | 1.692 |
| E2 | 16 | 0.0002 | 2 | 24 | 38.27 s | 1.409 | **1.268** |
| E3 | 16 | 0.0001 | 3 | 36 | 58.40 s | 1.468 | 1.349 |

**E2 obteve a menor loss de validação.** Isso é sinal de melhor previsão de tokens nesse conjunto, **não** prova de que todas as correções sejam tecnicamente melhores. A diferença não isola um único hiperparâmetro, pois entre E1 e E2 mudaram rank e learning rate.

## Avaliação e reprodução

```powershell
# Validação: gerar os mesmos 12 exemplos com base e 3 adaptadores
python scripts/evaluate.py --split val --nome base
python scripts/evaluate.py --split val --nome e1 --adapter models/e1/adapter
python scripts/evaluate.py --split val --nome e2 --adapter models/e2/adapter
python scripts/evaluate.py --split val --nome e3 --adapter models/e3/adapter

# Consolidar CSVs gerados, sem precisar rodar a GPU de novo
python scripts/avaliacao_final.py --split val

# Avaliação final do holdout: não treinar nem escolher modelos com base nesses resultados
python scripts/evaluate.py --split test --nome base
python scripts/evaluate.py --split test --nome e1 --adapter models/e1/adapter
python scripts/evaluate.py --split test --nome e2 --adapter models/e2/adapter
python scripts/evaluate.py --split test --nome e3 --adapter models/e3/adapter
python scripts/avaliacao_final.py --split test
```

O avaliador original só pontua a categoria quando a resposta contém um cabeçalho `Categoria: ...`. As respostas originais sem esse campo são contabilizadas como `SEM_CATEGORIA`, mesmo quando identificam conceitualmente o erro. O arquivo `scripts/avaliacao_final.py` separa **acerto formal**, **menção de categoria** e **formato com os três campos**. A **correção semântica** só pode ser validada por revisão humana, registrada em `resultados/revisao_*.csv`, com notas 0–2.

Nos testes de demonstração com `saldo = 100`, `parcelas = 0`, `print(saldo / parcelas)`, E2 identificou `ZeroDivisionError`, mas recomendou verificar `saldo` (variável incorreta); E3 identificou a categoria e indicou verificar `parcelas`, que é o divisor. O teste E3 sugeriu usar quantidade de parcelas maior que zero, apropriado a esse cenário, embora a regra matemática geral de divisão seja apenas divisor diferente de zero.

## Frontend

```powershell
python -m streamlit run app.py
```

Abrir `http://localhost:8501`, selecionar original ou E1/E2/E3, colar código e clicar **Analisar erro**. O modelo e seus adaptadores devem estar presentes localmente. `localhost` não oferece hospedagem pública.

## Estrutura

- `scripts/` — inferência, treinamento, avaliação e inspeção da GPU
- `data/` — dados separados em treino, validação e teste
- `models/` — adaptadores LoRA **locais**, ignorados pelo Git por padrão
- `resultados/` — CSVs, logs e métricas geradas no ambiente de execução
- `evidencias/` — capturas da GPU, treinamento e frontend
- `docs/RELATORIO_FINAL.md` — descrição dos experimentos e conclusões
- `docs/GUIA_ENTREGA.md` — checklist final e GitHub

Os pesos completos do Qwen são baixados do Hugging Face. Os adaptadores LoRA treinados não estão incluídos neste repositório por padrão. **Para demonstrar em outra máquina**, disponibilize os adaptadores por meio de um armazenamento adequado ou reproduza os comandos de treinamento; sem os adaptadores, apenas o modelo original poderá ser executado.

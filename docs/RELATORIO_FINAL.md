# Relatório final — CodeFix AI

## Objetivo

Desenvolver uma aplicação de IA generativa executada localmente em GPU capaz de classificar erros de Python, explicar a causa e sugerir correção. O projeto adapta por QLoRA a LLM pública Qwen2.5-0.5B-Instruct, sem treinamento do zero.

## Ambiente utilizado

- Windows 11, Python 3.11, ambiente virtual (`.venv`)
- NVIDIA GeForce RTX 3050 Laptop, 4 GB VRAM dedicada, CUDA Compute Capability 8.6
- Driver NVIDIA 610.60, CUDA suportado pelo driver 13.3
- PyTorch 2.11.0+cu128, com runtime CUDA 12.8
- GPU reconhecida por `torch.cuda.is_available() == True`; cálculo de tensores executado
- TRL, PEFT, BitsAndBytes, Hugging Face Transformers, Streamlit

## Problema e dados

Depuração textual de trechos Python com seis classes: SyntaxError, NameError, TypeError, IndexError, KeyError, ZeroDivisionError. Os 72 exemplos foram separados em 48/12/12 para treino/validação/teste. Não foram fornecidos dados para alegar generalização além desse conjunto didático.

## Método

Fine-tuning supervisionado QLoRA: pesos base quantizados em 4 bits (NF4), adaptadores LoRA sobre projeções de atenção. Testes feitos sequencialmente na RTX 3050, com batch físico 1, acumulação de gradientes 4, contexto máximo 256 tokens e seed 42. Treinamento sob Windows utilizou uma substituição do cálculo de perda para evitar kernel Triton indisponível.

## Experimentos já executados

| Modelo | LoRA r | LR | Épocas | Steps | Tempo | Loss treino | Loss validação |
|---|---:|---:|---:|---:|---:|---:|---:|
| E1 | 8 | 0.0001 | 2 | 24 | 37.57 s | 1.703 | 1.692 |
| E2 | 16 | 0.0002 | 2 | 24 | 38.27 s | 1.409 | **1.268** |
| E3 | 16 | 0.0001 | 3 | 36 | 58.40 s | 1.468 | 1.349 |

O E2 apresentou a menor loss na validação; portanto é o melhor candidato pelo critério de loss, mas isso não resolve a escolha por qualidade de resposta. Rank e learning rate foram alterados simultaneamente na comparação E1 × E2; não é possível atribuir a melhoria a um único fator. O treinamento foi estável e os adaptadores foram salvos em `models/e1`, `models/e2`, `models/e3`.

## Caso de demonstração: divisão por zero

Código:

```python
saldo = 100
parcelas = 0
print(saldo / parcelas)
```

- **Modelo original:** identificou divisão por zero, mas mudou a lógica da operação na correção.
- **E2:** forneceu Categoria: ZeroDivisionError; sugeriu verificar `saldo` em vez de `parcelas`, logo a orientação foi tecnicamente incorreta.
- **E3:** forneceu Categoria: ZeroDivisionError e orientou verificar `parcelas > 0`. Para quantidade de parcelas, é uma validação plausível; em termos genéricos, o divisor só precisa ser diferente de zero.

Esse caso sugere melhoria de formato com o treinamento e demonstra que menor loss não garante orientação correta em cada exemplo.

## Métricas de classificação e teste final

**A PREENCHER COM SAÍDAS REAIS. Não é correto inferir números a partir de uma única demonstração.**

Execute `python scripts/avaliacao_final.py --split val` depois de gerar CSVs de cada modelo. Em seguida, execute `scripts/evaluate.py --split test` para base, E1, E2 e E3, seguido de `python scripts/avaliacao_final.py --split test`.

Copie os resultados das tabelas geradas em `resultados/resumo_val.md` e `resultados/resumo_test.md` para esta seção antes da entrega.

Para qualidade da correção, revise exemplos usando `resultados/revisao_*.csv` (0 = errado/ausente, 1 = parcial, 2 = correto). Distinguir classificação, formato e validade técnica. O avaliador antigo mostrou 0/12 no modelo base e no E1 porque nenhuma resposta foi reconhecida no **formato exigido**, não porque todos os conceitos estivessem necessariamente incorretos.

## Evidências

- `evidencias/GPU_Informacoes.png`: verificação CUDA e tensores
- `evidencias/GPU_Treinamento_E1.png`: atividade de GPU durante treinamento
- `evidencias/Frontend_Original.png`: inferência original
- `evidencias/Frontend_E2.png`: resposta do modelo E2
- `evidencias/Frontend_E3.png`: resposta do modelo E3
- **Pendente:** captura adicional equivalente dentro do VS Code, conforme o enunciado

## Conclusão

O objetivo de construir e executar uma LLM local, treinar adaptadores na GPU, testar hiperparâmetros, registrar perda e oferecer interface de prompts foi atingido no equipamento. A avaliação mostra que o projeto é um protótipo acadêmico, e não uma ferramenta de depuração confiável para uso sem supervisão. A qualidade de resposta precisa de checagem humana e de dados adicionais para melhorar generalização.

## Limitações

Amostra muito pequena; apenas seis categorias; respostas potencialmente imprecisas; casos de validação similares aos de treino; apenas um hardware e uma configuração de semente; screenshots de Visual Studio precisam ser complementadas por VS Code. Os pesos treinados estão na máquina local e requerem disponibilização separada ou reprodução do treinamento para uso por terceiros.

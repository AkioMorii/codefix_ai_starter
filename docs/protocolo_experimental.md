# Protocolo experimental

**Objetivo:** avaliar se um ajuste fino supervisionado com QLoRA melhora a classificação de erros de Python e a qualidade das explicações.

## Dados e prevenção de vazamento

- `train.jsonl`: 48 exemplos, 8 por categoria, para treinamento.
- `val.jsonl`: 12 exemplos separados, 2 por categoria, para comparação dos hiperparâmetros.
- `test.jsonl`: 12 exemplos separados, 2 por categoria, para relatório final.
- Este material é **um conjunto inicial pequeno e ilustrativo**, não prova eficácia geral. Expandir para pelo menos 200–500 exemplos diversificados antes da entrega é recomendado.
- Manter os exemplos de teste intactos e não reutilizá-los na criação de prompts de treinamento.

## Hipóteses

H0: adaptação não melhora a acurácia de classificação do modelo base no teste.
H1: adaptação melhora a acurácia no teste.

## Experimentos

| Modelo | r | taxa de aprendizado | épocas | demais configurações |
|---|---:|---:|---:|---|
| E1 | 8 | 0.0001 | 2 | lote 1, acumulação 4, seed 42 |
| E2 | 16 | 0.0002 | 2 | lote 1, acumulação 4, seed 42 |
| E3 | 16 | 0.0001 | 3 | lote 1, acumulação 4, seed 42 |

É um experimento exploratório: vários parâmetros mudam entre E1 e E3 de modo indireto, mas a tabela deixa as mudanças explícitas. Comparar E1 versus E2 e E2 versus E3 com cuidado, ou criar desenho fatorial se houver tempo.

## Métricas

- Acurácia de classificação = classificações corretas / total de prompts.
- Loss de validação registrada no fim de cada época pelo trainer.
- Avaliação humana cega (notas 0-2): causa correta, correção funcional, clareza.
- Registrar latência por pergunta (opcional), memória de GPU máxima e hardware.
- Apresentar pelo menos 3 erros reais acertados e 3 falhas ou limitações, sem escolher apenas casos positivos.
- Comparar respostas às **mesmas** perguntas, com temperatura de geração efetivamente determinística (`do_sample=False`).

## Evidências obrigatórias

Capturar no VS Code: `nvidia-smi`, `python scripts/gpu_info.py`, treinamento com uso de VRAM, logs de avaliação, interface Streamlit, commit/repositório GitHub.

## Resultados

Preencher a tabela somente depois dos testes executados; jamais publicar valores inventados.

| Modelo | acurácia validação | acurácia teste (final) | loss validação | GPU VRAM pico |
|---|---|---|---|---|
| Base | pendente | pendente | não aplicável | pendente |
| E1 | pendente | pendente | pendente | pendente |
| E2 | pendente | pendente | pendente | pendente |
| E3 | pendente | pendente | pendente | pendente |

# Resumo de avaliação — val

Resultados gerados a partir dos CSVs de inferência existentes. A análise abaixo **não reexecuta a GPU**.

| Modelo | Total | Categoria explícita correta | Menção única correta | 3 campos presentes |
|---|---:|---:|---:|---:|
| base | 12 | 0/12 (0%) | 1/12 (8%) | 0/12 (0%) |
| e1 | 12 | 0/12 (0%) | 0/12 (0%) | 11/12 (92%) |
| e2 | 12 | 11/12 (92%) | 11/12 (92%) | 12/12 (100%) |
| e3 | 12 | 7/12 (58%) | 7/12 (58%) | 12/12 (100%) |

## Revisão humana (notas de 0 a 2; opcional, mas recomendada)

0 = errado/ausente; 1 = parcialmente correto; 2 = correto. Sem preenchimento, a qualidade da solução permanece **não avaliada**.

| Modelo | Causa (média, n) | Correção (média, n) |
|---|---:|---:|
| base | não avaliada | não avaliada |
| e1 | não avaliada | não avaliada |
| e2 | não avaliada | não avaliada |
| e3 | não avaliada | não avaliada |

## Interpretação

- **Categoria explícita correta:** regra estrita do avaliador original; não mede se a solução está certa.
- **Menção única correta:** encontra um único nome formal de erro em qualquer trecho; ainda não entende sinônimos como “divisão por zero”.
- **3 campos presentes:** mede a estrutura, não a veracidade da causa ou da correção.
- **Revisão humana:** avalie o conteúdo real sem executar códigos arbitrários fornecidos pela LLM.
- Não selecione hiperparâmetros usando o conjunto `test`; utilize o conjunto `val`.
- O conjunto possui apenas 12 exemplos por partição de avaliação, com incerteza alta.

# Guia de encerramento e GitHub

## Antes do commit

1. Coloque o **train.py corrigido** que funcionou no seu computador na pasta `scripts/`. Ele deve conter a classe `WindowsSFTTrainer`. Não substitua pelos scripts antigos do ZIP inicial.
2. Mantenha `models/e1`, `models/e2`, `models/e3` no computador. Eles são ignorados pelo Git por padrão; **não apague**, pois o Streamlit precisa dos adaptadores.
3. Copie as evidências para `evidencias/`. Capture uma imagem extra com `nvidia-smi` e/ou `scripts/gpu_info.py` em **VS Code**, conforme texto da atividade.
4. No PowerShell, a partir da pasta raiz, salve as versões reais instaladas:

   ```powershell
   python -m pip freeze > docs/ambiente_instalado.txt
   ```

5. Se os CSVs de `base_val`, `e1_val`, `e2_val`, `e3_val` já estiverem em `resultados`, consolide sem reexecutar os modelos:

   ```powershell
   python scripts/avaliacao_final.py --split val
   ```

6. Avalie os modelos no conjunto de **teste** (uma vez, sem ajustar hiperparâmetros a partir dele):

   ```powershell
   python scripts/evaluate.py --split test --nome base
   python scripts/evaluate.py --split test --nome e1 --adapter models/e1/adapter
   python scripts/evaluate.py --split test --nome e2 --adapter models/e2/adapter
   python scripts/evaluate.py --split test --nome e3 --adapter models/e3/adapter
   python scripts/avaliacao_final.py --split test
   ```

7. Faça uma revisão humana das correções em `resultados/revisao_test.csv`: preencha pelo menos uma amostra **representativa** dos modelos com notas 0, 1 ou 2 e observações. Reexecute `python scripts/avaliacao_final.py --split test` para atualizar as médias.
8. Cole os números efetivos de `resultados/resumo_test.md` no `docs/RELATORIO_FINAL.md` e remova os trechos assinalados como pendentes.

## Criar e publicar o repositório

Crie no GitHub um repositório vazio `CodeFix-AI` (sem gerar README/.gitignore automaticamente, pois já existem). No PowerShell do projeto:

```powershell
git init
git add .
git status
git commit -m "Entrega CodeFix AI: QLoRA, GPU, avaliacao e Streamlit"
git branch -M main
git remote add origin https://github.com/SEU_USUARIO/CodeFix-AI.git
git push -u origin main
```

Se `origin` já existir, use `git remote -v` e ajuste com `git remote set-url origin <URL>`. Se houver commits anteriores, não reinicialize o repositório à força; use `git add .`, `git commit` e `git push`.

**Confira o GitHub pela web:** README na página principal, código, `data/`, `docs/`, `evidencias/`, `resultados/` e `requirements.txt`. Não publique a pasta `.venv`, caches, tokens, credenciais ou arquivos privados.

## Modelos treinados no GitHub

O arquivo `.gitignore` ignora `models/*`. Portanto, clonar o repositório **não** fornece os pesos do E1/E2/E3. Para deixar o projeto demonstrável por outra pessoa, há duas opções:

- Reproduzir o treinamento com os comandos do README na máquina com GPU compatível.
- Publicar adaptadores em um local de distribuição apropriado (por exemplo, Hugging Face Hub) e indicar links e instruções para baixar e posicionar em `models/e*/adapter`.

Não anuncie que o repositório inclui adaptadores se eles estiverem apenas no computador local.

## Checklist rápido

- [ ] Print do CUDA/GPU no **VS Code**
- [ ] Prints do treinamento e da interface
- [ ] CSVs e relatório de avaliação final
- [ ] Resultados descritos sem exagerar acurácia/qualidade
- [ ] Código Windows corrigido versionado
- [ ] README, dados, protocolo e ambiente registrado
- [ ] GitHub acessível com todos os arquivos exigidos
- [ ] Forma de reproduzir os adaptadores explicada

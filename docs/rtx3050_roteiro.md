# Roteiro da RTX 3050 Laptop (4 GB)

1. Print no VS Code: `nvidia-smi` e `python --version`.
2. Ativar ambiente virtual Python 3.11.
3. Instalar PyTorch com CUDA pelo gerador oficial https://pytorch.org/get-started/locally/.
4. `python scripts/gpu_info.py` para teste de tensor GPU e arquivo `resultados/gpu.json`.
5. `python scripts/testar_modelo.py` para a resposta original.
6. `python scripts/train.py --nome smoke --r 8 --lr 0.0001 --epochs 1 --max-length 256 --max-steps 4 --sem-avaliacao`.
7. Se funcionar, executar E1, E2, E3 documentados no README, um de cada vez.
8. Validar E1, E2, E3 com o modelo original; escolher melhor; usar conjunto de teste apenas uma vez.
9. `streamlit run app.py`; capturar interface, uso de VRAM e métricas.

**Se a GPU ficar sem memória:** feche jogos/apps que usem RTX; verifique VRAM com `nvidia-smi`; reduza o comprimento para 192 ou 128 tokens. Não anuncie sucesso sem testar. A execução foi preparada para 4 GB, mas ainda não testada nessa máquina.

**Dependências:** use Python 3.11, PyTorch com GPU CUDA, `pip install -r requirements.txt` e verifique a importação; os pacotes recentes `bitsandbytes`/TRL suportam Windows CUDA, mas versões devem ser verificadas no computador.

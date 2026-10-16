"""Teste básico de inferência sem executar o código enviado pelo usuário."""
import torch
from predict import carregar_modelo, gerar


def main():
    if not torch.cuda.is_available():
        raise SystemExit('GPU CUDA indisponível. Execute scripts/gpu_info.py primeiro.')
    print('GPU:', torch.cuda.get_device_name(0))
    print('Carregando modelo original: Qwen2.5-0.5B-Instruct...')
    tokenizer, model = carregar_modelo()
    trecho = 'print(10 / 0)'
    print('Código enviado:', trecho)
    print('Resposta do modelo original:\n', gerar(trecho, tokenizer, model, max_tokens=140))
    print('Memória máxima reservada (MB):', round(torch.cuda.max_memory_reserved() / 1024**2, 1))


if __name__ == '__main__':
    main()

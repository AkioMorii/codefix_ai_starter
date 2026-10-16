import argparse
import csv
import json
import re
from pathlib import Path

from predict import carregar_modelo, gerar


def categoria(texto):
    achado = re.search(r'Categoria\s*:\s*([A-Za-z]+Error)', texto, flags=re.IGNORECASE)
    return achado.group(1) if achado else 'SEM_CATEGORIA'


def main():
    parser = argparse.ArgumentParser(description='Mede identificação de erros em dados não usados no treino')
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--adapter', default=None, help='Ex.: models/e1/adapter; omitir para modelo base')
    parser.add_argument('--nome', required=True, help='Identificador dos resultados')
    args = parser.parse_args()
    tokenizer, model = carregar_modelo(args.adapter)
    origem = Path('data') / (args.split + '.jsonl')
    itens = [json.loads(x) for x in origem.read_text(encoding='utf-8').splitlines() if x.strip()]
    linhas = []
    for idx, item in enumerate(itens, 1):
        prompt = item['prompt'][-1]['content'].split('```python\n', 1)[1].rsplit('\n```', 1)[0]
        resposta = gerar(prompt, tokenizer, model)
        prevista, esperada = categoria(resposta), item['category']
        ok = prevista.lower() == esperada.lower()
        linhas.append({'id': idx, 'categoria_esperada': esperada, 'categoria_prevista': prevista,
                       'acertou': int(ok), 'entrada': prompt, 'resposta_modelo': resposta})
        print(f'{idx}/{len(itens)}: {esperada} -> {prevista} ({"OK" if ok else "ERRO"})')
    destino = Path('resultados')
    destino.mkdir(exist_ok=True)
    path = destino / f'{args.nome}_{args.split}.csv'
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=linhas[0].keys())
        writer.writeheader()
        writer.writerows(linhas)
    resumo = {'experimento': args.nome, 'split': args.split, 'total': len(linhas),
              'acertos': sum(row['acertou'] for row in linhas),
              'acuracia': sum(row['acertou'] for row in linhas) / len(linhas),
              'csv': str(path)}
    (destino / f'{args.nome}_{args.split}.json').write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(resumo, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

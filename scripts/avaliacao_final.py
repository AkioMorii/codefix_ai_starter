#!/usr/bin/env python3
"""Analisa os CSVs gerados por scripts/evaluate.py, sem executar a LLM.

Execucao na raiz do projeto:
    python scripts/avaliacao_final.py --split val
    python scripts/avaliacao_final.py --split test

Conceitos diferentes, não equivalentes:
- categoria_explicita: Categoria: NomeDoErro (formato estrito)
- mencao_unica: um único tipo de erro mencionado em qualquer ponto
- campos_completos: Categoria, Causa e Correção como cabeçalhos
- causa e correção corretas: revisão HUMANA obrigatória
"""
import argparse
import csv
import json
import re
from pathlib import Path

ERROS = ('SyntaxError', 'NameError', 'TypeError', 'IndexError', 'KeyError', 'ZeroDivisionError')
ERROS_NORM = {x.lower(): x for x in ERROS}
PADRAO_ERRO = re.compile(r'\b(?:SyntaxError|NameError|TypeError|IndexError|KeyError|ZeroDivisionError)\b', re.I)
PADRAO_CATEGORIA = re.compile(r'(?im)^\s*(?:\*\*)?Categoria(?:\*\*)?\s*:\s*`?([A-Za-z]+Error)\b')


def categoria_explicita(resposta):
    m = PADRAO_CATEGORIA.search(resposta)
    return ERROS_NORM.get(m.group(1).lower(), 'OUTRA') if m else 'SEM_CATEGORIA'


def categoria_mencionada(resposta):
    encontrados = {ERROS_NORM[e.lower()] for e in PADRAO_ERRO.findall(resposta)}
    return next(iter(encontrados)) if len(encontrados) == 1 else ('MULTIPLAS' if encontrados else 'NAO_MENCIONADA')


def tem_campo(resposta, campo):
    return bool(re.search(r'(?im)^\s*(?:\*\*)?' + campo + r'(?:\*\*)?\s*:', resposta))


def nota(valor):
    try:
        v = int(valor)
        return v if v in (0, 1, 2) else None
    except (ValueError, TypeError):
        return None


def main():
    parser = argparse.ArgumentParser(description='Consolida as avaliações sem GPU, a partir dos CSVs já produzidos.')
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--resultados', default='resultados')
    args = parser.parse_args()
    pasta = Path(args.resultados)
    pasta.mkdir(exist_ok=True)
    detalhes = []
    tabela = []
    for modelo in ('base', 'e1', 'e2', 'e3'):
        origem = pasta / f'{modelo}_{args.split}.csv'
        if not origem.exists():
            print('Ainda não encontrado:', origem)
            continue
        with origem.open(newline='', encoding='utf-8-sig') as f:
            linhas = list(csv.DictReader(f))
        if not linhas:
            print('Arquivo sem linhas:', origem)
            continue
        strict_ok = mention_ok = format_ok = 0
        for item in linhas:
            texto = item.get('resposta_modelo', '')
            esperada = item.get('categoria_esperada', '').strip()
            estrita = categoria_explicita(texto)
            mencao = categoria_mencionada(texto)
            completo = all(tem_campo(texto, s) for s in ('Categoria', 'Causa', 'Correção'))
            acertou_explicita = estrita.lower() == esperada.lower()
            acertou_mencao = mencao.lower() == esperada.lower()
            strict_ok += acertou_explicita
            mention_ok += acertou_mencao
            format_ok += completo
            detalhes.append({
                'modelo': modelo, 'split': args.split, 'id': item.get('id', ''),
                'categoria_esperada': esperada, 'categoria_explicita': estrita,
                'categoria_mencionada': mencao, 'acerto_explicito': int(acertou_explicita),
                'acerto_mencao_unica': int(acertou_mencao), 'campos_completos': int(completo),
                'entrada': item.get('entrada',''), 'resposta_modelo': texto,
                'nota_causa_0_2': '', 'nota_correcao_0_2': '', 'observacoes': '',
            })
        total = len(linhas)
        tabela.append({'modelo': modelo, 'total': total, 'acerto_explicito': strict_ok,
                       'acerto_mencao_unica': mention_ok, 'formato_completo': format_ok})
    if not detalhes:
        raise SystemExit('Nenhum CSV encontrado. Execute scripts/evaluate.py primeiro.')

    revisao = pasta / f'revisao_{args.split}.csv'
    if revisao.exists():
        print('Revisão já existe; mantendo notas humanas:', revisao)
    else:
        with revisao.open('w', newline='', encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=detalhes[0].keys());writer.writeheader();writer.writerows(detalhes)
        print('Planilha para revisão humana criada:', revisao)
    # Recalcular notas humanas a partir do arquivo salvo, inclusive após edição manual.
    notas = {}
    with revisao.open(newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            m = row['modelo']; ent = notas.setdefault(m, {'causa': [], 'correcao': []})
            for col, key in (('nota_causa_0_2','causa'), ('nota_correcao_0_2','correcao')):
                n=nota(row.get(col));
                if n is not None: ent[key].append(n)
    lines = [f'# Resumo de avaliação — {args.split}', '',
             'Resultados gerados a partir dos CSVs de inferência existentes. A análise abaixo **não reexecuta a GPU**.', '',
             '| Modelo | Total | Categoria explícita correta | Menção única correta | 3 campos presentes |',
             '|---|---:|---:|---:|---:|']
    for t in tabela:
        n=t['total']
        lines.append(f"| {t['modelo']} | {n} | {t['acerto_explicito']}/{n} ({t['acerto_explicito']/n:.0%}) | {t['acerto_mencao_unica']}/{n} ({t['acerto_mencao_unica']/n:.0%}) | {t['formato_completo']}/{n} ({t['formato_completo']/n:.0%}) |")
    lines += ['', '## Revisão humana (notas de 0 a 2; opcional, mas recomendada)', '',
              '0 = errado/ausente; 1 = parcialmente correto; 2 = correto. Sem preenchimento, a qualidade da solução permanece **não avaliada**.', '',
              '| Modelo | Causa (média, n) | Correção (média, n) |', '|---|---:|---:|']
    for t in tabela:
        model=t['modelo']; scores=notas.get(model, {})
        def fmt(l): return f'{sum(l)/len(l):.2f}/2 (n={len(l)})' if l else 'não avaliada'
        lines.append(f"| {model} | {fmt(scores.get('causa',[]))} | {fmt(scores.get('correcao',[]))} |")
    lines += ['', '## Interpretação', '',
              '- **Categoria explícita correta:** regra estrita do avaliador original; não mede se a solução está certa.',
              '- **Menção única correta:** encontra um único nome formal de erro em qualquer trecho; ainda não entende sinônimos como “divisão por zero”.',
              '- **3 campos presentes:** mede a estrutura, não a veracidade da causa ou da correção.',
              '- **Revisão humana:** avalie o conteúdo real sem executar códigos arbitrários fornecidos pela LLM.',
              '- Não selecione hiperparâmetros usando o conjunto `test`; utilize o conjunto `val`.',
              '- O conjunto possui apenas 12 exemplos por partição de avaliação, com incerteza alta.', '']
    out = pasta / f'resumo_{args.split}.md'
    out.write_text('\n'.join(lines),encoding='utf-8')
    (pasta / f'resumo_{args.split}.json').write_text(json.dumps(tabela,indent=2,ensure_ascii=False),encoding='utf-8')
    print('\n'.join(lines[:7+len(tabela)]))
    print('\nResumo salvo em:', out)

if __name__ == '__main__':
    main()

from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
SYSTEM = ('Você é um assistente de depuração de Python. Analise o erro sem executar o código. '
          'Responda em português usando exatamente três campos: Categoria, Causa e Correção. '
          'Seja claro e não invente execução.')


def carregar_modelo(adapter=None):
    if not torch.cuda.is_available():
        raise RuntimeError('GPU CUDA indisponível. Não é permitido executar em CPU neste projeto.')
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=dtype,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype
        ),
        device_map={'': 0},
    )
    if adapter:
        if not Path(adapter).is_dir():
            raise FileNotFoundError('Adaptador não encontrado: ' + str(adapter))
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    return tokenizer, model


def gerar(texto, tokenizer, model, max_tokens=220):
    messages = [
        {'role': 'system', 'content': SYSTEM},
        {'role': 'user', 'content': 'Analise este código Python e informe o erro:\n```python\n' + texto + '\n```'},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors='pt',
    ).to('cuda:0')
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0, inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()


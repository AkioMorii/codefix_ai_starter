import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import BitsAndBytesConfig, Trainer
from trl import SFTConfig, SFTTrainer

BASE_MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'


class WindowsSFTTrainer(SFTTrainer):
    """Usa a perda padrao do Transformers, evitando o kernel Triton do TRL 1.15."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        return Trainer.compute_loss(
            self,
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )



def main():
    parser = argparse.ArgumentParser(description='Treinamento supervisionado QLoRA com GPU')
    parser.add_argument('--nome', required=True, help='Nome do experimento, ex.: e1')
    parser.add_argument('--r', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=float, default=2)
    parser.add_argument('--max-length', type=int, default=256, help='Tokens por amostra; iniciar com 256 em GPU de 4 GB')
    parser.add_argument('--max-steps', type=int, default=-1, help='Ex.: 4 para um teste rápido; -1 para executar épocas completas')
    parser.add_argument('--sem-avaliacao', action='store_true', help='Ignora validação durante o teste rápido')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA não disponível: os experimentos devem rodar em GPU.')
    device_name = torch.cuda.get_device_name(0)
    print('GPU de treinamento:', device_name)
    bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    output = Path('models') / args.nome
    output.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset('json', data_files={'train': 'data/train.jsonl', 'validation': 'data/val.jsonl'})
    train_ds = dataset['train'].remove_columns(['category'])
    val_ds = dataset['validation'].remove_columns(['category'])

    trainer = WindowsSFTTrainer(
        model=BASE_MODEL,
        train_dataset=train_ds,
        eval_dataset=None if args.sem_avaliacao else val_ds,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        ),
        peft_config=LoraConfig(
            r=args.r,
            lora_alpha=args.r * 2,
            lora_dropout=0.05,
            target_modules=['q_proj', 'v_proj'],
            bias='none',
            task_type='CAUSAL_LM',
        ),
        args=SFTConfig(
            output_dir=str(output / 'checkpoint'),
            model_init_kwargs={'dtype': dtype},
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            max_length=args.max_length,
            max_steps=args.max_steps,
            completion_only_loss=True,
            eval_strategy='no' if args.sem_avaliacao else 'epoch',
            save_strategy='no',
            logging_steps=1,
            report_to='none',
            bf16=bf16,
            fp16=not bf16,
            seed=42,
            data_seed=42,
        ),
    )
    result = trainer.train()
    adapter = output / 'adapter'
    trainer.save_model(str(adapter))
    (output / 'logs.json').write_text(
        json.dumps(trainer.state.log_history, indent=2, ensure_ascii=False, default=str), encoding='utf-8'
    )
    (output / 'config.json').write_text(json.dumps({
        'base_model': BASE_MODEL, 'gpu': device_name, 'r': args.r,
        'lr': args.lr, 'epochs': args.epochs, 'max_length': args.max_length,
        'max_steps': args.max_steps, 'eval_during_train': not args.sem_avaliacao,
        'train_metrics': result.metrics,
    }, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    print('Treinamento finalizado; adaptador salvo em:', adapter)


if __name__ == '__main__':
    main()
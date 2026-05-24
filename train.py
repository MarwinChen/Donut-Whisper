import os
import torch
import numpy as np
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional
from transformers import HfArgumentParser, WhisperFeatureExtractor, WhisperTokenizer, DonutProcessor
from model.modeling_whisper import WhisperModel
import transformers
import random
import torch.distributed as dist

from transformers import GenerationConfig
from model import DonutWhisper, DonutWhisperAttn, WhisperOnly, DonutOnly, DisWhisperOnly, DisDonutWhisper, DonutWhisperED
from dataset.vistext_dataset import VistextDataset, VistextDataCollator
from train.vistext_trainer import VistextTrainer
from train.dis_whisper_trainer import VistextTrainer_DisWhisper

from peft import LoraConfig, get_peft_model


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    whisper_path: str = "openai/whisper-base"
    whisper_large_path: str = "openai/whisper-large-v3"
    image_model_path: str = "naver-clova-ix/donut-base-finetuned-cord-v2"
    train_data: Optional[str] = None
    eval_data: Optional[str] = None
    test_data: Optional[str] = None
    do_test: bool = False
    ckpt: Optional[str] = None
    ckpt_for_dis: Optional[str] = None
    model_type: str = "donut_whisper"
    train_encoder: bool = False
    label_smoothing_factor: float = 0
    max_grad_norm: float = 1.0
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    tokenizer_language: str = "zh"
    tokenizer_task: str = "transcribe"
    fusion_type: str = "sliding_window_q_former"
    fusion_window_size: int = 64
    fusion_stride: int = 32
    fusion_overlap: bool = True
    fusion_num_heads: int = 8
    fusion_dropout: float = 0.1

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    acc = predictions == labels
    mask = labels != -100
    accuracy = acc[mask].float().mean()
    return {'accuracy': accuracy, 'acc_sum': acc[mask].sum().item(), 'sum': mask.sum().item()}

# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     preds = predictions.argmax(-1)
#     acc = preds == labels
#     accuracy = acc[labels != -100].mean()
#     return {'accuracy': accuracy}

def _none_if_requested(value):
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {"", "none", "null"}:
        return None
    return value


def _strip_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key[len("module."):] if key.startswith("module.") else key
        new_state_dict[new_key] = value
    return new_state_dict


def _validate_args(training_args):
    training_args.ckpt = _none_if_requested(training_args.ckpt)
    training_args.ckpt_for_dis = _none_if_requested(training_args.ckpt_for_dis)
    training_args.deepspeed = _none_if_requested(training_args.deepspeed)

    if training_args.do_test:
        if not training_args.test_data:
            raise ValueError("--test_data is required when --do_test is true.")
    else:
        missing = [
            name
            for name in ("train_data", "eval_data")
            if not getattr(training_args, name)
        ]
        if missing:
            raise ValueError(f"Missing required training data argument(s): {', '.join(missing)}")


def _get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _get_world_size():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def _barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = HfArgumentParser(TrainingArguments)
    training_args, = parser.parse_args_into_dataclasses()
    _validate_args(training_args)
    training_args.logging_dir = os.path.join(training_args.output_dir, "logs")

    seed = training_args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    image_processor = DonutProcessor.from_pretrained(training_args.image_model_path)
    wav_processor = WhisperFeatureExtractor.from_pretrained(training_args.whisper_path)
    wav_processor_large = WhisperFeatureExtractor.from_pretrained(
        training_args.whisper_large_path or training_args.whisper_path
    )
    whisper_model = None

    if training_args.model_type == "donut_only":
        tokenizer = image_processor.tokenizer
        tokenizer_zh = WhisperTokenizer.from_pretrained(
            training_args.whisper_large_path or training_args.whisper_path,
            multilingual=True,
            language=training_args.tokenizer_language,
            task=training_args.tokenizer_task,
        )

    else:
        # tokenizer = WhisperTokenizer.from_pretrained(training_args.whisper_path, multilingual=False, language="en", task='transcribe')
        tokenizer_zh = WhisperTokenizer.from_pretrained(
            training_args.whisper_path,
            multilingual=True,
            language=training_args.tokenizer_language,
            task=training_args.tokenizer_task,
        )
        # tokenizer = WhisperTokenizer.from_pretrained(training_args.whisper_large_path, multilingual=False, language="en", task='transcribe')
        tokenizer = tokenizer_zh

        whisper_model = WhisperModel.from_pretrained(training_args.whisper_path)
        # whisper_model.resize_token_embeddings(len(tokenizer))

    
    
    
    # whisper_model_for_tea = WhisperModel.from_pretrained(training_args.whisper_path)
    # whisper_large_model = WhisperModel.from_pretrained(training_args.whisper_large_path)

    if training_args.use_lora and whisper_model is not None:
        # LoRA 配置，只作用于 decoder
        decoder_modules = []
        for name, module in whisper_model.named_modules():
            if name.startswith("decoder") and any(n in name for n in ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]):
                decoder_modules.append(name)
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=decoder_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # 包裹 LoRA
        whisper_model = get_peft_model(whisper_model, lora_config)
        # for name, param in whisper_large_model.named_parameters():
        #     if name.startswith("base_model.model.decoder"):
        #         if "decoder.embed_tokens" in name or "decoder.embed_positions" in name or "self_attn_layer_norm" in name or "final_layer_norm" in name or "decoder.layer_norm" in name:
        #             param.requires_grad = True

    if training_args.model_type == "donut_whisper":
        model = DonutWhisper(whisper_model=whisper_model, image_model_path=training_args.image_model_path,
        fusion_type=training_args.fusion_type,
        fusion_config={
            "window_size": training_args.fusion_window_size,
            "stride": training_args.fusion_stride,
            "overlap": training_args.fusion_overlap,
            "num_heads": training_args.fusion_num_heads,
            "dropout": training_args.fusion_dropout,
        }).to(torch.float32).to(device)
    elif training_args.model_type == "donut_whisper_attn":
        model = DonutWhisperAttn(whisper_model=whisper_model, image_model_path=training_args.image_model_path).to(torch.float32).to(device)
    elif training_args.model_type == "whisper_only":
        model = WhisperOnly(whisper_model=whisper_model).to(torch.float32).to(device)
    elif training_args.model_type == "donut_only":
        model = DonutOnly(image_model_path=training_args.image_model_path).to(torch.float32).to(device)
    elif training_args.model_type == "distillation_whisper_only":
        # 需要先定义whisper_large_model
        whisper_large_model = WhisperModel.from_pretrained(training_args.whisper_large_path)
        model = DisWhisperOnly(whisper_model=whisper_large_model, model_type="large_whisper_only").to(torch.float32).to(device)
        tea_model = DisDonutWhisper(whisper_model=whisper_model, image_model_path=training_args.image_model_path).to(torch.float32).to(device)
        # tea_model = DisWhisperOnly(whisper_model=whisper_large_model, model_type="whisper_large").to(torch.float32).to(device)
        # tea_model = None
    elif training_args.model_type == 'donut_whisper_ed':
        model = DonutWhisperED(whisper_model=whisper_model, image_model_path=training_args.image_model_path).to(torch.float32).to(device)
    else:
        raise NotImplementedError


    if training_args.ckpt is not None and training_args.ckpt != "None":
        print(f"Load ckpt: {training_args.ckpt}")
        ckpt = torch.load(training_args.ckpt, map_location="cpu")
        new_ckpt = _strip_module_prefix(ckpt)

        kk = model.load_state_dict(new_ckpt, strict=False)
        print(len(kk.unexpected_keys), len(kk.missing_keys))

    if training_args.ckpt_for_dis is not None and training_args.ckpt_for_dis != "None" and training_args.model_type == "distillation_whisper_only":
        print(f"Load ckpt for distillation: {training_args.ckpt_for_dis}")
        ckpt = torch.load(training_args.ckpt_for_dis, map_location="cpu")
        new_ckpt = _strip_module_prefix(ckpt)

        kk = tea_model.load_state_dict(new_ckpt, strict=False)
        print(len(kk.unexpected_keys), len(kk.missing_keys))

    if not training_args.do_test:
        train_dataset = VistextDataset(training_args.train_data, image_processor, wav_processor, wav_processor_large)
        eval_dataset = VistextDataset(training_args.eval_data, image_processor, wav_processor, wav_processor_large)
        collate_fn = VistextDataCollator(tokenizer, tokenizer_zh)

        
        if training_args.deepspeed is not None:
            if training_args.model_type == "distillation_whisper_only":
                trainer = VistextTrainer_DisWhisper(model=model, tea_model=tea_model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=collate_fn, compute_metrics=compute_metrics)
            else:
                trainer = VistextTrainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=collate_fn, compute_metrics=compute_metrics)
        else:
            # optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate)
            if training_args.model_type == "distillation_whisper_only":
                trainer = VistextTrainer_DisWhisper(model=model, tea_model=tea_model, tokenizer=tokenizer, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=collate_fn, compute_metrics=compute_metrics)
            else:
                trainer = VistextTrainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=collate_fn, compute_metrics=compute_metrics) 
        
        if not training_args.use_lora or whisper_model is None:
            train_lst = ['mix_linear', 'image_linear',  "qformer", 'fusion_layer', 'decoder']
            if training_args.train_encoder:
                train_lst.append("encoder")
            for name, param in model.named_parameters():
                if any([it in name for it in train_lst]):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            train_lst = ['mix_linear', 'image_linear',  "qformer", 'fusion_layer']
            if training_args.train_encoder:
                train_lst.append("encoder")
            
            # 首先冻结所有参数
            for name, param in model.named_parameters():
                param.requires_grad = False
            
            # 然后只解冻指定的模块
            for name, param in model.named_parameters():
                if any([it in name for it in train_lst]):
                    param.requires_grad = True
                # 确保LoRA参数保持可训练状态
                elif "lora" in name.lower():
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        
        if training_args.model_type == "distillation_whisper_only":
            for name, param in tea_model.named_parameters():
                param.requires_grad = False

        temp_cnt, temp_total = 0, 0
        if _get_rank() == 0:
            for k, p in model.named_parameters():
                temp_total += 1
                if p.requires_grad:
                    print(k)
                    temp_cnt += 1

            print(temp_cnt, temp_total)
        # 总参数量
        total_params = sum(p.numel() for p in model.parameters())

        # 可训练参数量（requires_grad=True）
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Total params: {total_params}, Trainable params: {trainable_params}")
        trainer.train()
        # trainer.save_model("final_model")


    else:
        test_dataset = VistextDataset(training_args.test_data, image_processor, wav_processor, wav_processor_large)
        collate_fn = VistextDataCollator(tokenizer, tokenizer_zh)
        if training_args.model_type == "distillation_whisper_only":
            trainer = VistextTrainer_DisWhisper(model=model, tea_model=tea_model, tokenizer=tokenizer, args=training_args, train_dataset=test_dataset, eval_dataset=test_dataset, data_collator=collate_fn, compute_metrics=compute_metrics)
        else:
            trainer = VistextTrainer(model=model, args=training_args, train_dataset=test_dataset, eval_dataset=test_dataset, data_collator=collate_fn, compute_metrics=compute_metrics)
        
        generation_config = GenerationConfig(max_new_tokens=128, do_sample=False, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
        # if dist.get_rank() == 0:
        outputs = trainer.predict(test_dataset, generation_config=generation_config).metrics['results']
        for item in outputs:
            if item["pred"] is not None:
                item["pred"] = tokenizer.decode(item["pred"], skip_special_tokens=True)

        _barrier()
        if _get_rank() == 0:
            os.makedirs(training_args.output_dir, exist_ok=True)

        _barrier()
        with open(os.path.join(training_args.output_dir, f"results_{_get_rank()}.json"), 'w', encoding="utf-8") as fp:
            json.dump(outputs, fp, ensure_ascii=False)

        if _get_rank() == 0:
            res = []
            print("Start Merging")
            for i in range(_get_world_size()):
                with open(os.path.join(training_args.output_dir, f"results_{i}.json"), 'r', encoding="utf-8") as fp:
                    data_i = json.load(fp)
                res += data_i

            map_dic = {}
            new_res = []
            for item in res:
                if item["id"] not in map_dic:
                    map_dic[item["id"]] = 1
                    new_res.append(item)

            with open(os.path.join(training_args.output_dir, f"results_final.json"), 'w', encoding="utf-8") as fp:
                json.dump(new_res, fp, indent=4, ensure_ascii=False)
            print(os.path.join(training_args.output_dir, f"results_final.json"))

        _barrier()

if __name__ == "__main__":
    train()

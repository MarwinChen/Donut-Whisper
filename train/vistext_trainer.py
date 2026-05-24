from transformers import Trainer
import torch.distributed as dist
from tqdm import tqdm
from typing import Dict
# from transformers.trainer_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
import torch.nn as nn
import os
import torch
import numpy as np
from torch.utils.data import Dataset, Sampler, RandomSampler
from transformers.trainer_pt_utils import get_length_grouped_indices
from torch.cuda.amp import autocast
from transformers.trainer_utils import EvalPrediction, EvalLoopOutput
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import json
import copy
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.timing_tracer import TimingTracer, TimingContext

class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths=None,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size)
        return iter(indices)

class VistextTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metrics = None
        self._best_metric = float("-inf")
        self._best_checkpoint = None
        self.timing_tracer = None  # 时序记录器

    def _get_train_sampler(self):
        lengths = self.train_dataset.lengths
        return LengthGroupedSampler(
            self.args.train_batch_size,
            world_size=self.args.world_size * self.args.gradient_accumulation_steps,
            lengths=lengths,
        )
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop('labels')
            # labels_zh = inputs["labels_zh"]
        else:
            labels = None
            # labels_zh = inputs["labels_zh"]
        outputs = model(**inputs)
        if "loss" not in outputs:
            logits = outputs.logits
            logits = logits[:, :-1, :] # <|startoftranscript|><|en|><|transcribe|><|notimestamps|>

            targets = inputs["labels"]
            # targets_zh = inputs["labels_zh"]
            targets = targets[:, 1:]

            # targets_zh = targets_zh[:, 1:]
            criterion = nn.CrossEntropyLoss(ignore_index=-100)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            outputs = {"loss": loss, "logits": logits}

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]


        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                "The model did not return a loss from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
            )
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool = None,
        ignore_keys = None,
    ):

        with torch.no_grad():
            logits = self.model(**inputs).logits
            logits = logits[:, :-1, :] # <|startoftranscript|><|en|><|transcribe|><|notimestamps|>
            labels = inputs["labels"]
            # labels_zh = inputs["labels_zh"]
            labels = labels[:, 1:]
            # labels_zh = labels_zh[:, 1:]
        if prediction_loss_only:
            loss = nn.CrossEntropyLoss(ignore_index=-100)(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            return loss, logits, labels
        else:
            return None, logits, labels

    def predict(self, test_dataset, generation_config=None):
        test_dataloader = self.get_test_dataloader(test_dataset)
        return self.prediction_loop(test_dataloader, generation_config=generation_config)

    def prediction_loop(
            self, 
            dataloader, 
            generation_config = None,         
            description: str = "Evaluation",
            metric_key_prefix: str = "eval",
            prediction_loss_only: bool = None,
            ignore_keys=None,
        ):
        model = self.model
        batch_size = dataloader.batch_size
        self.model.eval()

        # 初始化时序记录器（仅在测试时启用）
        enable_timing = getattr(self.args, 'do_test', False)
        if enable_timing and self.timing_tracer is None:
            self.timing_tracer = TimingTracer(enabled=True)
            # 为模型注册时序记录
            if hasattr(model, 'set_timing_tracer'):
                model.set_timing_tracer(self.timing_tracer)

        all_traces = []  # 存储所有批次的时序轨迹

        if dist.get_rank() == 0:
            pbar = tqdm(dataloader, total=len(dataloader), desc=description)
        else:
            pbar = dataloader
        results = []
        acc_sum = 0
        sum = 0
        acc = 0

        for batch_idx, inputs in enumerate(pbar):
            # 开始新的时序轨迹
            if enable_timing and self.timing_tracer:
                self.timing_tracer.start_trace()
            labels = inputs.pop("labels")
            # labels_zh = inputs.pop("labels_zh")
            audios = inputs.pop('audios')
            texts = inputs.pop('texts')
            # texts_zh = inputs.pop('texts_zh')
            data_ids = inputs.pop('data_ids')

            inputs.pop("spectrograms_large")
            
            if self.model.model_type == "donut_only" or self.model.model_type == "donut_whisper_ed":
                inputs.pop("spectrograms")
                inputs["input_ids"] = labels[:, :1]
            elif self.model.model_type == "whisper_only":
                inputs["input_ids"] = labels[:, :4]
                inputs.pop("images")
                inputs.pop("images_len")
            else:
                inputs["input_ids"] = labels[:, :4]
            labels = labels.cpu()
            
            # 记录生成过程的时间
            if enable_timing and self.timing_tracer:
                with TimingContext(self.timing_tracer, "model_generate", {"batch_idx": batch_idx, "batch_size": len(data_ids)}):
                    with autocast():
                        output = self.model.generate(generation_config=generation_config, **inputs).cpu()
                # 结束当前轨迹
                trace = self.timing_tracer.end_trace()
                if trace:
                    all_traces.append(trace)
            else:
                with autocast():
                    output = self.model.generate(generation_config=generation_config, **inputs).cpu()
            
            for i in range(len(data_ids)):
                results.append({
                    "id": data_ids[i],
                    "ref": texts[i],
                    'labels': labels[i],
                    # "ref_zh": texts_zh[i],
                    "pred": output[i]
                })
            pred_list = [item["pred"] for item in results]
            label_list = [item["labels"] for item in results]
            max_len = max(
                max(len(p) for p in pred_list),
                max(len(l) for l in label_list)
            )
            padded_preds = pad_sequence(pred_list, batch_first=True, padding_value=-100)
            padded_labels = pad_sequence(label_list, batch_first=True, padding_value=-100)
            predictions = self.pad_to_maxlen(padded_preds, max_len)
            label_ids = self.pad_to_maxlen(padded_labels, max_len)
            metrics = self.compute_metrics(EvalPrediction(predictions=predictions, label_ids=label_ids))
            acc_sum += metrics['acc_sum']
            sum += metrics['sum']
            acc = acc_sum / sum if sum > 0 else 0
            if dist.get_rank() == 0:
                pbar.set_postfix({'ACC': f'{acc*100:.2f}%'})

        # 1. 汇总数值型指标
        acc_sum_tensor = torch.tensor(acc_sum, dtype=torch.float32, device='cuda')
        sum_tensor = torch.tensor(sum, dtype=torch.float32, device='cuda')
        dist.all_reduce(acc_sum_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_tensor, op=dist.ReduceOp.SUM)
        global_acc = acc_sum_tensor.item() / sum_tensor.item() if sum_tensor.item() > 0 else 0

        # 2. 汇总所有进程的 results
        all_results = [None for _ in range(dist.get_world_size())]
        
        dist.all_gather_object(all_results, results)

        metrics = {}
        results = []
        for part in all_results:
            if part is not None:
                for item in part:
                    item.pop("labels")
                    results.append(item)
        metrics['accuracy'] = global_acc
        metrics['results'] = results
        self.metrics = metrics
        
        # 保存时序信息
        if enable_timing and self.timing_tracer:
            # 汇总所有进程的时序信息
            all_traces_list = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(all_traces_list, all_traces)
            
            if dist.get_rank() == 0:
                # 合并所有进程的轨迹
                merged_traces = []
                for traces in all_traces_list:
                    if traces:
                        merged_traces.extend(traces)
                
                # 从合并后的轨迹中计算统计信息
                final_statistics = TimingTracer.compute_statistics_from_traces(merged_traces)
                
                # 保存到文件
                trace_file = os.path.join(self.args.output_dir, "trace.json")
                self.timing_tracer.save_trace(trace_file, merged_traces, final_statistics)
                print(f"时序信息已保存到: {trace_file}")
        
        output = EvalLoopOutput(
            predictions=None,
            label_ids=None,
            metrics=metrics,
            num_samples=len(results), 
        )
        return output

    def evaluation_loop(
        self, 
        dataloader, 
        generation_config = None,         
        description: str = "Evaluation",
        metric_key_prefix: str = "eval",
        prediction_loss_only: bool = None,
        ignore_keys=None,
    ):
        model = self.model
        batch_size = dataloader.batch_size
        self.model.eval()

        if dist.get_rank() == 0:
            pbar = tqdm(dataloader, total=len(dataloader), desc=description)
        else:
            pbar = dataloader
        results = []
        acc_sum = 0
        sum = 0
        acc = 0
        loss_sum = 0
        
        for inputs in pbar:
            labels = inputs['labels']
            # labels_zh = inputs.pop("labels_zh")
            audios = inputs.pop('audios')
            texts = inputs.pop('texts')
            # texts_zh = inputs.pop('texts_zh')
            data_ids = inputs.pop('data_ids')

            inputs.pop("spectrograms_large")
            
            if self.model.model_type == "donut_only":
                inputs.pop("spectrograms")
                inputs["input_ids"] = labels[:, :1]
            elif self.model.model_type == "whisper_only":
                inputs["input_ids"] = labels[:, :4]
                inputs.pop("images")
                inputs.pop("images_len")
            else:
                inputs["input_ids"] = labels[:, :4]
            output = self.model(**inputs).logits[:, :-1, :]
            loss = nn.CrossEntropyLoss(ignore_index=-100)(output.reshape(-1, output.size(-1)), labels[:, 1:].reshape(-1))
            loss_sum += loss.item()
            loss = loss / len(data_ids)
            output = torch.argmax(output, dim=-1)
            output = torch.cat([labels[:, :1], output], dim=1).cpu()
            labels = labels.cpu()
            
            for i in range(len(data_ids)):
                results.append({
                    "id": data_ids[i],
                    "ref": texts[i],
                    'labels': labels[i],
                    # "ref_zh": texts_zh[i],
                    "pred": output[i]
                })
            pred_list = [item["pred"] for item in results]
            label_list = [item["labels"] for item in results]
            max_len = max(
                max(len(p) for p in pred_list),
                max(len(l) for l in label_list)
            )
            padded_preds = pad_sequence(pred_list, batch_first=True, padding_value=-100)
            padded_labels = pad_sequence(label_list, batch_first=True, padding_value=-100)
            predictions = self.pad_to_maxlen(padded_preds, max_len)
            label_ids = self.pad_to_maxlen(padded_labels, max_len)
            metrics = self.compute_metrics(EvalPrediction(predictions=predictions, label_ids=label_ids))
            acc_sum += metrics['acc_sum']
            sum += metrics['sum']
            acc = acc_sum / sum if sum > 0 else 0
            if dist.get_rank() == 0:
                pbar.set_postfix({'ACC': f'{acc*100:.2f}%', 'loss': f'{loss.item():.4f}'})

        # 1. 汇总数值型指标
        acc_sum_tensor = torch.tensor(acc_sum, dtype=torch.float32, device='cuda')
        sum_tensor = torch.tensor(sum, dtype=torch.float32, device='cuda')
        loss_sum_tensor = torch.tensor(loss_sum, dtype=torch.float32, device='cuda')
        dist.all_reduce(acc_sum_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(loss_sum_tensor, op=dist.ReduceOp.SUM)
        global_acc = acc_sum_tensor.item() / sum_tensor.item() if sum_tensor.item() > 0 else 0

        # 2. 汇总所有进程的 results
        all_results = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(all_results, results)
        results = []
        for part in all_results:
            if part is not None:
                results.extend(part)

        global_loss = loss_sum_tensor.item() / len(results)

        metrics = {}
        metrics['accuracy'] = global_acc
        metrics['loss'] = global_loss
        metrics['results'] = results
        self.metrics = metrics
        output = EvalLoopOutput(
            predictions=None,
            label_ids=None,
            metrics=metrics,
            num_samples=len(results), 
        )
        return output
    
    @staticmethod
    def pad_to_maxlen(tensor, max_len, pad_value=-100):
        # tensor: [batch, cur_len]
        cur_len = tensor.size(1)
        if cur_len < max_len:
            pad = torch.full((tensor.size(0), max_len - cur_len), pad_value, dtype=tensor.dtype, device=tensor.device)
            tensor = torch.cat([tensor, pad], dim=1)
        return tensor

    def _save_checkpoint(self, model, trial):
        if dist.get_rank() != 0:
            return
        
        # 获取当前指标并确保删除results
        metrics = {}
        if hasattr(self, 'metrics') and self.metrics is not None:
            # 创建新的metrics字典，排除results
            metrics = {}
            for key, value in self.metrics.items():
                if key != 'results':
                    metrics[key] = value
        else:
            metrics = {}
        
        print('save_checkpoint metrics:', metrics)
        
        run_dir = self._get_output_dir(trial=trial)
        output_dir = run_dir
        
        # 保存模型权重
        param_grad_dic = {k: v.requires_grad for (k, v) in model.named_parameters()}
        state_dict = model.state_dict().copy()
        keys = list(state_dict.keys())
        for k in keys:
            if k in param_grad_dic.keys() and not param_grad_dic[k]:
                del state_dict[k]
        
        checkpoint_folder = os.path.join(output_dir, f"checkpoint-{self.state.global_step}")
        os.makedirs(checkpoint_folder, exist_ok=True)
        
        # 保存模型权重
        torch.save(state_dict, os.path.join(checkpoint_folder, "pytorch_model.bin"))
        
        # 保存训练状态
        state_dict = {
            "best_metric": None,
            "best_model_checkpoint": None,
            "epoch": self.state.epoch,
            "global_step": self.state.global_step,
            "metrics": metrics,
            "total_flos": getattr(self.state, "total_flos", 0),
        }
        
        # 更新best_metric和best_model_checkpoint
        if metrics:
            metric_to_check = "accuracy"
            metrics_dict = metrics.metrics if hasattr(metrics, 'metrics') else metrics
            metric_value = metrics_dict.get(metric_to_check)
            
            if metric_value:
                operator = True
                if operator:
                    better_than_best = metric_value > self._best_metric
                else:
                    better_than_best = metric_value < self._best_metric
                
                if better_than_best:
                    self._best_metric = metric_value
                    self._best_checkpoint = checkpoint_folder
                    print(f"\nNew best model! {metric_to_check} = {metric_value:.4f}")
                
                state_dict["best_metric"] = self._best_metric
                state_dict["best_model_checkpoint"] = self._best_checkpoint
        
        # 保存训练状态
        with open(os.path.join(output_dir, "trainer_state.json"), "w", encoding="utf-8") as f:
            json.dump(state_dict, f, indent=2, sort_keys=True)
        # 在checkpoint目录下也保存一份当前状态
        with open(os.path.join(checkpoint_folder, "trainer_state.json"), "w", encoding="utf-8") as f:
            json.dump(state_dict, f, indent=2, sort_keys=True)
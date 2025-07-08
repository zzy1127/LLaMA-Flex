import torch
from typing import List
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data import DataLoader
from ...extras import logging

logger = logging.get_logger(__name__)

class DynamicSelector:
    def __init__(self, dataset, accelerator, data_collator):
        self.dataset = dataset
        self.accelerator = accelerator
        self.seed = 42
        self.data_collator = data_collator
    
    def warmup(self, num_samples: int, replacement: bool) -> List[List[int]]:
        """
        在全局范围内随机抽取 num_samples 条样本索引，并将它们广播到所有进程，
        最后按 world_size 切分成各进程自己的索引列表。

        返回:
            split_indices: 长度为 world_size 的列表，每个元素是该 rank 的索引列表
        """
        # 进程数与当前 rank
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size = 1
            rank = 0

        if rank == 0:
            dataset_size = len(self.dataset)
            gen = torch.Generator()
            gen.manual_seed(self.seed)

            if replacement:
                # 有放回地随机抽样
                full_indices = torch.randint(
                    low=0, high=dataset_size, size=(num_samples,), generator=gen
                ).tolist()
            else:
                # 无放回抽样：随机打乱再切片
                if num_samples > dataset_size:
                    raise ValueError(
                        f"Cannot sample {num_samples} without replacement from {dataset_size} samples"
                    )
                full_indices = torch.randperm(dataset_size, generator=gen)[:num_samples].tolist()
        else:
            full_indices = None

        # 广播完整列表到所有进程
        # 注意：broadcast_object_list 要求传入 list 容器
        obj = [full_indices]
        if dist.is_available() and dist.is_initialized():
            dist.broadcast_object_list(obj, src=0)
            full_indices = obj[0]
        else:
            # 单卡直接使用主进程生成的
            full_indices = full_indices or []

        # 切分成各 rank 的子列表
        split_indices = [
            full_indices[i::world_size] for i in range(world_size)
        ]

        return split_indices


    def select(self, model, step_id: int, num_samples: int):
        """
        1) 各卡并行前向、收集 local_losses
        2) gather 到主进程
        3) 主进程按 loss 从大到小 topk
        4) 切分成 world_size 份
        5) 广播 splits 给所有 rank
        返回：长度为 world_size 的 list，每项是该 rank 的索引列表
        """
        model.eval()

        # 1) 构造 DataLoader，让 accelerate 自动注入 DistributedSampler 并做 device placement
        dataloader = DataLoader(
            self.dataset,
            batch_size=4,
            shuffle=False,
            num_workers=2,
            collate_fn=self.data_collator,
        )
        dataloader = self.accelerator.prepare(dataloader)
        device = self.accelerator.device

        # 2) 各卡跑前向，收集 loss
        local_losses = []
        for batch in tqdm(
            dataloader,
            desc=f"[Selector step {step_id}]",
            disable=not self.accelerator.is_main_process,
            dynamic_ncols=True
        ):
            # with torch.no_grad():
            #     loss = model(**batch).loss.detach().cpu()
            # local_losses.append(loss)
            first_key = next(iter(batch))
            batch_size = batch[first_key].size(0)
            fake_loss = torch.ones(batch_size, device=device)
            local_losses.append(fake_loss)
            
        local_losses = torch.cat(local_losses)

        # 3) 所有进程参与 gather，将各自的 local_losses 收到主进程
        gathered_losses = self.accelerator.gather(local_losses)

        # 4) 主进程做 topk 筛选并切分
        if self.accelerator.is_main_process:
            # 取 loss 最大的 num_samples 个索引
            topk = torch.topk(gathered_losses, k=num_samples, largest=True)
            sel = topk.indices.tolist()

            world_size = dist.get_world_size()
            splits = [sel[i::world_size] for i in range(world_size)]
        else:
            splits = None

        # 5) 广播 splits（in-place）
        splits_list = [splits]                    # 先包成 list
        dist.broadcast_object_list(splits_list, src=0)
        splits = splits_list[0]                   # 再取出来

        return splits



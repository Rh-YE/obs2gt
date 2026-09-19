"""与"公平比较协议"相关的 Lightning 回调, 独立于具体训练引擎。"""
import os

import torch
from pytorch_lightning.callbacks import Callback


class UnifiedMetricCheckpoint(Callback):
    """按与 loss_type 无关的统一指标 (val 上非加权 MAE-to-GT) 选择最优
    checkpoint, 修复审计条目 A8: 旧流程里 monitor="val/loss/total" 这个字符串
    在四个损失变体间相同, 但其数值定义随 loss_type 切换而不同 (chi2 的 total
    是 chi2 值, mse 的 total 是 mse 值)——用它挑 best 等价于"每个损失按自己的
    标准挑分数最高的自己", 不构成跨损失可比较的选择标准。

    2026-07-10 (paper_plan_v2): 统一尺从 MSE 改为 MAE, 与论文主评价尺
    (MAE-to-GT) 保持同构——率-失真曲线的"失真"读数与 checkpoint 选择用的是
    同一把尺, 不产生选点标准与判决标准不一致的缺口。

    每个 validation epoch 结束后, 对 val_dataloader 用当前权重做一次
    plain forward (不触发训练引擎各自的特殊推理路径, 如 R2R 的 M 次平均——那是
    最终测试阶段的推理方式, 与"训练过程中用来挑 checkpoint 的探针"是两件事,
    这里只需要一个快速、对所有引擎都可比的分数), 计算 mean(|pred-gt|), 全部
    损失变体用同一把尺子, 取该指标最小的 epoch 保存为 best_unified.ckpt。

    不依赖也不覆盖原有的 model.monitor / ModelCheckpoint / CheckpointLinker 行为
    (那套体系仍然产出 last.ckpt, 供不需要跨损失可比性的场景使用); 下游评测脚本
    应改为读取 best_unified.ckpt。
    """

    def __init__(self, max_val_batches: int = 20):
        super().__init__()
        self.max_val_batches = max_val_batches
        self.best_metric = float("inf")

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        val_loader = trainer.datamodule.val_dataloader() if trainer.datamodule is not None else None
        if val_loader is None:
            return

        device = pl_module.device
        was_training = pl_module.training
        pl_module.eval()

        total_sq_err, total_n = 0.0, 0
        for i, batch in enumerate(val_loader):
            if i >= self.max_val_batches:
                break
            if "gt" not in batch:
                # 数据集不提供 GT (真实观测/无监督评测场景), 该回调不适用, 直接跳过。
                pl_module.train(was_training)
                return
            img = batch[pl_module.input_key].to(device)
            gt = batch["gt"].to(device)
            # 与 inner_training_step (autoencoder.py:288-294,
            # direct_reconstruction.py _FixedTargetEngine) 保持相同的输入构造:
            # cat_invvar=True 时网络期望 [图像, sigma] 拼接的多通道输入, 单独传
            # 单通道图像会在第一层卷积报通道数不匹配。sigma_film 的 direct-
            # reconstruction 系引擎不使用 (forward/decode 不处理 sigma 参数,
            # 见 direct_reconstruction.py 的既有说明), 此处不特殊处理。
            x = img
            if getattr(pl_module, "with_invvar", False) and getattr(pl_module, "cat_invvar", False):
                if "error" not in batch:
                    pl_module.train(was_training)
                    return
                x = torch.cat([img, batch["error"].to(device)], dim=1)
            try:
                _, pred, _ = pl_module(x)
            except Exception:
                # 部分引擎 forward 需要额外参数 (如未来的条件化变体), 无法用
                # 最简单的单参数调用时静默跳过, 不影响正常训练。
                pl_module.train(was_training)
                return
            abs_err = (pred - gt).abs()
            total_sq_err += abs_err.sum().item()
            total_n += abs_err.numel()

        pl_module.train(was_training)
        if total_n == 0:
            return
        metric = total_sq_err / total_n
        pl_module.log("val/mae_to_gt_unified", metric, prog_bar=True, logger=True,
                      sync_dist=True, batch_size=1)

        if metric < self.best_metric:
            self.best_metric = metric
            ckpt_dir = trainer.checkpoint_callback.dirpath if trainer.checkpoint_callback else None
            if ckpt_dir is None:
                return
            os.makedirs(ckpt_dir, exist_ok=True)
            dst = os.path.join(ckpt_dir, "best_unified.ckpt")
            trainer.save_checkpoint(dst)


class OwnLossBestCheckpoint(Callback):
    """按每个 run 自己的训练目标损失 (model.monitor, 即 val/loss/{loss_type})
    选择最优 checkpoint, 存为 best_own_loss.ckpt——与 UnifiedMetricCheckpoint
    (跨损失统一 MAE 尺子, best_unified.ckpt) 是两把不同的尺, 都保留、不互相
    覆盖。

    背景: chi2/mse/mae 三个 run 此前都只有 best_unified.ckpt 和逐 epoch/每
    10k 步的稀疏快照 (last.ckpt 用于恢复训练, 不代表最优点)。用户明确要求
    "各自挑各自训练损失的最低点", 而不是统一尺——统一尺是审计条目 A8 修的
    另一个问题 (跨损失公平比较), 两者用途不同、不能互相替代。之前只能靠
    离线扫 wandb 历史反查最优 step、再去磁盘找最接近的稀疏快照做近似, 现在
    直接在训练时精确落盘。

    读的是 pl_module.log 已经记录的 `val/loss/{loss_type}` 这个标量 (与
    model.monitor 同一个 key), 不重新做一次推理——UnifiedMetricCheckpoint
    需要自己跑 forward 是因为它要的指标 (跨损失统一 MAE) 训练引擎不会自己算;
    这里要的指标就是训练引擎自己已经算好并 log 过的损失, 直接读 trainer 的
    logged metrics 即可, 且不与 R2R 等特殊推理路径冲突。
    """

    def __init__(self):
        super().__init__()
        self.best_metric = float("inf")

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        monitor_key = getattr(pl_module, "monitor", None)
        if not monitor_key:
            return
        metrics = trainer.callback_metrics
        if monitor_key not in metrics:
            return
        value = metrics[monitor_key]
        value = float(value.detach().cpu()) if hasattr(value, "detach") else float(value)

        if value < self.best_metric:
            self.best_metric = value
            ckpt_dir = trainer.checkpoint_callback.dirpath if trainer.checkpoint_callback else None
            if ckpt_dir is None:
                return
            os.makedirs(ckpt_dir, exist_ok=True)
            dst = os.path.join(ckpt_dir, "best_own_loss.ckpt")
            trainer.save_checkpoint(dst)


class PlateauEarlyStopping(Callback):
    """paper_plan_v3 修订 2 的早停: "训练到收敛而非固定 epoch, 收敛看重建,
    不强求 KL 平台"。

    判据: 两条 val 曲线 (该 run 自己的训练目标 val/loss/total + 统一失真
    val/mae_to_gt_unified) 同时满足"最近 window 个 epoch 内相对波动幅度
    (max-min)/max(|max|,|min|) < rel_tol", 首次满足后再跑 confirm_epochs 个
    epoch 复核仍满足才停 (任一 epoch 判据破功则重新计时)。

    刻意不监控 KL/率 (2026-07-11 用户指正): β 极小时 KL 项对目标函数几乎
    没有贡献, 率的缓慢漂移是正常现象, 强求其平台可能永不触发; 率取停止
    时刻的实测值即可。率的尾部斜率由终评脚本另行记录, 供"容量上限"类
    结论筛选已平台化的 run。

    产出: 每个 val epoch 记 val/plateau_flag (0/1) 进 metrics.csv, 停止时
    print 收敛 epoch; 若跑满 max_epochs 仍未触发, flag 全 0 即"未收敛"标志。
    """

    def __init__(self, window: int = 15, rel_tol: float = 0.01,
                 confirm_epochs: int = 5, min_epochs: int = 30,
                 monitors=("val/loss/total", "val/mae_to_gt_unified")):
        super().__init__()
        self.window = window
        self.rel_tol = rel_tol
        self.confirm_epochs = confirm_epochs
        self.min_epochs = min_epochs
        self.monitors = list(monitors)
        self.history = {m: [] for m in self.monitors}
        self.plateau_since = None

    def _tail_is_flat(self, values):
        if len(values) < self.window:
            return False
        tail = values[-self.window:]
        lo, hi = min(tail), max(tail)
        denom = max(abs(hi), abs(lo), 1e-12)
        return (hi - lo) / denom < self.rel_tol

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        # 注意: val/mae_to_gt_unified 由 UnifiedMetricCheckpoint 在同一钩子里
        # 产出, 回调顺序不保证——取到的可能是上一 epoch 的值, 对 15-epoch 平台
        # 判据只是 1 epoch 的相位差, 无害。
        for m in self.monitors:
            v = trainer.callback_metrics.get(m)
            if v is None:
                return  # 任一监控量缺失则本 epoch 不判 (如 unified 回调被跳过)
            self.history[m].append(float(v))

        flat = all(self._tail_is_flat(self.history[m]) for m in self.monitors)
        pl_module.log("val/plateau_flag", 1.0 if flat else 0.0, logger=True,
                      sync_dist=True, batch_size=1)

        epoch = trainer.current_epoch
        if epoch + 1 < self.min_epochs:
            return
        if not flat:
            self.plateau_since = None
            return
        if self.plateau_since is None:
            self.plateau_since = epoch
            print(f"[PlateauEarlyStopping] epoch {epoch}: 双平台判据首次满足, "
                  f"再跑 {self.confirm_epochs} epoch 复核")
        elif epoch - self.plateau_since >= self.confirm_epochs:
            trainer.should_stop = True
            print(f"[PlateauEarlyStopping] epoch {epoch}: 平台保持 "
                  f"{epoch - self.plateau_since} epoch, 判定收敛, 停止训练")

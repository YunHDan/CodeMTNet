import math
from collections import Counter
from torch.optim.lr_scheduler import _LRScheduler


class MultiStepRestartLR(_LRScheduler):
    """ MultiStep with restarts learning rate scheme.

    Args:
        optimizer (torch.nn.optimizer): Torch optimizer.
        milestones (list): Iterations that will decrease learning rate.
        gamma (float): Decrease ratio. Default: 0.1.
        restarts (list): Restart iterations. Default: [0].
        restart_weights (list): Restart weights at each restart iteration.
            Default: [1].
        last_epoch (int): Used in _LRScheduler. Default: -1.
    """

    def __init__(self, optimizer, milestones, gamma=0.1, restarts=(0, ), restart_weights=(1, ), last_epoch=-1):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.restarts = restarts
        self.restart_weights = restart_weights
        assert len(self.restarts) == len(self.restart_weights), 'restarts and their weights do not match.'
        super(MultiStepRestartLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch in self.restarts:
            weight = self.restart_weights[self.restarts.index(self.last_epoch)]
            return [group['initial_lr'] * weight for group in self.optimizer.param_groups]
        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma**self.milestones[self.last_epoch] for group in self.optimizer.param_groups]


def get_position_from_periods(iteration, cumulative_period):
    """Get the position from a period list.

    It will return the index of the right-closest number in the period list.
    For example, the cumulative_period = [100, 200, 300, 400],
    if iteration == 50, return 0;
    if iteration == 210, return 2;
    if iteration == 300, return 2.

    Args:
        iteration (int): Current iteration.
        cumulative_period (list[int]): Cumulative period list.

    Returns:
        int: The position of the right-closest number in the period list.
    """
    for i, period in enumerate(cumulative_period):
        if iteration <= period:
            return i


class CosineAnnealingRestartLR(_LRScheduler):
    """ Cosine annealing with restarts learning rate scheme.

    An example of config:
    periods = [10, 10, 10, 10]
    restart_weights = [1, 0.5, 0.5, 0.5]
    eta_min=1e-7

    It has four cycles, each has 10 iterations. At 10th, 20th, 30th, the
    scheduler will restart with the weights in restart_weights.

    Args:
        optimizer (torch.nn.optimizer): Torch optimizer.
        periods (list): Period for each cosine anneling cycle.
        restart_weights (list): Restart weights at each restart iteration.
            Default: [1].
        eta_min (float): The mimimum lr. Default: 0.
        last_epoch (int): Used in _LRScheduler. Default: -1.
    """

    def __init__(self, optimizer, periods, restart_weights=(1, ), eta_min=0, last_epoch=-1):
        self.periods = periods
        self.restart_weights = restart_weights
        self.eta_min = eta_min
        assert (len(self.periods) == len(
            self.restart_weights)), 'periods and restart_weights should have the same length.'
        self.cumulative_period = [sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))]
        super(CosineAnnealingRestartLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        idx = get_position_from_periods(self.last_epoch, self.cumulative_period)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_period[idx - 1]
        current_period = self.periods[idx]

        return [
            self.eta_min + current_weight * 0.5 * (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * ((self.last_epoch - nearest_restart) / current_period)))
            for base_lr in self.base_lrs
        ]


class CosineAnnealingRestartCyclicLR(_LRScheduler):
    """ Cosine annealing with restarts learning rate scheme.
    An example of config:
    periods = [10, 10, 10, 10]
    restart_weights = [1, zero.5, zero.5, zero.5]
    eta_min=1e-7
    It has four cycles, each has 10 iterations. At 10th, 20th, 30th, the
    scheduler will restart with the weights in restart_weights.
    Args:
        optimizer (torch.nn.optimizer): Torch optimizer.
        periods (list): Period for each cosine anneling cycle.
        restart_weights (list): Restart weights at each restart iteration.
            Default: [1].
        eta_min (float): The mimimum lr. Default: zero.
        last_epoch (int): Used in _LRScheduler. Default: -1.
    """

    def __init__(self,
                 optimizer,  # PyTorch优化器对象，用于更新模型参数。
                 periods,  # 一个列表，其中每个元素表示一个余弦退火周期的长度（以迭代次数为单位）
                 restart_weights=(1,),  # 一个列表，其中每个元素表示在每个重启迭代时的权重。默认值为[1]，意味着每个周期的重启权重都是1。
                 eta_mins=(0,),  # 一个列表，其中每个元素表示在每个周期中的最小学习率。
                 last_epoch=-1):  # 上一个epoch的索引，用于在继续训练时继续调度。默认为-1，表示新的开始。
        self.periods = periods
        self.restart_weights = restart_weights
        self.eta_mins = eta_mins
        assert (len(self.periods) == len(self.restart_weights)
                ), 'periods and restart_weights should have the same length.'
        self.cumulative_period = [
            sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))
        ]  # cumulative_period 的每个元素表示从训练开始到当前周期结束时的总迭代次数。
        # 帮助确定在给定迭代次数（self.last_epoch）下，模型当前处于哪个周期。
        super(CosineAnnealingRestartCyclicLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        idx = get_position_from_periods(self.last_epoch,
                                        self.cumulative_period)  # 确定当前迭代位于哪个周期
        current_weight = self.restart_weights[idx]  # 根据当前周期获取重置的权重
        nearest_restart = 0 if idx == 0 else self.cumulative_period[idx - 1]  # 获取最近一个重启点的位置
        current_period = self.periods[idx]  # 当前周期的长度
        eta_min = self.eta_mins[idx]  # 当前周期允许的最小学习率

        return [
            eta_min + current_weight * 0.5 * (float(base_lr) - float(eta_min)) *
            (1 + math.cos(math.pi * (
                    (self.last_epoch - nearest_restart) / current_period)))
            for base_lr in self.base_lrs
        ]
        # 使用余弦退火公式计算新的学习率，并返回一个学习率列表，每个基础学习率对应一个优化器中的参数组
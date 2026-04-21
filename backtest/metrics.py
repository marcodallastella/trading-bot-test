import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class BacktestMetrics:
    total_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    profit_factor: float
    kelly_fraction: float
    total_trades: int
    winning_trades: int
    losing_trades: int


def compute_metrics(
    equity_curve: pd.Series,
    trade_pnls: list[float],
) -> BacktestMetrics:
    n = len(equity_curve)
    start = float(equity_curve.iloc[0])
    end   = float(equity_curve.iloc[-1])

    total_return_pct = round((end - start) / start * 100, 4)

    years = n / 252.0
    if years > 0 and start > 0:
        annualized_return_pct = round(((end / start) ** (1 / years) - 1) * 100, 4)
    else:
        annualized_return_pct = 0.0

    daily_returns = equity_curve.pct_change().dropna()

    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe_ratio = round(float(daily_returns.mean() / daily_returns.std() * np.sqrt(252)), 4)
    else:
        sharpe_ratio = 0.0

    downside = daily_returns[daily_returns < 0]
    if len(downside) > 1 and downside.std() > 0:
        sortino_ratio = round(float(daily_returns.mean() / downside.std() * np.sqrt(252)), 4)
    else:
        sortino_ratio = float("inf") if daily_returns.mean() > 0 else 0.0

    rolling_max = equity_curve.cummax()
    drawdowns = (equity_curve - rolling_max) / rolling_max * 100
    max_drawdown_pct = round(float(abs(drawdowns.min())), 4)

    winners = [p for p in trade_pnls if p > 0]
    losers  = [p for p in trade_pnls if p <= 0]
    total_trades = len(trade_pnls)
    winning_trades = len(winners)
    losing_trades  = len(losers)

    win_rate_pct = round(winning_trades / total_trades * 100, 4) if total_trades > 0 else 0.0

    gross_profit = sum(winners)
    gross_loss   = abs(sum(losers))
    profit_factor = round(gross_profit / gross_loss, 4) if gross_loss > 0 else float("inf")

    if total_trades > 0 and losing_trades > 0:
        w = winning_trades / total_trades
        avg_win  = gross_profit / winning_trades if winning_trades > 0 else 0.0
        avg_loss = gross_loss / losing_trades
        r = avg_win / avg_loss if avg_loss > 0 else 0.0
        kelly_fraction = round(w - (1 - w) / r, 4) if r > 0 else 0.0
    else:
        kelly_fraction = 0.0

    return BacktestMetrics(
        total_return_pct=total_return_pct,
        annualized_return_pct=annualized_return_pct,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        max_drawdown_pct=max_drawdown_pct,
        win_rate_pct=win_rate_pct,
        profit_factor=profit_factor,
        kelly_fraction=kelly_fraction,
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
    )

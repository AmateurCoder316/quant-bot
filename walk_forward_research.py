import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

from strategy_core import COMMISSION_RATE, SLIPPAGE_RATE
from strategy_research import (
    CANDIDATES,
    download_universe,
    research_score,
    run_candidate,
    trading_days,
)


TOP_N = 5
DEFAULT_WORKERS = 4

# Worker processes receive these once when the pool starts. On Linux we use
# fork, so the large read-only market-data frames are shared copy-on-write
# instead of being serialized for every one of the 216 candidate jobs.
_WORKER_FRAMES = None
_WORKER_INDEX = None
_WORKER_TRAIN_DAYS = None


def days_for_years(index, years):
    years = set(years)
    return [day for day in trading_days(index) if day.year in years]


def available_years(index):
    return sorted({day.year for day in trading_days(index)})


def format_metrics(m):
    return (
        f"ret={m['return']:+6.2f}% "
        f"sharpe={m['sharpe']:+5.2f} "
        f"PF={m['profit_factor']:.2f} "
        f"trades={m['trades']:>4} "
        f"avg={m['avg_trade']:+.3f}%"
    )


def worker_count():
    override = os.getenv("QUANT_RESEARCH_WORKERS")
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass

    cpu_count = os.cpu_count() or 1
    return max(1, min(DEFAULT_WORKERS, cpu_count - 1 if cpu_count > 1 else 1))


def _init_worker(frames, index, train_days):
    global _WORKER_FRAMES, _WORKER_INDEX, _WORKER_TRAIN_DAYS
    _WORKER_FRAMES = frames
    _WORKER_INDEX = index
    _WORKER_TRAIN_DAYS = train_days


def _evaluate_training_candidate(params):
    _, metrics = run_candidate(
        _WORKER_FRAMES,
        _WORKER_INDEX,
        _WORKER_TRAIN_DAYS,
        params,
        COMMISSION_RATE,
        SLIPPAGE_RATE,
    )
    return params, metrics


def rank_candidates(frames, index, train_days):
    ranked = []
    workers = worker_count()

    print(f"    parallel workers: {workers}", flush=True)

    # Linux/EndeavourOS supports fork. It is useful here because the parent has
    # already loaded ~373k bars, and the children can initially share those
    # pages instead of each receiving a fresh serialized copy per candidate.
    context = get_context("fork")

    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_init_worker,
        initargs=(frames, index, train_days),
    ) as executor:
        futures = [executor.submit(_evaluate_training_candidate, params) for params in CANDIDATES]

        completed = 0
        for future in as_completed(futures):
            params, m = future.result()
            ranked.append((research_score(m), params, m))
            completed += 1

            if completed % 10 == 0 or completed == len(CANDIDATES):
                print(
                    f"    evaluated {completed:>3}/{len(CANDIDATES)} candidates "
                    f"({completed / len(CANDIDATES) * 100:5.1f}%)",
                    flush=True,
                )

    ranked.sort(key=lambda row: row[0], reverse=True)
    return ranked


def evaluate_test(frames, index, test_days, params):
    cost_result, cost_metrics = run_candidate(
        frames,
        index,
        test_days,
        params,
        COMMISSION_RATE,
        SLIPPAGE_RATE,
    )

    gross_result, gross_metrics = run_candidate(
        frames,
        index,
        test_days,
        params,
        0.0,
        0.0,
    )

    friction = cost_result[3] + cost_result[4]
    break_even_round_trip = max(0.0, gross_metrics["avg_trade"])

    return {
        "cost": cost_metrics,
        "gross": gross_metrics,
        "friction": friction,
        "break_even_round_trip": break_even_round_trip,
        "break_even_per_side": break_even_round_trip / 2,
    }


def main():
    frames, index = download_universe()
    years = available_years(index)

    if len(years) < 2:
        raise ValueError("Walk-forward research needs at least two calendar years of data.")

    print("=" * 104)
    print("BT5 MULTI-YEAR WALK-FORWARD RESEARCH")
    print("=" * 104)
    print(f"Years available: {', '.join(map(str, years))}")
    print(f"Candidates:       {len(CANDIDATES)}")
    print(f"Parallel workers: {worker_count()}")
    print(
        f"Cost model:       commission={COMMISSION_RATE:.3%}/side, "
        f"slippage={SLIPPAGE_RATE:.3%}/side"
    )
    print()
    print("For each fold, candidates are selected using ONLY earlier years.")
    print("The following year is then evaluated without changing the selected parameters.")

    fold_results = []
    selection_counts = defaultdict(int)

    for test_position in range(1, len(years)):
        train_years = years[:test_position]
        test_year = years[test_position]
        train_days = days_for_years(index, train_years)
        test_days = days_for_years(index, [test_year])

        if not train_days or not test_days:
            continue

        print("\n" + "=" * 104)
        print(f"FOLD: train {train_years[0]}-{train_years[-1]} -> unseen {test_year}")
        print("=" * 104)
        print(
            f"Training days: {len(train_days)} | Test days: {len(test_days)} | "
            f"Training candidates: {len(CANDIDATES)}"
        )

        ranked = rank_candidates(frames, index, train_days)
        finalists = ranked[:TOP_N]

        print("\nTOP TRAINING CANDIDATES")
        for rank, (_, params, train_m) in enumerate(finalists, 1):
            print(f"  {rank}. {params.name:<58} {format_metrics(train_m)}")

        # Rank #1 is frozen before seeing the next year's results.
        _, selected_params, selected_train_m = finalists[0]
        selection_counts[selected_params.name] += 1

        test = evaluate_test(frames, index, test_days, selected_params)

        print("\nWALK-FORWARD RESULT")
        print(f"  selected:   {selected_params.name}")
        print(f"  train:      {format_metrics(selected_train_m)}")
        print(f"  test costs: {format_metrics(test['cost'])}")
        print(f"  test gross: {format_metrics(test['gross'])}")
        print(f"  friction:   ${test['friction']:,.2f}")
        print(
            f"  break-even: {test['break_even_round_trip']:.3f}% round trip "
            f"({test['break_even_per_side']:.3f}%/side)"
        )

        fold_results.append(
            {
                "train_years": train_years,
                "test_year": test_year,
                "params": selected_params,
                "train": selected_train_m,
                "test_cost": test["cost"],
                "test_gross": test["gross"],
                "friction": test["friction"],
            }
        )

    print("\n" + "=" * 104)
    print("WALK-FORWARD SUMMARY")
    print("=" * 104)

    if not fold_results:
        print("No valid folds were produced.")
        return

    profitable_gross = sum(row["test_gross"]["return"] > 0 for row in fold_results)
    profitable_cost = sum(row["test_cost"]["return"] > 0 for row in fold_results)
    total_test_trades = sum(row["test_cost"]["trades"] for row in fold_results)
    total_friction = sum(row["friction"] for row in fold_results)

    for row in fold_results:
        train_label = f"{row['train_years'][0]}-{row['train_years'][-1]}"
        print(
            f"{train_label:>9} -> {row['test_year']}: "
            f"gross={row['test_gross']['return']:+6.2f}% "
            f"cost={row['test_cost']['return']:+6.2f}% "
            f"PFgross={row['test_gross']['profit_factor']:.2f} "
            f"trades={row['test_cost']['trades']:>4} | {row['params'].name}"
        )

    print()
    print(f"Gross-profitable unseen years: {profitable_gross}/{len(fold_results)}")
    print(f"Cost-profitable unseen years:  {profitable_cost}/{len(fold_results)}")
    print(f"Total unseen trades:           {total_test_trades}")
    print(f"Total modeled friction:        ${total_friction:,.2f}")

    print("\nPARAMETER STABILITY")
    for name, count in sorted(selection_counts.items(), key=lambda item: item[1], reverse=True):
        print(f"  selected in {count} fold(s): {name}")

    print("\nINTERPRETATION")
    print("A robust strategy should not depend on one lucky year or one exact parameter combination.")
    print("We care about repeatable unseen gross edge first, then whether that edge survives realistic costs.")
    print("Do not optimize against the final unseen year after seeing this report; doing so would leak test data.")


if __name__ == "__main__":
    main()

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import argparse
import random
import time


EPSILON = 1e-6


@dataclass
class Node:
    key: int
    probability: float
    left: Optional["Node"] = None
    right: Optional["Node"] = None


def validate_input(keys: list[int], probabilities: list[float]) -> None:
    if not keys:
        raise ValueError("At least one key is required.")

    if len(keys) != len(probabilities):
        raise ValueError("The number of keys and probabilities must be equal.")

    if keys != sorted(keys):
        raise ValueError("Keys must be given in sorted ascending order.")

    if len(set(keys)) != len(keys):
        raise ValueError("Keys must be unique.")

    if any(p < 0 for p in probabilities):
        raise ValueError("Probabilities cannot be negative.")

    total = sum(probabilities)
    if abs(total - 1.0) > EPSILON:
        raise ValueError(
            f"Probabilities must sum approximately to 1. Current sum = {total:.8f}"
        )


def interval_sum(prefix: list[float], i: int, j: int) -> float:
    """Return sum(probabilities[i:j+1]) in O(1)."""
    return prefix[j + 1] - prefix[i]


def optimal_bst(
    keys: list[int], probabilities: list[float]
) -> tuple[list[list[float]], list[list[Optional[int]]], Optional[Node]]:
    """
    Construct the OBST using standard O(n^3) dynamic programming.

    Returns:
        cost_table: C[i][j]
        root_table: index of selected root for subproblem i..j
        root: reconstructed optimal BST
    """
    validate_input(keys, probabilities)

    n = len(keys)
    cost = [[0.0 for _ in range(n)] for _ in range(n)]
    root_choice: list[list[Optional[int]]] = [
        [None for _ in range(n)] for _ in range(n)
    ]

    prefix = [0.0] * (n + 1)
    for i, p in enumerate(probabilities):
        prefix[i + 1] = prefix[i] + p

    for i in range(n):
        cost[i][i] = probabilities[i]
        root_choice[i][i] = i

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            weight = interval_sum(prefix, i, j)

            best_cost = float("inf")
            best_root = None

            for r in range(i, j + 1):
                left_cost = cost[i][r - 1] if r > i else 0.0
                right_cost = cost[r + 1][j] if r < j else 0.0

                candidate = left_cost + right_cost + weight

                if candidate < best_cost:
                    best_cost = candidate
                    best_root = r

            cost[i][j] = best_cost
            root_choice[i][j] = best_root

    tree_root = reconstruct_tree(keys, probabilities, root_choice, 0, n - 1)
    return cost, root_choice, tree_root


def reconstruct_tree(
    keys: list[int],
    probabilities: list[float],
    root_table: list[list[Optional[int]]],
    i: int,
    j: int,
) -> Optional[Node]:
    if i > j:
        return None

    r = root_table[i][j]
    if r is None:
        return None

    node = Node(keys[r], probabilities[r])
    node.left = reconstruct_tree(keys, probabilities, root_table, i, r - 1)
    node.right = reconstruct_tree(keys, probabilities, root_table, r + 1, j)
    return node


def expected_search_cost(root: Optional[Node], level: int = 1) -> float:
    if root is None:
        return 0.0

    return (
        root.probability * level
        + expected_search_cost(root.left, level + 1)
        + expected_search_cost(root.right, level + 1)
    )


def collect_depths(
    root: Optional[Node],
    depth: int = 0,
    result: Optional[list[tuple[int, float, int, int]]] = None,
) -> list[tuple[int, float, int, int]]:
    if result is None:
        result = []

    if root is None:
        return result

    result.append((root.key, root.probability, depth, depth + 1))
    collect_depths(root.left, depth + 1, result)
    collect_depths(root.right, depth + 1, result)
    return result


def average_depth(root: Optional[Node]) -> float:
    info = collect_depths(root)
    if not info:
        return 0.0
    return sum(depth for _, _, depth, _ in info) / len(info)


def print_tree(
    root: Optional[Node],
    prefix: str = "",
    is_left: bool = True,
) -> None:
    if root is None:
        return

    if root.right:
        print_tree(
            root.right,
            prefix + ("│   " if is_left else "    "),
            False,
        )

    print(prefix + ("└── " if is_left else "┌── ") + str(root.key))

    if root.left:
        print_tree(
            root.left,
            prefix + ("    " if is_left else "│   "),
            True,
        )


def print_cost_table(keys: list[int], cost: list[list[float]]) -> None:
    n = len(keys)

    print("\nDYNAMIC PROGRAMMING COST TABLE C[i,j]")
    print("-" * (12 + 12 * n))
    print(f"{'i\\j':>8}", end="")
    for key in keys:
        print(f"{key:>12}", end="")
    print()

    for i in range(n):
        print(f"{keys[i]:>8}", end="")
        for j in range(n):
            if j < i:
                print(f"{'-':>12}", end="")
            else:
                print(f"{cost[i][j]:>12.4f}", end="")
        print()


def print_root_table(
    keys: list[int], root_table: list[list[Optional[int]]]
) -> None:
    n = len(keys)

    print("\nROOT TABLE")
    print("-" * (12 + 12 * n))
    print(f"{'i\\j':>8}", end="")
    for key in keys:
        print(f"{key:>12}", end="")
    print()

    for i in range(n):
        print(f"{keys[i]:>8}", end="")
        for j in range(n):
            if j < i or root_table[i][j] is None:
                print(f"{'-':>12}", end="")
            else:
                selected_key = keys[root_table[i][j]]
                print(f"{selected_key:>12}", end="")
        print()


# Conventional BST

def bst_insert(root: Optional[Node], key: int, probability: float) -> Node:
    """Insert one key using ordinary BST insertion."""
    if root is None:
        return Node(key, probability)

    if key < root.key:
        root.left = bst_insert(root.left, key, probability)
    else:
        root.right = bst_insert(root.right, key, probability)

    return root


def build_conventional_bst(
    keys: list[int], probabilities: list[float]
) -> Optional[Node]:
    root = None
    for key, probability in zip(keys, probabilities):
        root = bst_insert(root, key, probability)
    return root


# Display / Analysis

def analyze_input(keys: list[int], probabilities: list[float]) -> None:
    """Run and display the complete analysis for one input set."""
    validate_input(keys, probabilities)

    print("\nINPUT")
    print("-" * 32)
    print(f"{'Key':>10} {'Probability':>16}")
    for key, probability in zip(keys, probabilities):
        print(f"{key:>10} {probability:>16.4f}")
    print(f"{'Total':>10} {sum(probabilities):>16.4f}")

    start = time.perf_counter()
    cost, roots, obst_root = optimal_bst(keys, probabilities)
    obst_time = time.perf_counter() - start

    print_cost_table(keys, cost)
    print_root_table(keys, roots)

    minimum_cost = cost[0][len(keys) - 1]
    verified_cost = expected_search_cost(obst_root)

    print("\nOPTIMAL BST")
    print("-" * 32)
    print_tree(obst_root)

    print("\nKEY DEPTH / LEVEL")
    print("-" * 48)
    print(f"{'Key':>10} {'Probability':>14} {'Depth':>10} {'Level':>10}")

    depth_data = sorted(collect_depths(obst_root), key=lambda item: item[0])
    for key, probability, depth, level in depth_data:
        print(f"{key:>10} {probability:>14.4f} {depth:>10} {level:>10}")

    print("\nOBST RESULTS")
    print("-" * 48)
    print(f"Minimum expected search cost : {minimum_cost:.6f}")
    print(f"Verified cost from final tree: {verified_cost:.6f}")
    print(f"Average node depth           : {average_depth(obst_root):.6f}")
    print(f"Construction time            : {obst_time:.8f} seconds")

    if abs(minimum_cost - verified_cost) <= EPSILON:
        print("Verification                 : PASS")
    else:
        print("Verification                 : FAIL")

    # Compare against ordinary BST insertion.
    start = time.perf_counter()
    conventional_root = build_conventional_bst(keys, probabilities)
    conventional_time = time.perf_counter() - start

    conventional_cost = expected_search_cost(conventional_root)
    conventional_avg_depth = average_depth(conventional_root)

    print("\nCONVENTIONAL BST")
    print("-" * 32)
    print_tree(conventional_root)

    print("\nCOMPARISON")
    print("-" * 72)
    print(
        f"{'Tree':<20}"
        f"{'Expected Cost':>18}"
        f"{'Avg. Depth':>16}"
        f"{'Build Time (s)':>18}"
    )
    print(
        f"{'Optimal BST':<20}"
        f"{minimum_cost:>18.6f}"
        f"{average_depth(obst_root):>16.6f}"
        f"{obst_time:>18.8f}"
    )
    print(
        f"{'Conventional BST':<20}"
        f"{conventional_cost:>18.6f}"
        f"{conventional_avg_depth:>16.6f}"
        f"{conventional_time:>18.8f}"
    )


# Experiments

def make_random_test_data(
    n: int, rng: random.Random
) -> tuple[list[int], list[float]]:
    """Generate reproducible sorted keys and random probabilities summing to 1."""
    keys = [10 * (i + 1) for i in range(n)]

    raw = [rng.random() + 0.01 for _ in range(n)]
    total = sum(raw)
    probabilities = [x / total for x in raw]

    return keys, probabilities


def run_experiments(
    sizes: list[int] = [5, 10, 20, 50, 100],
    seed: int = 42,
) -> None:
    rng = random.Random(seed)

    print("\nEXPERIMENTAL RESULTS")
    print("-" * 76)
    print(
        f"{'n':>8}"
        f"{'Execution Time (s)':>24}"
        f"{'Minimum Expected Cost':>26}"
        f"{'Remarks':>18}"
    )

    previous_time = None

    for n in sizes:
        keys, probabilities = make_random_test_data(n, rng)

        start = time.perf_counter()
        cost, _, _ = optimal_bst(keys, probabilities)
        elapsed = time.perf_counter() - start

        min_cost = cost[0][n - 1]

        if previous_time is None:
            remarks = "baseline"
        else:
            ratio = elapsed / previous_time if previous_time > 0 else 0.0
            remarks = f"{ratio:.2f}x previous"

        print(
            f"{n:>8}"
            f"{elapsed:>24.8f}"
            f"{min_cost:>26.6f}"
            f"{remarks:>18}"
        )

        previous_time = elapsed

    print("\nNote: Execution times vary by computer and between runs.")
    print("The theoretical time complexity of this implementation is O(n^3).")


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

def read_interactive_input() -> tuple[list[int], list[float]]:
    """Read project input interactively."""
    print("Optimal Binary Search Tree - Input")
    print("----------------------------------")

    n = int(input("Number of keys n: ").strip())
    if n <= 0:
        raise ValueError("n must be greater than 0.")

    print(f"Enter {n} sorted keys separated by spaces:")
    keys = list(map(int, input().split()))

    print(f"Enter {n} probabilities separated by spaces:")
    probabilities = list(map(float, input().split()))

    if len(keys) != n:
        raise ValueError(f"Expected {n} keys, but received {len(keys)}.")

    if len(probabilities) != n:
        raise ValueError(
            f"Expected {n} probabilities, but received {len(probabilities)}."
        )

    validate_input(keys, probabilities)
    return keys, probabilities


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimal Binary Search Tree using Dynamic Programming"
    )

    parser.add_argument(
        "--sample",
        action="store_true",
        help="Run the assignment's sample input.",
    )

    parser.add_argument(
        "--experiment",
        action="store_true",
        help="Run experiments for n = 5, 10, 20, 50, 100.",
    )

    args = parser.parse_args()

    try:
        if args.sample:
            keys = [10, 20, 30, 40, 50]
            probabilities = [0.10, 0.20, 0.40, 0.20, 0.10]
            analyze_input(keys, probabilities)

        elif args.experiment:
            run_experiments()

        else:
            keys, probabilities = read_interactive_input()
            analyze_input(keys, probabilities)

    except ValueError as error:
        print(f"\nInput error: {error}")


if __name__ == "__main__":
    main()

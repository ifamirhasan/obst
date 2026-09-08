# Optimal Binary Search Tree (OBST) Project

## Requirements

- Python 3.10 or newer
- No external Python packages are required.

## Files

- `obst_project.py` - Complete Python implementation.
- `sample_input.txt` - Example input values for reference.

## Run interactively

```bash
python obst_project.py
```

Example values:

```text
Number of keys n: 5
Enter 5 sorted keys separated by spaces:
10 20 30 40 50
Enter 5 probabilities separated by spaces:
0.10 0.20 0.40 0.20 0.10
```

## Run the built-in sample

```bash
python obst_project.py --sample
```

## Run the required experiments

```bash
python obst_project.py --experiment
```

This tests:

- n = 5
- n = 10
- n = 20
- n = 50
- n = 100

The generated probabilities are reproducible because the experiment uses a fixed random seed.

## Program output

For a normal input, the program prints:

1. Keys and probabilities
2. Dynamic Programming cost table
3. Root table
4. Optimal BST structure
5. Depth and level of every key
6. Minimum expected search cost
7. Expected cost calculated again from the reconstructed tree
8. Verification result
9. Construction time
10. Conventional BST
11. Comparison of OBST vs conventional BST

## Dynamic Programming recurrence

For keys i through j:

```text
C[i,j] = min over r=i..j (
    C[i,r-1] + C[r+1,j]
) + sum(p[i..j])
```

Base case:

```text
C[i,i] = p[i]
```

An empty subtree has cost 0.

## Complexity

There are O(n^2) intervals `(i,j)`.

For every interval, the algorithm tries up to O(n) possible roots.

Therefore:

```text
Time complexity = O(n^3)
Space complexity = O(n^2)
```

The tree reconstruction itself is O(n).

## Important note about the conventional BST comparison

The assignment requires sorted input. The comparison BST in this implementation uses ordinary insertion in the supplied order.

Therefore, when sorted keys are inserted one by one, the conventional BST becomes right-skewed. This demonstrates why the arrangement of keys can significantly affect search cost.

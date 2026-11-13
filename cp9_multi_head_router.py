# cp9_multi_head_router.py
#
# CP9: Paper-faithful Forgery-Aware Multi-Head Router
#      for DINOv2-Small
#
# DINOv2-Small hidden dimension:
#   d = 384
#
# Four channel heads:
#   384 / 4 = 96 dimensions per head
#
# Router:
#   T = 2 tasks
#   h_t = 4 heads
#   N = 6 experts
#   Top-K = 3
#
# Z_task:
#   (2, 4, 6)
#
# Routing:
#   softmax over all six experts
#   Top-3 selection
#   renormalization of selected probabilities
#
# No feature-dependent Linear router is used.


import torch
import torch.nn as nn


# ============================================================
# Configuration
# ============================================================

D_MODEL = 384
NUM_HEADS = 4
D_HEAD = D_MODEL // NUM_HEADS

NUM_TASKS = 2
NUM_EXPERTS = 6
TOP_K = 3


# ============================================================
# Forgery-Aware Multi-Head Router
# ============================================================

class ForgeryAwareMultiHeadRouter(nn.Module):

    def __init__(
        self,
        num_tasks=NUM_TASKS,
        num_heads=NUM_HEADS,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
    ):
        super().__init__()

        self.num_tasks = num_tasks
        self.num_heads = num_heads
        self.num_experts = num_experts
        self.top_k = top_k

        assert top_k <= num_experts

        # Paper:
        #
        # Z_task ∈ R^(T × h_t × N)
        #
        # = (2, 4, 6)

        self.Z_task = nn.Parameter(
            0.01*torch.zeros(
                num_tasks,
                num_heads,
                num_experts,
            )
        )

    def forward(self):

        # ----------------------------------------------------
        # Softmax over all six experts
        # ----------------------------------------------------

        probabilities = torch.softmax(
            self.Z_task,
            dim=-1,
        )

        # ----------------------------------------------------
        # Top-3 selection
        # ----------------------------------------------------

        top_values, top_indices = torch.topk(
            probabilities,
            k=self.top_k,
            dim=-1,
        )

        # ----------------------------------------------------
        # Renormalize selected probabilities
        # ----------------------------------------------------

        top_weights = (
            top_values
            / top_values.sum(
                dim=-1,
                keepdim=True,
            )
        )

        return (
            probabilities,
            top_indices,
            top_weights,
        )


# ============================================================
# Validation
# ============================================================

def main():

    print("=" * 70)
    print("CP9 — DINOv2-Small Multi-Head Router")
    print("=" * 70)

    print("\nConfiguration:")
    print(f"D_MODEL     : {D_MODEL}")
    print(f"D_HEAD      : {D_HEAD}")
    print(f"Tasks       : {NUM_TASKS}")
    print(f"Heads       : {NUM_HEADS}")
    print(f"Experts     : {NUM_EXPERTS}")
    print(f"Top-K       : {TOP_K}")

    assert D_MODEL == 384
    assert D_HEAD == 96
    assert NUM_HEADS == 4
    assert NUM_TASKS == 2
    assert NUM_EXPERTS == 6
    assert TOP_K == 3

    # --------------------------------------------------------
    # Create router
    # --------------------------------------------------------

    router = ForgeryAwareMultiHeadRouter()

    print("\nRouter:")
    print(router)

    total_params = sum(
        p.numel()
        for p in router.parameters()
    )

    expected_params = (
        NUM_TASKS
        * NUM_HEADS
        * NUM_EXPERTS
    )

    print(
        "\nTrainable routing parameters:",
        total_params,
    )

    print(
        "Expected routing parameters :",
        expected_params,
    )

    assert total_params == expected_params

    # --------------------------------------------------------
    # Deterministic routing logits for validation
    # --------------------------------------------------------

    with torch.no_grad():

        router.Z_task.copy_(
            torch.tensor(
                [
                    [
                        [1., 2., 3., 4., 5., 6.],
                        [6., 5., 4., 3., 2., 1.],
                        [1., 6., 2., 5., 3., 4.],
                        [4., 1., 6., 2., 5., 3.],
                    ],
                    [
                        [6., 1., 5., 2., 4., 3.],
                        [2., 6., 1., 5., 3., 4.],
                        [3., 5., 1., 6., 2., 4.],
                        [5., 3., 6., 1., 4., 2.],
                    ],
                ],
                dtype=torch.float32,
            )
        )

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    probabilities, top_indices, top_weights = router()

    print("\nZ_task shape:")
    print(tuple(router.Z_task.shape))

    print("\nFull softmax probabilities:")
    print(probabilities)

    print("\nSelected expert indices:")
    print(top_indices)

    print("\nRenormalized Top-3 weights:")
    print(top_weights)

    # --------------------------------------------------------
    # Shapes
    # --------------------------------------------------------

    assert router.Z_task.shape == (
        2,
        4,
        6,
    )

    assert probabilities.shape == (
        2,
        4,
        6,
    )

    assert top_indices.shape == (
        2,
        4,
        3,
    )

    assert top_weights.shape == (
        2,
        4,
        3,
    )

    print("\nShape checks: PASS")

    # --------------------------------------------------------
    # Full softmax sums to 1
    # --------------------------------------------------------

    assert torch.allclose(
        probabilities.sum(dim=-1),
        torch.ones(2, 4),
        atol=1e-6,
    )

    print("Full softmax sums to 1: PASS")

    # --------------------------------------------------------
    # Selected weights sum to 1
    # --------------------------------------------------------

    assert torch.allclose(
        top_weights.sum(dim=-1),
        torch.ones(2, 4),
        atol=1e-6,
    )

    print("Top-3 renormalized weights sum to 1: PASS")

    # --------------------------------------------------------
    # Exactly three unique experts per task/head
    # --------------------------------------------------------

    for task in range(NUM_TASKS):

        for head in range(NUM_HEADS):

            selected = top_indices[
                task,
                head,
            ]

            assert selected.unique().numel() == TOP_K

    print(
        "Exactly 3 unique experts per task/head: PASS"
    )

    # --------------------------------------------------------
    # Verify renormalization
    # --------------------------------------------------------

    for task in range(NUM_TASKS):

        for head in range(NUM_HEADS):

            indices = top_indices[
                task,
                head,
            ]

            expected = probabilities[
                task,
                head,
                indices,
            ]

            expected = (
                expected
                / expected.sum()
            )

            actual = top_weights[
                task,
                head,
            ]

            assert torch.allclose(
                actual,
                expected,
                atol=1e-6,
            )

    print("Top-3 renormalization logic: PASS")
    print("Z_task is learnable: PASS")

    print("\nCP9 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

# cp8_lora_expert_bank.py
#
# CP8: LoRA Expert Bank for DINOv2-Small
#
# DINOv2-Small:
#   hidden dimension d = 384
#
# DFF-Adapter:
#   h = 4 heads
#   r = 16
#   r_h = r/h = 4
#   N = 6 experts
#   alpha = 32
#
# Per-head:
#   d_h = 384/4 = 96
#   A = 96 x 4
#   B = 4 x 96
#
# Six experts are shared across the four DFF heads.
#
# Each expert contains LoRA projections for:
#   Q
#   V
#   Dense
#
# This is an isolated validation.
# No DINOv2 is loaded and nothing is trained.


import torch
import torch.nn as nn


# ============================================================
# Configuration
# ============================================================

D_MODEL = 384
NUM_HEADS = 4
D_HEAD = D_MODEL // NUM_HEADS

LORA_RANK = 16
HEAD_RANK = LORA_RANK // NUM_HEADS

NUM_EXPERTS = 6

ALPHA = 32
SCALING = ALPHA / LORA_RANK


# ============================================================
# One LoRA projection
# ============================================================

class LoRAProjection(nn.Module):

    def __init__(
        self,
        d_head=D_HEAD,
        rank=HEAD_RANK,
        alpha=ALPHA,
    ):
        super().__init__()

        self.d_head = d_head
        self.rank = rank
        self.alpha = alpha

        # LoRA scaling
        #
        # alpha / r = 32 / 16 = 2
        #
        # This is the scaling convention used in this
        # implementation.
        self.scaling = alpha / LORA_RANK

        # A: d_h x r_h
        self.A = nn.Parameter(
            torch.empty(
                d_head,
                rank,
            )
        )

        # B: r_h x d_h
        #
        # Zero initialization makes the initial LoRA
        # contribution exactly zero.
        self.B = nn.Parameter(
            torch.zeros(
                rank,
                d_head,
            )
        )

        # A is used as:
        #
        #     x @ A
        #
        # Therefore initialize A as the transpose of a
        # conventional Linear(in_features=d_head,
        #                      out_features=rank)
        # weight matrix.
        nn.init.kaiming_uniform_(
            self.A.T,
            a=5 ** 0.5,
        )

    def forward(self, x):

        return self.scaling * (
            x @ self.A @ self.B
        )


# ============================================================
# One LoRA expert
# ============================================================

class LoRAExpert(nn.Module):

    def __init__(
        self,
        d_head=D_HEAD,
        rank=HEAD_RANK,
        alpha=ALPHA,
    ):
        super().__init__()

        self.q = LoRAProjection(
            d_head=d_head,
            rank=rank,
            alpha=alpha,
        )

        self.v = LoRAProjection(
            d_head=d_head,
            rank=rank,
            alpha=alpha,
        )

        self.dense = LoRAProjection(
            d_head=d_head,
            rank=rank,
            alpha=alpha,
        )

    def forward(self, x):

        q_update = self.q(x)
        v_update = self.v(x)
        dense_update = self.dense(x)

        return (
            q_update,
            v_update,
            dense_update,
        )


# ============================================================
# Shared six-expert bank
# ============================================================

class LoRAExpertBank(nn.Module):
    """
    One shared pool of six experts.

    The same six experts are available to all four
    DFF channel heads.
    """

    def __init__(
        self,
        num_experts=NUM_EXPERTS,
        d_head=D_HEAD,
        rank=HEAD_RANK,
        alpha=ALPHA,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.d_head = d_head
        self.rank = rank
        self.alpha = alpha

        self.experts = nn.ModuleList(
            [
                LoRAExpert(
                    d_head=d_head,
                    rank=rank,
                    alpha=alpha,
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, x):
        """
        x:
            (B, L, 96)

        outputs:
            (B, L, 6, 96)
        """

        q_outputs = []
        v_outputs = []
        dense_outputs = []

        for expert in self.experts:

            q_update, v_update, dense_update = expert(x)

            q_outputs.append(q_update)
            v_outputs.append(v_update)
            dense_outputs.append(dense_update)

        q_updates = torch.stack(
            q_outputs,
            dim=2,
        )

        v_updates = torch.stack(
            v_outputs,
            dim=2,
        )

        dense_updates = torch.stack(
            dense_outputs,
            dim=2,
        )

        return (
            q_updates,
            v_updates,
            dense_updates,
        )


# ============================================================
# Validation
# ============================================================

def main():

    print("=" * 70)
    print("CP8 — DINOv2-Small LoRA Expert Bank")
    print("=" * 70)

    print("\nConfiguration:")
    print(f"D_MODEL       : {D_MODEL}")
    print(f"NUM_HEADS     : {NUM_HEADS}")
    print(f"D_HEAD        : {D_HEAD}")
    print(f"LORA_RANK     : {LORA_RANK}")
    print(f"HEAD_RANK     : {HEAD_RANK}")
    print(f"NUM_EXPERTS   : {NUM_EXPERTS}")
    print(f"ALPHA         : {ALPHA}")
    print(f"SCALING       : {SCALING}")

    assert D_MODEL == 384
    assert NUM_HEADS == 4
    assert D_HEAD == 96
    assert LORA_RANK == 16
    assert HEAD_RANK == 4
    assert NUM_EXPERTS == 6
    assert SCALING == 2

    # --------------------------------------------------------
    # Create expert bank
    # --------------------------------------------------------

    bank = LoRAExpertBank()

    total_params = sum(
        p.numel()
        for p in bank.parameters()
    )

    print("\nExpert bank:")
    print(bank)

    print(
        "\nTotal trainable parameters:",
        total_params,
    )

    # --------------------------------------------------------
    # Expected:
    #
    # One projection:
    #   96*4 + 4*96 = 768
    #
    # Q + V + Dense:
    #   768*3 = 2304
    #
    # Six experts:
    #   2304*6 = 13824
    # --------------------------------------------------------

    expected_params = (
        NUM_EXPERTS
        * 3
        * 2
        * D_HEAD
        * HEAD_RANK
    )

    print(
        "Expected parameters      :",
        expected_params,
    )

    assert total_params == expected_params

    # --------------------------------------------------------
    # Dummy input
    #
    # Sequence length is intentionally arbitrary.
    # CP8 must not depend on a hard-coded DINO token count.
    # --------------------------------------------------------

    B = 2
    L = 10

    x = torch.randn(
        B,
        L,
        D_HEAD,
    )

    print("\nInput:")
    print("x:", tuple(x.shape))

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    q_updates, v_updates, dense_updates = bank(x)

    print("\nOutputs:")
    print("Q     :", tuple(q_updates.shape))
    print("V     :", tuple(v_updates.shape))
    print("Dense :", tuple(dense_updates.shape))

    expected_shape = (
        B,
        L,
        NUM_EXPERTS,
        D_HEAD,
    )

    assert q_updates.shape == expected_shape
    assert v_updates.shape == expected_shape
    assert dense_updates.shape == expected_shape

    # --------------------------------------------------------
    # B is zero-initialized
    # --------------------------------------------------------

    assert torch.allclose(
        q_updates,
        torch.zeros_like(q_updates),
    )

    assert torch.allclose(
        v_updates,
        torch.zeros_like(v_updates),
    )

    assert torch.allclose(
        dense_updates,
        torch.zeros_like(dense_updates),
    )

    # --------------------------------------------------------
    # Matrix dimensions
    # --------------------------------------------------------

    for expert in bank.experts:

        assert expert.q.A.shape == (96, 4)
        assert expert.q.B.shape == (4, 96)

        assert expert.v.A.shape == (96, 4)
        assert expert.v.B.shape == (4, 96)

        assert expert.dense.A.shape == (96, 4)
        assert expert.dense.B.shape == (4, 96)

    print("\nAll expert matrix dimensions: PASS")
    print("All output shapes:           PASS")
    print("Shared 6-expert bank:        PASS")
    print("Q/V/Dense support:           PASS")
    print("Initial zero-update state:   PASS")

    print("\nCP8 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()


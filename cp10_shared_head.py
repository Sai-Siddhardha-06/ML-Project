# cp10_shared_head.py
#
# CP10: Shared Head Routing for DINOv2-Small
#
# Phase-2 implementation:
#   - Reuses the six LoRA experts from CP8
#   - Uses one learned shared routing vector Z_shared
#   - All six experts participate through softmax
#
# DINOv2-Small:
#   d = 384
#
# DFF:
#   h = 4
#   d_h = 96
#   r = 16
#   r_h = 4
#   N = 6
#
# This checkpoint validates the shared routing branch.
# Full transformer-block integration is handled later.


import torch
import torch.nn as nn

from cp8_lora_expert_bank import (
    LoRAExpertBank,
    D_MODEL,
    NUM_HEADS,
    D_HEAD,
    LORA_RANK,
    HEAD_RANK,
    NUM_EXPERTS,
)


# ============================================================
# Configuration
# ============================================================

BETA = 1.0


# ============================================================
# Shared Head
# ============================================================

class SharedHead(nn.Module):
    """
    Shared routing branch.

    Reuses the six LoRA experts created in CP8.

    Z_shared:
        (6,)

    Routing:
        six-way softmax

    No Top-K is used.
    """

    def __init__(self):

        super().__init__()

        # IMPORTANT:
        # Reuse the CP8 expert bank.
        #
        # No second expert bank is created here.
        self.expert_bank = LoRAExpertBank()

        # One global shared routing vector.
        self.Z_shared = nn.Parameter(
            torch.zeros(NUM_EXPERTS)
        )

    def forward(self, x):

        """
        x:
            (B, L, 96)

        Returns:
            f_share:
                (B, L, 96)

            weights:
                (6,)
        """

        # ----------------------------------------------------
        # Six-way softmax
        # ----------------------------------------------------

        weights = torch.softmax(
            self.Z_shared,
            dim=-1,
        )

        # ----------------------------------------------------
        # CP8 expert bank
        # ----------------------------------------------------

        _, _, dense_updates = self.expert_bank(x)

        # ----------------------------------------------------
        # Weighted sum over six experts
        #
        # dense_updates:
        #   (B, L, 6, 96)
        #
        # weights:
        #   (6,)
        #
        # result:
        #   (B, L, 96)
        # ----------------------------------------------------

        f_share = torch.sum(
            dense_updates
            * weights.view(
                1,
                1,
                NUM_EXPERTS,
                1,
            ),
            dim=2,
        )

        f_share = BETA * f_share

        return f_share, weights


# ============================================================
# Validation
# ============================================================

def main():

    print("=" * 70)
    print("CP10 — DINOv2-Small Shared Head")
    print("=" * 70)

    print("\nConfiguration:")
    print(f"D_MODEL       : {D_MODEL}")
    print(f"NUM_HEADS     : {NUM_HEADS}")
    print(f"D_HEAD        : {D_HEAD}")
    print(f"LORA_RANK     : {LORA_RANK}")
    print(f"HEAD_RANK     : {HEAD_RANK}")
    print(f"NUM_EXPERTS   : {NUM_EXPERTS}")
    print(f"BETA          : {BETA}")

    # --------------------------------------------------------
    # Configuration checks
    # --------------------------------------------------------

    assert D_MODEL == 384
    assert NUM_HEADS == 4
    assert D_HEAD == 96
    assert LORA_RANK == 16
    assert HEAD_RANK == 4
    assert NUM_EXPERTS == 6

    # --------------------------------------------------------
    # Create shared head
    # --------------------------------------------------------

    model = SharedHead()

    print("\nModel:")
    print(model)

    # --------------------------------------------------------
    # Parameter count
    #
    # CP8 bank:
    #   13,824
    #
    # Z_shared:
    #   6
    #
    # Total:
    #   13,830
    # --------------------------------------------------------

    total_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    expected_params = 13824 + NUM_EXPERTS

    print(
        "\nTotal trainable parameters:",
        total_params,
    )

    print(
        "Expected parameters      :",
        expected_params,
    )

    assert total_params == expected_params

    # --------------------------------------------------------
    # Dummy input
    # --------------------------------------------------------

    BATCH = 2
    SEQ_LEN = 10

    x = torch.randn(
        BATCH,
        SEQ_LEN,
        D_HEAD,
    )

    print("\nInput:")
    print("x:", tuple(x.shape))

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    f_share, weights = model(x)

    print("\nShared routing weights:")
    print(weights)

    print("\nShared output:")
    print("f_share:", tuple(f_share.shape))

    # --------------------------------------------------------
    # Shape checks
    # --------------------------------------------------------

    assert f_share.shape == (
        BATCH,
        SEQ_LEN,
        D_HEAD,
    )

    assert weights.shape == (
        NUM_EXPERTS,
    )

    print("\nShared output shape: PASS")
    print("Shared routing shape: PASS")

    # --------------------------------------------------------
    # Softmax check
    # --------------------------------------------------------

    assert torch.allclose(
        weights.sum(),
        torch.tensor(1.0),
        atol=1e-6,
    )

    print("Six-way softmax sums to 1: PASS")

    # --------------------------------------------------------
    # Every expert receives positive probability
    # --------------------------------------------------------

    assert torch.all(weights > 0)

    print("All 6 experts receive weight: PASS")

    # --------------------------------------------------------
    # Initial zero-update state
    #
    # CP8 initializes B to zero.
    # Therefore the shared output is initially zero.
    # --------------------------------------------------------

    assert torch.allclose(
        f_share,
        torch.zeros_like(f_share),
        atol=1e-7,
    )

    print("Initial zero-update state: PASS")

    # --------------------------------------------------------
    # Non-zero update test
    #
    # Modify one CP8 expert's Dense B matrix.
    # The shared branch must then produce a non-zero output.
    # --------------------------------------------------------

    with torch.no_grad():

        original_B = (
            model.expert_bank
            .experts[0]
            .dense
            .B
            .clone()
        )

        model.expert_bank.experts[0].dense.B.fill_(0.1)

        f_share_modified, _ = model(x)

        model.expert_bank.experts[0].dense.B.copy_(
            original_B
        )

    assert not torch.allclose(
        f_share_modified,
        torch.zeros_like(f_share_modified),
    )

    print("Shared branch responds to expert update: PASS")

    # --------------------------------------------------------
    # Confirm CP8 bank is actually reused
    # --------------------------------------------------------

    assert len(model.expert_bank.experts) == NUM_EXPERTS

    print("CP8 six-expert bank reused: PASS")

    print("\nCP10 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

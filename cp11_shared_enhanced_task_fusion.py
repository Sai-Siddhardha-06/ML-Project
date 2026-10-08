import torch


# ============================================================
# CP11 — Shared-Enhanced Task Fusion
# DINOv2-Small dummy-tensor stage
# ============================================================

print("=" * 70)
print("CP11 — Shared-Enhanced Task Fusion")
print("=" * 70)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

D_MODEL = 384
NUM_HEADS = 4
D_HEAD = D_MODEL // NUM_HEADS

BATCH = 2


print("\nConfiguration:")
print(f"D_MODEL       : {D_MODEL}")
print(f"NUM_HEADS     : {NUM_HEADS}")
print(f"D_HEAD        : {D_HEAD}")


# ============================================================
# Shared-Enhanced Task Fusion
# ============================================================

class SharedEnhancedTaskFusion:
    """
    CP11 dummy-tensor fusion.

    For one task:

        f_share:
            (B, 96)

        task_head_outputs:
            (B, 3, 96)

    Concatenate:

        [f_share, task_head_1, task_head_2, task_head_3]

    giving:

        (B, 384)

    Then add residually to the 384-dimensional CLS token:

        f_fused = h_cls + fused_update

    giving:

        (B, 384)

    The same operation is performed independently for:

        1. authenticity
        2. forgery-type
    """

    def __call__(
        self,
        h_cls_auth,
        f_share_auth,
        task_heads_auth,
        h_cls_forgery,
        f_share_forgery,
        task_heads_forgery,
    ):
        # ----------------------------------------------------
        # Validate authenticity dimensions
        # ----------------------------------------------------

        assert h_cls_auth.shape == (BATCH, D_MODEL)
        assert f_share_auth.shape == (BATCH, D_HEAD)
        assert task_heads_auth.shape == (
            BATCH,
            NUM_HEADS - 1,
            D_HEAD,
        )

        # ----------------------------------------------------
        # Validate forgery-type dimensions
        # ----------------------------------------------------

        assert h_cls_forgery.shape == (BATCH, D_MODEL)
        assert f_share_forgery.shape == (BATCH, D_HEAD)
        assert task_heads_forgery.shape == (
            BATCH,
            NUM_HEADS - 1,
            D_HEAD,
        )

        # ----------------------------------------------------
        # Concatenate shared + 3 task-specific heads
        #
        # 96 + 3*96 = 384
        # ----------------------------------------------------

        fused_auth = torch.cat(
            [
                f_share_auth.unsqueeze(1),
                task_heads_auth,
            ],
            dim=1,
        )

        fused_forgery = torch.cat(
            [
                f_share_forgery.unsqueeze(1),
                task_heads_forgery,
            ],
            dim=1,
        )

        # ----------------------------------------------------
        # Flatten head dimension
        #
        # (B,4,96) -> (B,384)
        # ----------------------------------------------------

        fused_auth = fused_auth.reshape(
            BATCH,
            D_MODEL,
        )

        fused_forgery = fused_forgery.reshape(
            BATCH,
            D_MODEL,
        )

        # ----------------------------------------------------
        # Residual fusion with CLS
        # ----------------------------------------------------

        output_auth = h_cls_auth + fused_auth
        output_forgery = h_cls_forgery + fused_forgery

        return (
            output_auth,
            output_forgery,
            fused_auth,
            fused_forgery,
        )


# ============================================================
# Instantiate
# ============================================================

fusion = SharedEnhancedTaskFusion()


# ============================================================
# Dummy inputs
# ============================================================

# Authenticity branch

h_cls_auth = torch.randn(
    BATCH,
    D_MODEL,
)

f_share_auth = torch.randn(
    BATCH,
    D_HEAD,
)

task_heads_auth = torch.randn(
    BATCH,
    NUM_HEADS - 1,
    D_HEAD,
)


# Forgery-type branch

h_cls_forgery = torch.randn(
    BATCH,
    D_MODEL,
)

f_share_forgery = torch.randn(
    BATCH,
    D_HEAD,
)

task_heads_forgery = torch.randn(
    BATCH,
    NUM_HEADS - 1,
    D_HEAD,
)


print("\nInput shapes:")

print(f"h_cls_auth        : {tuple(h_cls_auth.shape)}")
print(f"f_share_auth      : {tuple(f_share_auth.shape)}")
print(f"task_heads_auth   : {tuple(task_heads_auth.shape)}")

print(f"h_cls_forgery     : {tuple(h_cls_forgery.shape)}")
print(f"f_share_forgery   : {tuple(f_share_forgery.shape)}")
print(f"task_heads_forgery: {tuple(task_heads_forgery.shape)}")


# ============================================================
# Forward pass
# ============================================================

(
    output_auth,
    output_forgery,
    fused_auth,
    fused_forgery,
) = fusion(
    h_cls_auth,
    f_share_auth,
    task_heads_auth,
    h_cls_forgery,
    f_share_forgery,
    task_heads_forgery,
)


print("\nIntermediate shapes:")
print(f"fused_auth        : {tuple(fused_auth.shape)}")
print(f"fused_forgery     : {tuple(fused_forgery.shape)}")

print("\nFinal fused representations:")
print(f"output_auth       : {tuple(output_auth.shape)}")
print(f"output_forgery    : {tuple(output_forgery.shape)}")


# ============================================================
# Validation 1 — shared + task concatenation dimension
# ============================================================

assert (
    D_HEAD + (NUM_HEADS - 1) * D_HEAD
    == D_MODEL
)

print("\n96 + 3 × 96 = 384: PASS")


# ============================================================
# Validation 2 — concatenated authenticity representation
# ============================================================

assert fused_auth.shape == (
    BATCH,
    D_MODEL,
)

print("Authenticity concatenation shape: PASS")


# ============================================================
# Validation 3 — concatenated forgery representation
# ============================================================

assert fused_forgery.shape == (
    BATCH,
    D_MODEL,
)

print("Forgery-type concatenation shape: PASS")


# ============================================================
# Validation 4 — authenticity residual output
# ============================================================

assert output_auth.shape == (
    BATCH,
    D_MODEL,
)

print("Authenticity residual output shape: PASS")


# ============================================================
# Validation 5 — forgery residual output
# ============================================================

assert output_forgery.shape == (
    BATCH,
    D_MODEL,
)

print("Forgery-type residual output shape: PASS")


# ============================================================
# Validation 6 — verify authenticity residual equation
# ============================================================

expected_auth = (
    h_cls_auth
    + fused_auth
)

assert torch.allclose(
    output_auth,
    expected_auth,
    atol=1e-6,
)

print("Authenticity residual equation: PASS")


# ============================================================
# Validation 7 — verify forgery residual equation
# ============================================================

expected_forgery = (
    h_cls_forgery
    + fused_forgery
)

assert torch.allclose(
    output_forgery,
    expected_forgery,
    atol=1e-6,
)

print("Forgery-type residual equation: PASS")


# ============================================================
# Validation 8 — branches remain independent
# ============================================================

assert not torch.equal(
    output_auth,
    output_forgery,
)

print("Authenticity/forgery branches independent: PASS")


print("\nCP11 COMPLETE")
print("=" * 70)

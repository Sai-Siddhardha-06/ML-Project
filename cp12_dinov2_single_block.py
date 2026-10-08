import math
import torch
import torch.nn as nn
from transformers import AutoModel


# ============================================================
# CP12 — DINOv2-Small + DFF Adapter in One Real Transformer Block
# ============================================================

print("=" * 70)
print("CP12 — DINOv2-Small Single-Block DFF Integration")
print("=" * 70)


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "facebook/dinov2-small"

D_MODEL = 384

# DFF dimensions from CP8–CP11.
NUM_DFF_HEADS = 4
D_HEAD = 96

LORA_RANK = 16
HEAD_RANK = 4

NUM_EXPERTS = 6
NUM_TASKS = 2
TOP_K = 3

ALPHA = 32
SCALING = ALPHA / LORA_RANK   # 2.0

BATCH_SIZE = 2
IMAGE_SIZE = 224

# CP12 integrates one task branch at a time.
# 0 = authenticity
# 1 = forgery type
TASK_ID = 0


# ============================================================
# CP8 — LoRA expert bank
# ============================================================

class LoRAProjection(nn.Module):
    """
    One 96-D low-rank LoRA projection.

    A: (96, 4)
    B: (4, 96)

    x @ A @ B -> (B,L,96)
    """

    def __init__(self):
        super().__init__()

        self.A = nn.Parameter(
            torch.empty(D_HEAD, HEAD_RANK)
        )

        # Zero initialization means the initial DFF update is zero.
        self.B = nn.Parameter(
            torch.zeros(HEAD_RANK, D_HEAD)
        )

        # A is used as x @ A.
        # A has shape (96,4), so initialize A.T so Kaiming
        # sees fan-in = 96.
        nn.init.kaiming_uniform_(
            self.A.T,
            a=math.sqrt(5)
        )

    def forward(self, x):
        """
        x:
            (B,L,96)

        returns:
            (B,L,96)
        """

        return SCALING * (
            x @ self.A @ self.B
        )


class LoRAExpert(nn.Module):
    """
    One expert containing Q, V and Dense low-rank projections.
    """

    def __init__(self):
        super().__init__()

        self.q = LoRAProjection()
        self.v = LoRAProjection()
        self.dense = LoRAProjection()


class LoRAExpertBank(nn.Module):
    """
    Shared pool of six experts.

    For a single 96-D DFF head:

        input:
            (B,L,96)

        outputs:
            Q:
                (B,L,6,96)

            V:
                (B,L,6,96)

            Dense:
                (B,L,6,96)
    """

    def __init__(self):
        super().__init__()

        self.experts = nn.ModuleList(
            [
                LoRAExpert()
                for _ in range(NUM_EXPERTS)
            ]
        )

    def forward(self, x):

        q_outputs = []
        v_outputs = []
        dense_outputs = []

        for expert in self.experts:

            q_outputs.append(
                expert.q(x)
            )

            v_outputs.append(
                expert.v(x)
            )

            dense_outputs.append(
                expert.dense(x)
            )

        q_updates = torch.stack(
            q_outputs,
            dim=2
        )

        v_updates = torch.stack(
            v_outputs,
            dim=2
        )

        dense_updates = torch.stack(
            dense_outputs,
            dim=2
        )

        return (
            q_updates,
            v_updates,
            dense_updates,
        )


# ============================================================
# CP9 — Task-specific router
# ============================================================

class ForgeryAwareMultiHeadRouter(nn.Module):
    """
    Input-independent task router from CP9.

    Z_task:
        (NUM_TASKS, NUM_DFF_HEADS, NUM_EXPERTS)

    For each task/head:

        softmax
            ->
        Top-K
            ->
        renormalization
    """

    def __init__(self):
        super().__init__()

        self.Z_task = nn.Parameter(
            0.01 * torch.randn(
                NUM_TASKS,
                NUM_DFF_HEADS,
                NUM_EXPERTS
            )
        )

    def forward(self, task_id):

        logits = self.Z_task[task_id]

        # (4,6)
        probs = torch.softmax(
            logits,
            dim=-1
        )

        # (4,3), (4,3)
        top_values, top_indices = torch.topk(
            probs,
            k=TOP_K,
            dim=-1
        )

        # Renormalize selected Top-K weights.
        top_weights = (
            top_values
            / top_values.sum(
                dim=-1,
                keepdim=True
            )
        )

        return (
            top_indices,
            top_weights,
        )


# ============================================================
# CP12 — DFF adapter for one real DINOv2 projection
# ============================================================

class DFFProjectionAdapter(nn.Module):
    """
    DFF adapter attached to one real DINOv2 projection.

    The SAME expert bank and SAME task router are shared across
    Q, V and Dense.

    projection_type:

        "q"
            use expert.q

        "v"
            use expert.v

        "dense"
            use expert.dense

    For each DFF head:

        task branch:
            Top-3 routing

        shared branch:
            all 6 experts

    CP11 Phase-2 fusion:

        1 shared 96-D representation
        +
        3 task-specific 96-D representations
        =
        384-D residual update
    """

    def __init__(
        self,
        projection_type,
        expert_bank,
        router,
        Z_shared,
    ):
        super().__init__()

        if projection_type not in {
            "q",
            "v",
            "dense",
        }:
            raise ValueError(
                f"Invalid projection_type={projection_type}. "
                f"Expected 'q', 'v', or 'dense'."
            )

        self.projection_type = projection_type

        # One shared CP8 expert bank.
        self.expert_bank = expert_bank

        # One shared CP9 task router.
        self.router = router

        # One shared CP10 routing vector.
        #
        # The SAME Parameter is passed to Q, V and Dense.
        self.Z_shared = Z_shared

    def _select_projection_updates(
        self,
        q_updates,
        v_updates,
        dense_updates,
    ):
        """
        Select the correct expert branch for the real
        DINOv2 projection being wrapped.
        """

        if self.projection_type == "q":
            return q_updates

        if self.projection_type == "v":
            return v_updates

        return dense_updates

    def forward(
        self,
        hidden_states,
        task_id=0,
    ):
        """
        hidden_states:

            (B,L,384)

        returns:

            DFF residual update:
                (B,L,384)
        """

        B, L, D = hidden_states.shape

        assert D == D_MODEL, (
            f"Expected hidden dimension {D_MODEL}, "
            f"got {D}"
        )

        # --------------------------------------------------------
        # Split 384-D hidden state into four 96-D DFF heads.
        # --------------------------------------------------------

        head_inputs = hidden_states.reshape(
            B,
            L,
            NUM_DFF_HEADS,
            D_HEAD,
        )

        # --------------------------------------------------------
        # CP9 task routing.
        # --------------------------------------------------------

        top_indices, top_weights = self.router(
            task_id
        )

        # top_indices:
        #     (4,3)
        #
        # top_weights:
        #     (4,3)

        # --------------------------------------------------------
        # CP10 shared routing.
        # --------------------------------------------------------

        shared_weights = torch.softmax(
            self.Z_shared,
            dim=-1,
        )

        # (6,)

        task_outputs = []
        shared_outputs = []

        # ========================================================
        # Process each DFF head.
        # ========================================================

        for head_idx in range(NUM_DFF_HEADS):

            x_head = head_inputs[
                :,
                :,
                head_idx,
                :
            ]

            # (B,L,96)

            # ----------------------------------------------------
            # Evaluate shared CP8 expert bank.
            # ----------------------------------------------------

            q_updates, v_updates, dense_updates = (
                self.expert_bank(x_head)
            )

            # Each:
            # (B,L,6,96)

            # ----------------------------------------------------
            # Select branch corresponding to real projection.
            # ----------------------------------------------------

            projection_updates = (
                self._select_projection_updates(
                    q_updates,
                    v_updates,
                    dense_updates,
                )
            )

            # (B,L,6,96)

            # ----------------------------------------------------
            # Task-specific Top-3 routing.
            # ----------------------------------------------------

            selected_updates = (
                projection_updates[
                    :,
                    :,
                    top_indices[head_idx],
                    :
                ]
            )

            # (B,L,3,96)

            selected_weights = (
                top_weights[head_idx]
            )

            # (3,)

            task_update = (
                selected_updates
                * selected_weights.view(
                    1,
                    1,
                    TOP_K,
                    1,
                )
            ).sum(dim=2)

            # (B,L,96)

            # ----------------------------------------------------
            # Shared six-expert routing.
            # ----------------------------------------------------

            shared_update = (
                projection_updates
                * shared_weights.view(
                    1,
                    1,
                    NUM_EXPERTS,
                    1,
                )
            ).sum(dim=2)

            # (B,L,96)

            task_outputs.append(
                task_update
            )

            shared_outputs.append(
                shared_update
            )

        # ========================================================
        # Stack four DFF heads.
        # ========================================================

        task_outputs = torch.stack(
            task_outputs,
            dim=2,
        )

        # (B,L,4,96)

        shared_outputs = torch.stack(
            shared_outputs,
            dim=2,
        )

        # (B,L,4,96)

        # ========================================================
        # CP11 Phase-2 fusion.
        #
        # One shared representation:
        #
        #     shared_outputs[:,:,0,:]
        #
        # plus three task-specific representations:
        #
        #     task_outputs[:,:,:3,:]
        #
        # gives:
        #
        #     96 + 3*96 = 384
        # ========================================================

        shared_slot = shared_outputs[
            :,
            :,
            0,
            :
        ]

        # (B,L,96)

        task_slots = task_outputs[
            :,
            :,
            :3,
            :
        ]

        # (B,L,3,96)

        # --------------------------------------------------------
        # Concatenate.
        # --------------------------------------------------------

        fused = torch.cat(
            [
                shared_slot.unsqueeze(2),
                task_slots,
            ],
            dim=2,
        )

        # (B,L,4,96)

        # --------------------------------------------------------
        # Flatten four 96-D slots.
        # --------------------------------------------------------

        fused = fused.reshape(
            B,
            L,
            D_MODEL,
        )

        # (B,L,384)

        return fused


# ============================================================
# Wrapped DINOv2 Linear projection
# ============================================================

class DFFWrappedLinear(nn.Module):
    """
    Wrap one frozen DINOv2 Linear projection.

    Forward:

        original_output = original_linear(x)

        dff_update = adapter(x)

        output = original_output + dff_update

    The original DINOv2 Linear remains frozen.
    """

    def __init__(
        self,
        original_linear,
        adapter,
        task_id,
        projection_type,
    ):
        super().__init__()

        self.original_linear = original_linear
        self.adapter = adapter
        self.task_id = task_id
        self.projection_type = projection_type

    def forward(self, x):

        # Original frozen DINOv2 projection.
        base_output = self.original_linear(x)

        # DFF residual update.
        dff_update = self.adapter(
            x,
            task_id=self.task_id,
        )

        # Residual injection.
        return base_output + dff_update


# ============================================================
# Load DINOv2-Small
# ============================================================

print("\nLoading DINOv2-Small...")

model = AutoModel.from_pretrained(
    MODEL_NAME
)

print("Model loaded.")


# ============================================================
# Freeze entire DINOv2 backbone
# ============================================================

for parameter in model.parameters():
    parameter.requires_grad = False


# ============================================================
# Select exactly one real transformer block
# ============================================================

block = model.encoder.layer[0]

print("\nSelected block:")
print(type(block).__name__)

print("\nOriginal target projections:")

print(
    "Q     :",
    type(
        block.attention.attention.query
    ).__name__
)

print(
    "V     :",
    type(
        block.attention.attention.value
    ).__name__
)

print(
    "Dense :",
    type(
        block.attention.output.dense
    ).__name__
)


# ============================================================
# Save original frozen projections
# ============================================================

original_query = (
    block.attention.attention.query
)

original_value = (
    block.attention.attention.value
)

original_dense = (
    block.attention.output.dense
)


# ============================================================
# Verify originals are frozen before wrapping
# ============================================================

assert all(
    not parameter.requires_grad
    for parameter in original_query.parameters()
)

assert all(
    not parameter.requires_grad
    for parameter in original_value.parameters()
)

assert all(
    not parameter.requires_grad
    for parameter in original_dense.parameters()
)

print(
    "Original Q/V/Dense frozen status: PASS"
)


# ============================================================
# One shared CP8 expert bank
# One shared CP9 task router
# One shared CP10 routing vector
# ============================================================

expert_bank = LoRAExpertBank()

router = ForgeryAwareMultiHeadRouter()

Z_shared = nn.Parameter(
    torch.zeros(NUM_EXPERTS)
)


# ============================================================
# Projection-specific adapters
#
# All three share:
#     expert_bank
#     router
#     Z_shared
#
# Only the selected projection branch differs.
# ============================================================

q_adapter = DFFProjectionAdapter(
    projection_type="q",
    expert_bank=expert_bank,
    router=router,
    Z_shared=Z_shared,
)

v_adapter = DFFProjectionAdapter(
    projection_type="v",
    expert_bank=expert_bank,
    router=router,
    Z_shared=Z_shared,
)

dense_adapter = DFFProjectionAdapter(
    projection_type="dense",
    expert_bank=expert_bank,
    router=router,
    Z_shared=Z_shared,
)


# ============================================================
# Replace Q, V and attention-output Dense
# ============================================================

block.attention.attention.query = (
    DFFWrappedLinear(
        original_query,
        q_adapter,
        TASK_ID,
        "q",
    )
)

block.attention.attention.value = (
    DFFWrappedLinear(
        original_value,
        v_adapter,
        TASK_ID,
        "v",
    )
)

block.attention.output.dense = (
    DFFWrappedLinear(
        original_dense,
        dense_adapter,
        TASK_ID,
        "dense",
    )
)


# ============================================================
# Verify wrappers were installed
# ============================================================

assert isinstance(
    block.attention.attention.query,
    DFFWrappedLinear,
)

assert isinstance(
    block.attention.attention.value,
    DFFWrappedLinear,
)

assert isinstance(
    block.attention.output.dense,
    DFFWrappedLinear,
)

print(
    "Q/V/Dense wrapper installation: PASS"
)


# ============================================================
# Verify trainable parameters
# ============================================================

trainable_model_params = [
    (name, parameter)
    for name, parameter in model.named_parameters()
    if parameter.requires_grad
]

frozen_model_params = [
    (name, parameter)
    for name, parameter in model.named_parameters()
    if not parameter.requires_grad
]


print("\nTrainable model parameters:")

for name, parameter in trainable_model_params:

    print(
        f"  {name:<80} "
        f"{parameter.numel()}"
    )


total_trainable = sum(
    parameter.numel()
    for _, parameter in trainable_model_params
)

total_frozen = sum(
    parameter.numel()
    for _, parameter in frozen_model_params
)


print(
    "\nTotal trainable model parameters:",
    total_trainable
)

print(
    "Total frozen model parameters:",
    total_frozen
)


# ============================================================
# Expected parameter count
# ============================================================

EXPECTED_LORA_PARAMS = (
    NUM_EXPERTS
    * 3
    * (
        D_HEAD * HEAD_RANK
        + HEAD_RANK * D_HEAD
    )
)

EXPECTED_ROUTER_PARAMS = (
    NUM_TASKS
    * NUM_DFF_HEADS
    * NUM_EXPERTS
)

EXPECTED_SHARED_ROUTER_PARAMS = (
    NUM_EXPERTS
)

EXPECTED_TOTAL_ADAPTER_PARAMS = (
    EXPECTED_LORA_PARAMS
    + EXPECTED_ROUTER_PARAMS
    + EXPECTED_SHARED_ROUTER_PARAMS
)


print("\nExpected CP12 trainable parameter count:")

print(
    "LoRA expert bank      :",
    EXPECTED_LORA_PARAMS
)

print(
    "Task router           :",
    EXPECTED_ROUTER_PARAMS
)

print(
    "Shared router         :",
    EXPECTED_SHARED_ROUTER_PARAMS
)

print(
    "Expected total        :",
    EXPECTED_TOTAL_ADAPTER_PARAMS
)

assert EXPECTED_LORA_PARAMS == 13824

assert EXPECTED_ROUTER_PARAMS == 48

assert EXPECTED_SHARED_ROUTER_PARAMS == 6

assert EXPECTED_TOTAL_ADAPTER_PARAMS == 13878

assert total_trainable == EXPECTED_TOTAL_ADAPTER_PARAMS, (
    f"Trainable parameter mismatch: "
    f"expected {EXPECTED_TOTAL_ADAPTER_PARAMS}, "
    f"got {total_trainable}"
)

print(
    "Trainable parameter count: PASS"
)


# ============================================================
# Verify original DINOv2 parameters remain frozen
# ============================================================

assert all(
    not parameter.requires_grad
    for _, parameter in frozen_model_params
)

print(
    "Backbone frozen status: PASS"
)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "\nDevice:",
    device
)


# ============================================================
# Move model to device
# ============================================================

model = model.to(device)


# ============================================================
# Verify shared objects after model registration
# ============================================================

assert (
    q_adapter.expert_bank
    is v_adapter.expert_bank
)

assert (
    q_adapter.expert_bank
    is dense_adapter.expert_bank
)

assert (
    q_adapter.router
    is v_adapter.router
)

assert (
    q_adapter.router
    is dense_adapter.router
)

assert (
    q_adapter.Z_shared
    is v_adapter.Z_shared
)

assert (
    q_adapter.Z_shared
    is dense_adapter.Z_shared
)

print(
    "Shared expert bank: PASS"
)

print(
    "Shared task router: PASS"
)

print(
    "Shared Z_shared: PASS"
)


# ============================================================
# Dummy input
# ============================================================

pixel_values = torch.randn(
    BATCH_SIZE,
    3,
    IMAGE_SIZE,
    IMAGE_SIZE,
    device=device,
)


print("\nInput:")

print(
    "pixel_values:",
    tuple(pixel_values.shape)
)


# ============================================================
# Forward pass
# ============================================================

model.train()

outputs = model(
    pixel_values=pixel_values
)

last_hidden_state = (
    outputs.last_hidden_state
)


print("\nForward output:")

print(
    "last_hidden_state:",
    tuple(
        last_hidden_state.shape
    )
)


# ============================================================
# Forward shape checks
# ============================================================

assert last_hidden_state.ndim == 3

assert (
    last_hidden_state.shape[0]
    == BATCH_SIZE
)

assert (
    last_hidden_state.shape[-1]
    == D_MODEL
)

# Ordinary DINOv2-Small at 224x224:
# 1 CLS token + 256 patch tokens = 257.
assert (
    last_hidden_state.shape[1]
    == 257
)

print(
    "DINOv2 forward shape: PASS"
)


# ============================================================
# Backward pass
# ============================================================

# A scalar loss that depends on the entire final representation.
loss = last_hidden_state.mean()


print("\nLoss:")

print(
    loss.item()
)


# Clear old gradients.
model.zero_grad(
    set_to_none=True
)


# Backpropagation.
loss.backward()


print(
    "\nBackward completed."
)


# ============================================================
# Gradient audit
# ============================================================

adapter_gradients = []

frozen_gradients = []


for name, parameter in model.named_parameters():

    if parameter.requires_grad:

        adapter_gradients.append(
            (
                name,
                parameter.grad is not None,
                None
                if parameter.grad is None
                else float(
                    parameter.grad.abs().sum()
                ),
            )
        )

    else:

        frozen_gradients.append(
            (
                name,
                parameter.grad is not None,
            )
        )


# ============================================================
# Trainable adapter gradient audit
# ============================================================

print(
    "\nTrainable parameter gradient audit:"
)


missing_gradients = []


for (
    name,
    has_grad,
    grad_abs_sum,
) in adapter_gradients:

    print(
        f"  {name:<80}"
        f" grad={has_grad}"
        f" grad_abs_sum={grad_abs_sum}"
    )

    if not has_grad:

        missing_gradients.append(
            name
        )


assert len(
    adapter_gradients
) > 0


assert not missing_gradients, (
    "Trainable adapter parameters "
    "without gradients:\n"
    + "\n".join(
        missing_gradients
    )
)


print(
    "Adapter gradients: PASS"
)


# ============================================================
# Frozen DINOv2 gradient audit
# ============================================================

frozen_with_gradients = [
    name
    for (
        name,
        has_grad,
    ) in frozen_gradients
    if has_grad
]


print(
    "\nFrozen backbone gradient audit:"
)

print(
    "Frozen parameters with gradients:",
    len(
        frozen_with_gradients
    )
)


if frozen_with_gradients:

    print(
        "\nUnexpected frozen gradients:"
    )

    for name in frozen_with_gradients:

        print(
            " ",
            name
        )


assert not frozen_with_gradients, (
    "Frozen DINOv2 parameters "
    "received gradients:\n"
    + "\n".join(
        frozen_with_gradients
    )
)


print(
    "Frozen DINOv2 gradients: PASS"
)


# ============================================================
# Verify projection types
# ============================================================

assert (
    block.attention.attention.query
    .adapter
    .projection_type
    == "q"
)

assert (
    block.attention.attention.value
    .adapter
    .projection_type
    == "v"
)

assert (
    block.attention.output.dense
    .adapter
    .projection_type
    == "dense"
)


print(
    "\nQ integration: PASS"
)

print(
    "V integration: PASS"
)

print(
    "Dense integration: PASS"
)


# ============================================================
# Final CP12 result
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "CP12 COMPLETE"
)

print(
    "=" * 70
)

print(
    "Real DINOv2-Small block       : PASS"
)

print(
    "Q/V/Dense adapter integration : PASS"
)

print(
    "Forward shape                 : PASS"
)

print(
    "Backward pass                 : PASS"
)

print(
    "Adapter gradients             : PASS"
)

print(
    "Frozen DINOv2 gradients       : PASS"
)

print(
    "Shared expert bank            : PASS"
)

print(
    "Shared task router            : PASS"
)

print(
    "Shared Z_shared               : PASS"
)

print(
    "=" * 70
)

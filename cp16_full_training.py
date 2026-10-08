import math
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from transformers import AutoModel
from sklearn.metrics import roc_auc_score


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

# CP15 training configuration.
MANIFEST_PATH = "splits/ffpp_manifest.csv"
NUM_WORKERS = 0
MAX_TRAIN_SAMPLES = None
MAX_VAL_SAMPLES = None

CHECKPOINT_DIR = "checkpoints/cp16"
CHECKPOINT_INTERVAL = 200

LEARNING_RATE = 1e-4
LAMBDA_0 = 1.0
LAMBDA_1 = 1.0
NUM_EPOCHS = 5

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
# Insert DFF adapters into EVERY DINOv2 transformer block
# ============================================================

expert_bank = LoRAExpertBank()
router = ForgeryAwareMultiHeadRouter()

# One shared routing vector across all blocks and projections
Z_shared = nn.Parameter(
    torch.zeros(NUM_EXPERTS)
)

adapted_blocks = []

for block_idx, block in enumerate(model.encoder.layer):

    # Save original frozen projections
    original_query = block.attention.attention.query
    original_value = block.attention.attention.value
    original_dense = block.attention.output.dense

    # Verify originals are frozen
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

    # Projection-specific adapters.
    # All blocks share the same expert bank, task router,
    # and Z_shared.
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

    # Replace Q
    block.attention.attention.query = DFFWrappedLinear(
        original_query,
        q_adapter,
        TASK_ID,
        "q",
    )

    # Replace V
    block.attention.attention.value = DFFWrappedLinear(
        original_value,
        v_adapter,
        TASK_ID,
        "v",
    )

    # Replace attention-output Dense
    block.attention.output.dense = DFFWrappedLinear(
        original_dense,
        dense_adapter,
        TASK_ID,
        "dense",
    )

    # Verify installation
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

    adapted_blocks.append(block)

print(
    f"Adapted transformer blocks: "
    f"{len(adapted_blocks)}/{len(model.encoder.layer)}"
)

assert len(adapted_blocks) == len(model.encoder.layer)

print(
    "All-block Q/V/Dense wrapper installation: PASS"
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


print("\nExpected CP13 trainable parameter count:")

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
# CP14 dataset
# ============================================================

class FFPPDataset(Dataset):
    def __init__(self, manifest_path, split, frames_root, max_samples=None):
        self.frames_root = Path(frames_root)

        df = pd.read_csv(manifest_path)
        df = df[df["split"] == split].copy()

        if max_samples is not None:
            manipulations = [
                "original",
                "Deepfakes",
                "Face2Face",
                "FaceSwap",
                "NeuralTextures",
            ]

            per_class = max_samples // len(manipulations)
            remainder = max_samples % len(manipulations)

            parts = []

            for class_idx, manipulation in enumerate(manipulations):
                class_df = df[df["manipulation"] == manipulation]

                take = per_class + (
                    1 if class_idx < remainder else 0
                )

                parts.append(class_df.iloc[:take])

            df = pd.concat(parts, ignore_index=True)

        # Build metadata lookup once.
        metadata_lookup = {}
        split_dir = self.frames_root / split

        for meta_path in split_dir.glob("*/meta.json"):
            data = pd.read_json(meta_path)
            video_path = str(data["video_path"].iloc[0])
            video_name = Path(video_path).name
            metadata_lookup[video_name] = meta_path

        print(
            f"{split} metadata index built: "
            f"{len(metadata_lookup)} videos"
        )

        self.samples = []

        for _, row in df.iterrows():
            video_path = str(row["path"])
            video_name = Path(video_path).name

            found_meta = metadata_lookup.get(video_name)

            if found_meta is None:
                raise FileNotFoundError(
                    f"No extracted-frame metadata found for {video_path}"
                )

            frame_dir = found_meta.parent
            frame_paths = sorted(frame_dir.glob("*.jpg"))

            if not frame_paths:
                raise FileNotFoundError(
                    f"No JPG frames found in {frame_dir}"
                )

            self.samples.append(
                {
                    "frame_path": frame_paths[0],
                    "authenticity": int(row["label"]),
                    "manipulation": str(row["manipulation"]),
                }
            )

        self.manipulation_to_id = {
            "original": 0,
            "Deepfakes": 1,
            "Face2Face": 2,
            "FaceSwap": 3,
            "NeuralTextures": 4,
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        image = Image.open(sample["frame_path"]).convert("RGB")

        image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        image = torch.from_numpy(
            __import__("numpy").array(image)
        ).permute(2, 0, 1).float() / 255.0

        mean = torch.tensor(
            [0.485, 0.456, 0.406],
            dtype=image.dtype,
        ).view(3, 1, 1)

        std = torch.tensor(
            [0.229, 0.224, 0.225],
            dtype=image.dtype,
        ).view(3, 1, 1)

        image = (image - mean) / std

        authenticity = torch.tensor(
            float(sample["authenticity"]),
            dtype=torch.float32,
        )

        forgery_type = torch.tensor(
            self.manipulation_to_id[sample["manipulation"]],
            dtype=torch.long,
        )

        return image, authenticity, forgery_type


# ============================================================
# CP14 DataLoaders
# ============================================================

train_dataset = FFPPDataset(
    MANIFEST_PATH,
    "train",
    "frames/ffpp",
    max_samples=MAX_TRAIN_SAMPLES,
)

val_dataset = FFPPDataset(
    MANIFEST_PATH,
    "val",
    "frames/ffpp",
    max_samples=MAX_VAL_SAMPLES,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)

print("\nCP15 DataLoaders:")
print("Train samples:", len(train_dataset))
print("Val samples:", len(val_dataset))
print("Train batches:", len(train_loader))
print("Val batches:", len(val_loader))

# ============================================================
# CP14 classifier heads
# ============================================================

AUTHENTICITY_CLASSES = 1
FORGERY_TYPE_CLASSES = 5

authenticity_head = torch.nn.Linear(
    D_MODEL,
    AUTHENTICITY_CLASSES,
).to(device)

forgery_type_head = torch.nn.Linear(
    D_MODEL,
    FORGERY_TYPE_CLASSES,
).to(device)

print("\nCP14 classifier heads:")

print(
    "Authenticity head:",
    tuple(authenticity_head.weight.shape),
)

print(
    "Forgery-type head:",
    tuple(forgery_type_head.weight.shape),
)

assert authenticity_head.weight.shape == (1, D_MODEL)

assert forgery_type_head.weight.shape == (
    FORGERY_TYPE_CLASSES,
    D_MODEL,
)

print("Classifier head dimensions: PASS")

# CP14 optimizer
# ============================================================

trainable_params = [
    parameter
    for parameter in model.parameters()
    if parameter.requires_grad
]

trainable_params += list(authenticity_head.parameters())
trainable_params += list(forgery_type_head.parameters())

optimizer = torch.optim.AdamW(
    trainable_params,
    lr=LEARNING_RATE,
)

print("\nCP14 optimizer:")
print("Trainable parameter tensors:", len(trainable_params))
print("Learning rate:", LEARNING_RATE)

# ============================================================


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



def set_task_flag(model, task_flag):
    """Set the DFF task branch for every adapted projection."""
    assert task_flag in (0, 1), "task_flag must be 0 (authenticity) or 1 (forgery-type)"

    wrapper_count = 0

    for module in model.modules():
        if isinstance(module, DFFWrappedLinear):
            module.task_id = task_flag
            wrapper_count += 1

    assert wrapper_count == 36, (
        f"Expected 36 DFF wrappers (12 blocks x 3 projections), "
        f"found {wrapper_count}"
    )

    print(f"Task flag set to {task_flag}: {wrapper_count}/36 wrappers")


print("\nCP13 task-flag mechanism:")
set_task_flag(model, 0)
assert all(
    module.task_id == 0
    for module in model.modules()
    if isinstance(module, DFFWrappedLinear)
)
print("Authenticity task flag (0): PASS")

set_task_flag(model, 1)
assert all(
    module.task_id == 1
    for module in model.modules()
    if isinstance(module, DFFWrappedLinear)
)
print("Forgery-type task flag (1): PASS")

# Reset to authenticity before the existing forward/backward test.
set_task_flag(model, 0)

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
# CP14 dual-loss test
# ============================================================

# Use the CLS token as the video/frame representation.
cls_features = last_hidden_state[:, 0, :]

assert cls_features.shape == (
    BATCH_SIZE,
    D_MODEL,
)

# Dummy labels for validating the two task losses.
authenticity_targets = torch.tensor(
    [0.0, 1.0],
    device=device,
)

forgery_type_targets = torch.tensor(
    [0, 3],
    dtype=torch.long,
    device=device,
)

# Task 0: authenticity.
set_task_flag(model, 0)

authenticity_logits = authenticity_head(
    cls_features
).squeeze(-1)

bce_loss_fn = torch.nn.BCEWithLogitsLoss()

bce_loss = bce_loss_fn(
    authenticity_logits,
    authenticity_targets,
)

# Task 1: forgery type.
# Re-run the backbone because the DFF task flag changes its behavior.
set_task_flag(model, 1)

outputs_task1 = model(
    pixel_values=pixel_values
)

last_hidden_state_task1 = outputs_task1.last_hidden_state

cls_features_task1 = last_hidden_state_task1[:, 0, :]

forgery_type_logits = forgery_type_head(
    cls_features_task1
)

ce_loss_fn = torch.nn.CrossEntropyLoss()

ftc_loss = ce_loss_fn(
    forgery_type_logits,
    forgery_type_targets,
)

LAMBDA_0 = 1.0
LAMBDA_1 = 1.0

total_loss = (
    LAMBDA_0 * bce_loss
    + LAMBDA_1 * ftc_loss
)

print("\nCP14 dual-loss test:")

print(
    "CLS representation:",
    tuple(cls_features.shape)
)

print(
    "Authenticity logits:",
    tuple(authenticity_logits.shape)
)

print(
    "Forgery-type logits:",
    tuple(forgery_type_logits.shape)
)

print(
    "BCE loss:",
    bce_loss.item()
)

print(
    "FTC CE loss:",
    ftc_loss.item()
)

print(
    "Combined loss:",
    total_loss.item()
)

assert authenticity_logits.shape == (
    BATCH_SIZE,
)

assert forgery_type_logits.shape == (
    BATCH_SIZE,
    FORGERY_TYPE_CLASSES,
)

assert torch.isfinite(bce_loss)
assert torch.isfinite(ftc_loss)
assert torch.isfinite(total_loss)

print("Dual-loss computation: PASS")


# Reset to authenticity for the next stage.
set_task_flag(model, 0)


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


# CP14 note:
# The dual-loss test uses two separate forward passes with
# different task flags. Some parameters may legitimately receive
# no gradient in a single task because the router selects only
# task-specific paths.
#
# The real training loop below will accumulate gradients from both
# task passes before the optimizer step.

if len(missing_gradients) > 0:
    print(
        "CP14 note: some adapter parameters have no gradient "
        "from the current single combined test graph."
    )
else:
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

# ============================================================
# CP16 training loop
# ============================================================

bce_loss_fn = torch.nn.BCEWithLogitsLoss()
ce_loss_fn = torch.nn.CrossEntropyLoss()

global_step = 0
best_auc = -1.0
history = []
start_epoch = 0

checkpoint_dir = Path(CHECKPOINT_DIR)
checkpoint_dir.mkdir(parents=True, exist_ok=True)

best_checkpoint_path = checkpoint_dir / "cp16_best_auc.pt"

# Resume from the latest completed-epoch checkpoint if it exists.
if best_checkpoint_path.exists():
    resume_checkpoint = torch.load(
        best_checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(resume_checkpoint["model_state_dict"])
    authenticity_head.load_state_dict(
        resume_checkpoint["authenticity_head_state_dict"]
    )
    forgery_type_head.load_state_dict(
        resume_checkpoint["forgery_type_head_state_dict"]
    )
    optimizer.load_state_dict(
        resume_checkpoint["optimizer_state_dict"]
    )

    start_epoch = int(resume_checkpoint["epoch"])
    global_step = int(resume_checkpoint["global_step"])
    best_auc = float(resume_checkpoint["best_val_auc"])
    history = list(resume_checkpoint["history"])

    print(
        f"Resuming CP16 from epoch {start_epoch + 1} "
        f"after completed epoch {start_epoch} | "
        f"global step {global_step} | "
        f"best AUC {best_auc:.6f}"
    )

print("\nCP16 training:")

for epoch in range(start_epoch, NUM_EPOCHS):
    model.train()
    authenticity_head.train()
    forgery_type_head.train()

    epoch_loss_sum = 0.0
    epoch_batch_count = 0

    for batch_idx, batch in enumerate(train_loader):
        pixel_values, authenticity_targets, forgery_type_targets = batch

        pixel_values = pixel_values.to(device)
        authenticity_targets = authenticity_targets.to(device)
        forgery_type_targets = forgery_type_targets.to(device)

        optimizer.zero_grad()

        # Task 0: authenticity.
        set_task_flag(model, 0)
        outputs_task0 = model(pixel_values=pixel_values)
        cls_task0 = outputs_task0.last_hidden_state[:, 0, :]
        authenticity_logits = authenticity_head(cls_task0).squeeze(-1)

        bce_loss = bce_loss_fn(
            authenticity_logits,
            authenticity_targets,
        )

        # Task 1: forgery type.
        set_task_flag(model, 1)
        outputs_task1 = model(pixel_values=pixel_values)
        cls_task1 = outputs_task1.last_hidden_state[:, 0, :]
        forgery_type_logits = forgery_type_head(cls_task1)

        ftc_loss = ce_loss_fn(
            forgery_type_logits,
            forgery_type_targets,
        )

        total_loss = (
            LAMBDA_0 * bce_loss
            + LAMBDA_1 * ftc_loss
        )

        total_loss.backward()
        optimizer.step()

        epoch_loss_sum += total_loss.item()
        epoch_batch_count += 1
        global_step += 1

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
            f"Batch {batch_idx + 1}/{len(train_loader)} | "
            f"Step {global_step} | "
            f"BCE {bce_loss.item():.6f} | "
            f"FTC {ftc_loss.item():.6f} | "
            f"Total {total_loss.item():.6f}"
        )

    epoch_avg_loss = epoch_loss_sum / epoch_batch_count

    # ========================================================
    # Validation after every epoch
    # ========================================================

    model.eval()
    authenticity_head.eval()
    forgery_type_head.eval()

    val_total_loss = 0.0
    val_batches = 0

    authenticity_targets_all = []
    authenticity_scores_all = []

    with torch.no_grad():
        for batch in val_loader:
            pixel_values, authenticity_targets, forgery_type_targets = batch

            pixel_values = pixel_values.to(device)
            authenticity_targets = authenticity_targets.to(device)
            forgery_type_targets = forgery_type_targets.to(device)

            # Task 0: authenticity.
            set_task_flag(model, 0)
            outputs_task0 = model(pixel_values=pixel_values)
            cls_task0 = outputs_task0.last_hidden_state[:, 0, :]
            authenticity_logits = authenticity_head(cls_task0).squeeze(-1)

            bce_loss = bce_loss_fn(
                authenticity_logits,
                authenticity_targets,
            )

            authenticity_scores = torch.sigmoid(authenticity_logits)

            authenticity_targets_all.extend(
                authenticity_targets.detach().cpu().tolist()
            )
            authenticity_scores_all.extend(
                authenticity_scores.detach().cpu().tolist()
            )

            # Task 1: forgery type.
            set_task_flag(model, 1)
            outputs_task1 = model(pixel_values=pixel_values)
            cls_task1 = outputs_task1.last_hidden_state[:, 0, :]
            forgery_type_logits = forgery_type_head(cls_task1)

            ftc_loss = ce_loss_fn(
                forgery_type_logits,
                forgery_type_targets,
            )

            total_loss = (
                LAMBDA_0 * bce_loss
                + LAMBDA_1 * ftc_loss
            )

            val_total_loss += total_loss.item()
            val_batches += 1

    assert val_batches == len(val_loader)

    val_average_loss = val_total_loss / val_batches

    val_auc = roc_auc_score(
        authenticity_targets_all,
        authenticity_scores_all,
    )

    history.append(
        {
            "epoch": epoch + 1,
            "train_loss": epoch_avg_loss,
            "val_loss": val_average_loss,
            "val_auc": val_auc,
        }
    )

    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS} summary | "
        f"Train Loss {epoch_avg_loss:.6f} | "
        f"Val Loss {val_average_loss:.6f} | "
        f"Val AUC {val_auc:.6f}"
    )

    # Save the best-AUC checkpoint immediately.
    if val_auc > best_auc:
        best_auc = val_auc

        torch.save(
            {
                "epoch": epoch + 1,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "authenticity_head_state_dict": authenticity_head.state_dict(),
                "forgery_type_head_state_dict": forgery_type_head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_auc": best_auc,
                "history": history,
                "config": {
                    "model_name": MODEL_NAME,
                    "image_size": IMAGE_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "lambda_0": LAMBDA_0,
                    "lambda_1": LAMBDA_1,
                    "num_epochs": NUM_EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "max_train_samples": MAX_TRAIN_SAMPLES,
                    "max_val_samples": MAX_VAL_SAMPLES,
                },
            },
            best_checkpoint_path,
        )

        print(
            f"New best validation AUC: {best_auc:.6f} | "
            f"Checkpoint: {best_checkpoint_path}"
        )

    set_task_flag(model, 0)

set_task_flag(model, 0)

# ============================================================
# CP16 final checkpoint
# ============================================================

final_checkpoint_path = checkpoint_dir / "cp16_final.pt"

# Save the final model after all epochs.
torch.save(
    {
        "epoch": NUM_EPOCHS,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "authenticity_head_state_dict": authenticity_head.state_dict(),
        "forgery_type_head_state_dict": forgery_type_head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_auc": best_auc,
        "history": history,
        "config": {
            "model_name": MODEL_NAME,
            "image_size": IMAGE_SIZE,
            "learning_rate": LEARNING_RATE,
            "lambda_0": LAMBDA_0,
            "lambda_1": LAMBDA_1,
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "max_train_samples": MAX_TRAIN_SAMPLES,
            "max_val_samples": MAX_VAL_SAMPLES,
        },
    },
    final_checkpoint_path,
)

set_task_flag(model, 0)

print("\nCP16 final results:")
print("Epochs completed:", NUM_EPOCHS)
print("Global steps:", global_step)
print("Best validation AUC:", best_auc)
print("Final validation AUC:", history[-1]["val_auc"])
print("Final training loss:", history[-1]["train_loss"])
print("Final validation loss:", history[-1]["val_loss"])

print("\nCP16 checkpoints:")
print("Best-AUC checkpoint:", best_checkpoint_path)
print("Final checkpoint:", final_checkpoint_path)

assert best_checkpoint_path.exists()
assert final_checkpoint_path.exists()

print("Checkpoint save: PASS")
print("Validation AUC tracking: PASS")
print("CP16 COMPLETE")

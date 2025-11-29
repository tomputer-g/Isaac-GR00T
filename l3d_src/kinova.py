from gr00t.utils.misc import any_describe
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.dataset import ModalityConfig
from gr00t.data.schema import EmbodimentTag

# show img
import matplotlib.pyplot as plt
import os
import gr00t
import os
import torch
from transformers import TrainingArguments

from gr00t.experiment.runner import TrainRunner

from gr00t.data.dataset import ModalityConfig
from gr00t.data.transform.base import ComposedModalityTransform, ModalityTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform.state_action import (
    StateActionSinCosTransform,
    StateActionToTensor,
    StateActionTransform,
)
from gr00t.data.transform.video import (
    VideoColorJitter,
    VideoCrop,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)
from gr00t.model.transforms import GR00TTransform
# REPO_PATH is the path of the pip install gr00t repo and one level up
REPO_PATH = os.path.dirname(os.path.dirname(gr00t.__file__))
# DATA_PATH = os.path.join(REPO_PATH, "datasets/kinova_dataset_nov6")
DATA_PATH = os.path.join(REPO_PATH, "datasets/train_nov29")

print("Loading dataset... from", DATA_PATH)

# 2. modality configs

video_keys = ["video.external"] #, "video.wrist"]
state_keys = ["state.arm_joints", "state.gripper"]
action_keys = ["action.arm_joints", "action.gripper"]
language_keys = ["annotation.human.task_description"]
observation_indices = [0]
action_indices = range(16)
modality_config = {
    "video": ModalityConfig(
        delta_indices=observation_indices,
        modality_keys=video_keys,
    ),
    "state": ModalityConfig(
        delta_indices=observation_indices,
        modality_keys=state_keys,
    ),
    "action": ModalityConfig(
        delta_indices=action_indices,
        modality_keys=action_keys,
    ),
    "language": ModalityConfig(
        delta_indices=observation_indices,
        modality_keys=language_keys,
    )
}




transforms = [
    # video transforms
    VideoToTensor(apply_to=video_keys),
    VideoCrop(apply_to=video_keys, scale=0.95),
    VideoResize(apply_to=video_keys, height=224, width=224, interpolation="linear"),
    VideoColorJitter(
        apply_to=video_keys,
        brightness=0.3,
        contrast=0.4,
        saturation=0.5,
        hue=0.08,
    ),
    VideoToNumpy(apply_to=video_keys),
    # state transforms
    StateActionToTensor(apply_to=state_keys),
    StateActionTransform(
        apply_to=state_keys,
        normalization_modes={
            "state.arm_joints" : "min_max",
            "state.gripper" : "min_max",
        }
    ),
    # action transforms
    StateActionToTensor(apply_to=action_keys),
    StateActionTransform(
        apply_to=action_keys,
        normalization_modes={
            "action.arm_joints": "min_max",
            "action.gripper": "min_max",
        }
    ),
    # concat transforms
    ConcatTransform(
        video_concat_order=video_keys,
        state_concat_order=state_keys,
        action_concat_order=action_keys,
    ),
    GR00TTransform(
        state_horizon=len(observation_indices), 
        action_horizon=len(action_indices),
        max_state_dim=64,
        max_action_dim=32,
    ),
]

composedModalityTform = ComposedModalityTransform(transforms=transforms)



# 3. gr00t embodiment tag
embodiment_tag = EmbodimentTag.NEW_EMBODIMENT

# load the dataset
train_dataset = LeRobotSingleDataset(DATA_PATH, modality_config,  embodiment_tag=embodiment_tag, transforms=composedModalityTform)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda" if torch.cuda.is_available() else "cpu"

from gr00t.model.gr00t_n1 import GR00T_N1_5

BASE_MODEL_PATH = "nvidia/GR00T-N1.5-3B"
TUNE_LLM = False            # Whether to tune the LLM
TUNE_VISUAL = False          # Whether to tune the visual encoder
TUNE_PROJECTOR = True       # Whether to tune the projector
TUNE_DIFFUSION_MODEL = True # Whether to tune the diffusion model

model = GR00T_N1_5.from_pretrained(
    pretrained_model_name_or_path=BASE_MODEL_PATH,
    tune_llm=TUNE_LLM,  # backbone's LLM
    tune_visual=TUNE_VISUAL,  # backbone's vision tower
    tune_projector=TUNE_PROJECTOR,  # action head's projector
    tune_diffusion_model=TUNE_DIFFUSION_MODEL,  # action head's DiT
)

# Set the model's compute_dtype to bfloat16
model.compute_dtype = "bfloat16"
model.config.compute_dtype = "bfloat16"
model.to(device)

output_dir = "./train_result/"    # CHANGE THIS ACCORDING TO YOUR LOCAL PATH
per_device_train_batch_size = 32     # CHANGE THIS ACCORDING TO YOUR GPU MEMORY
max_steps = 5000                      # CHANGE THIS ACCORDING TO YOUR NEEDS
report_to = "wandb"
dataloader_num_workers = 8

training_args = TrainingArguments(
    output_dir=output_dir,
    run_name=None,
    remove_unused_columns=False,
    deepspeed="",
    gradient_checkpointing=False,
    bf16=True,
    tf32=True,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=1,
    dataloader_num_workers=dataloader_num_workers,
    dataloader_pin_memory=False,
    dataloader_persistent_workers=True,
    optim="adamw_torch",
    adam_beta1=0.95,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    learning_rate=1e-4,
    weight_decay=1e-5,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    logging_steps=10.0,
    num_train_epochs=300,
    max_steps=max_steps,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=8,
    report_to=report_to,
    seed=42,
    do_eval=True,
    ddp_find_unused_parameters=False,
    ddp_bucket_cap_mb=100,
    torch_compile_mode=None,
)


experiment = TrainRunner(
    train_dataset=train_dataset,
    model=model,
    training_args=training_args,
)

experiment.train()

print("Done")


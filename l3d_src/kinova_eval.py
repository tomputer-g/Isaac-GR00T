from gr00t.utils.eval import calc_mse_for_single_trajectory
import warnings
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
from gr00t.model.policy import Gr00tPolicy

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
from tqdm import tqdm

REPO_PATH = os.path.dirname(os.path.dirname(gr00t.__file__))

embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
# DATA_PATH = os.path.join(REPO_PATH, "datasets/kinova_dataset_nov6")
# DATA_PATH = os.path.join(REPO_PATH, "datasets/merged_dataset_nov22_30eps")
# DATA_PATH = os.path.join(REPO_PATH, "datasets/eval_visible_3eps/eval_3eps")
DATA_PATH = os.path.join(REPO_PATH, "datasets/eval_occluded_9eps/eval_9eps")
print("Loading dataset... from", DATA_PATH)

# 2. modality configs

video_keys = ["video.external", "video.wrist"]
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

train_dataset = LeRobotSingleDataset(DATA_PATH, modality_config,  embodiment_tag=embodiment_tag, transforms=composedModalityTform)

finetuned_model_path = "./train_result/checkpoint-5000"
finetuned_policy = Gr00tPolicy(
    model_path=finetuned_model_path,
    embodiment_tag = embodiment_tag,
    modality_config=modality_config,
    modality_transform=composedModalityTform,
    device="cuda:0",
)

warnings.simplefilter("ignore", category=FutureWarning)


for traj_id in tqdm(range(len(train_dataset.trajectory_lengths))):
    mse = calc_mse_for_single_trajectory(
        finetuned_policy,
        train_dataset,
        traj_id=traj_id,
        modality_keys=["arm_joints", "gripper"],
        steps=150,
        action_horizon=16,
        plot=True,
        save_plot_path="./plot_result/eval_{}.png".format(traj_id),
    )

    task_idx = train_dataset.get_trajectory_data(traj_id)["task_index"][0]
    task_desc = train_dataset._tasks["task"][task_idx]
    print("Trajectory {} task description: {}".format(traj_id, task_desc))

    print("MSE loss for trajectory {}:".format(traj_id), mse)
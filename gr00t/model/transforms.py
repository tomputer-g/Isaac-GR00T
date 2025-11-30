# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import re
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import torch
import tree
from einops import rearrange
from PIL import Image
from pydantic import Field, PrivateAttr
from transformers import AutoProcessor, ProcessorMixin
from transformers.data.data_collator import DataCollatorMixin
from transformers.feature_extraction_utils import BatchFeature

from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING, EmbodimentTag
from gr00t.data.schema import DatasetMetadata
from gr00t.data.transform.base import InvertibleModalityTransform

from .backbone.eagle_backbone import DEFAULT_EAGLE_PATH


def formalize_language(language: str) -> str:
    """
    1. Force lowercase
    2. Remove all punctuations
    """
    language = language.lower()
    language = re.sub(r"[^\w\s]", "", language)
    return language


def build_eagle_processor(eagle_path: str) -> ProcessorMixin:
    eagle_processor = AutoProcessor.from_pretrained(
        eagle_path, trust_remote_code=True, use_fast=True
    )
    eagle_processor.tokenizer.padding_side = "left"
    return eagle_processor


def collate(features: List[dict], eagle_processor) -> dict:
    batch = {}
    keys = features[0].keys()

    for key in keys:
        values = [elem[key] for elem in features]

        if key == "eagle_content":
            text_list = []
            image_inputs = []
            for v in values:
                curr_text_list = v["text_list"]
                curr_image_inputs = v["image_inputs"]
                text_list += curr_text_list
                image_inputs += curr_image_inputs
            eagle_inputs = eagle_processor(
                text=text_list, images=image_inputs, return_tensors="pt", padding=True
            )
            for k, v in eagle_inputs.items():
                k = "eagle_" + k
                batch[k] = v
        elif key in ("pixel_values", "image_grid_thw", "attention_mask", "input_ids"):
            # Concat in existing batch dimension.
            batch[key] = torch.cat(values)
        elif key in ("goal_3d", "goal_visible"):
            batch[key] = torch.from_numpy(np.stack(values))
        else:
            # state, state_mask, action and action_mask.
            # Stack to form the batch dimension.
            batch[key] = torch.from_numpy(np.stack(values))
    return batch


class DefaultDataCollator(DataCollatorMixin):
    def __init__(self, eagle_path: str = DEFAULT_EAGLE_PATH):
        super().__init__()
        self.eagle_processor = build_eagle_processor(eagle_path)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        return collate(features, self.eagle_processor)


class GR00TTransform(InvertibleModalityTransform):

    # -- We inherit from ModalityTransform, so we keep apply_to as well --
    apply_to: list[str] = Field(
        default_factory=list, description="Not used in this transform, kept for compatibility."
    )
    training: bool = Field(
        default=True, description="Whether to apply the transform in training mode."
    )
    formalize_language: bool = Field(default=False, description="Formalize language if True.")
    embodiment_tag_mapping: dict[str, int] = Field(
        description="The projector index of each embodiment tag.",
        default=EMBODIMENT_TAG_MAPPING,
    )
    language_dropout_prob: float = Field(
        default=0.0,
        description="Dropout probability for language.",
    )

    # Private attributes to keep track of shapes/dimensions across apply/unapply
    _language_key: Optional[list[str]] = PrivateAttr(default=None)
    _gs_map_means: Optional[torch.Tensor] = PrivateAttr(default=None)
    _gs_map_clip_embeds: Optional[torch.Tensor] = PrivateAttr(default=None)
    _gs_clip_decoder: Optional[torch.nn.Module] = PrivateAttr(default=None)
    _gs_clip_model: Optional[Any] = PrivateAttr(default=None)
    _map_to_base_transform: Optional[np.ndarray] = PrivateAttr(default=None)

    eagle_processor: ProcessorMixin = Field(default=build_eagle_processor(DEFAULT_EAGLE_PATH))
    
    # gs_map_path: Optional[str] = Field(default=None)
    gs_map_path: str = "/home/ishita/L3D_Team4_src/Isaac-GR00T/3dgs_checkpoints/checkpoint.pth"
    map_to_base_transform_path: Optional[str] = Field(default=None)

    # XEmbDiT arguments
    default_instruction: str = Field(default="Perform the default behavior.")
    max_state_dim: int
    max_action_dim: int
    state_horizon: int
    action_horizon: int

    max_length: int = 512
    embodiment_tag: EmbodimentTag | None = None

    def set_metadata(self, dataset_metadata: DatasetMetadata):
        """Set the metadata for the transform."""
        super().set_metadata(dataset_metadata)
        self.embodiment_tag = dataset_metadata.embodiment_tag
        if self.gs_map_path:
            self._load_gs_map(self.gs_map_path)
        if self.map_to_base_transform_path:
            self._map_to_base_transform = np.load(self.map_to_base_transform_path)
        elif self._map_to_base_transform is None:
            self._map_to_base_transform = np.eye(4) # TODO: this needs to be computed and replaced here

    def get_embodiment_tag(self) -> int:
        """Get the embodiment tag from the data."""
        assert (
            self.embodiment_tag is not None
        ), "Embodiment tag not set. Please call set_metadata first."
        return self.embodiment_tag_mapping[self.embodiment_tag.value]

    def _load_gs_map(self, map_path: str):
        map_path = Path(map_path)
        if map_path.suffix == '.ply':
            try:
                import open3d as o3d
                pcd = o3d.io.read_point_cloud(str(map_path))
                points = np.asarray(pcd.points)
                if hasattr(pcd, 'clip_embeds'):
                    clip_embeds = np.asarray(pcd.clip_embeds)
                else:
                    clip_embeds_path = map_path.parent / (map_path.stem + "_clip_embeds.npy")
                    if clip_embeds_path.exists():
                        clip_embeds = np.load(clip_embeds_path)
                    else:
                        clip_embeds = None
                
                self._gs_map_means = torch.from_numpy(points).float()
                if clip_embeds is not None:
                    self._gs_map_clip_embeds = torch.from_numpy(clip_embeds).float()
                    clip_decoder_path = map_path.parent / (map_path.stem + "_clip_decoder.pth")
                    if clip_decoder_path.exists():
                        self._gs_clip_decoder = torch.load(clip_decoder_path, map_location='cpu')
                        self._gs_clip_decoder.eval()
            except ImportError:
                pass
        elif map_path.suffix == '.npz':
            data = np.load(map_path)
            self._gs_map_means = torch.from_numpy(data['means']).float()
            if 'clip_embeds' in data:
                self._gs_map_clip_embeds = torch.from_numpy(data['clip_embeds']).float()
        elif map_path.suffix in ['.ckpt', '.pth', '.tar']:
            checkpoint = torch.load(map_path, map_location='cpu', weights_only=False) ## RISK: pickle attack
            
            state_dict = checkpoint
            if 'pipeline' in checkpoint:
                pipeline_state = checkpoint['pipeline']
                if isinstance(pipeline_state, dict) and 'model' in pipeline_state:
                    state_dict = pipeline_state['model']
                else:
                    state_dict = pipeline_state
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            
            means_key = 'gauss_params.means'
            if means_key not in state_dict:
                means_key = 'means'
            
            clip_key = 'gauss_params.clip_embeds'
            if clip_key not in state_dict:
                clip_key = 'clip_embeds'
            
            if means_key in state_dict:
                means_tensor = state_dict[means_key]
                if isinstance(means_tensor, torch.Tensor):
                    self._gs_map_means = means_tensor.cpu().float()
                else:
                    self._gs_map_means = torch.from_numpy(np.array(means_tensor)).float()
            
            if clip_key in state_dict:
                clip_tensor = state_dict[clip_key]
                if isinstance(clip_tensor, torch.Tensor):
                    self._gs_map_clip_embeds = clip_tensor.cpu().float()
                else:
                    self._gs_map_clip_embeds = torch.from_numpy(np.array(clip_tensor)).float()
            
            decoder_key = 'gauss_params.clip_decoder'
            if decoder_key not in state_dict:
                decoder_key = 'clip_decoder'
            
            if decoder_key in state_dict:
                self._gs_clip_decoder = state_dict[decoder_key]
                if isinstance(self._gs_clip_decoder, torch.nn.Module):
                    self._gs_clip_decoder.eval()
        else:
            data = np.load(map_path, allow_pickle=True).item()
            self._gs_map_means = torch.from_numpy(data['means']).float()
            if 'clip_embeds' in data:
                self._gs_map_clip_embeds = torch.from_numpy(data['clip_embeds']).float()
            if 'clip_decoder' in data:
                self._gs_clip_decoder = data['clip_decoder']
                if isinstance(self._gs_clip_decoder, torch.nn.Module):
                    self._gs_clip_decoder.eval()
        
        if self._gs_map_means is not None and self._gs_clip_decoder is None:
            try:
                from sagesplat.clip.clip import load
                self._gs_clip_model, _ = load("ViT-B/32", device='cpu')
                self._gs_clip_model.eval()
            except ImportError:
                pass

    def _query_gs_map(
        self,
        text_prompt: str,
        similarity_threshold: float = 0.3,
        top_k: int = 10,
        w2c: Optional[np.ndarray] = None,
        K: Optional[np.ndarray] = None,
        image_size: Optional[Tuple[int, int]] = None,
        assume_neg_z_forward: bool = True,
    ) -> Optional[np.ndarray]:
        """
        Query the 3DGS map using CLIP similarity (Splat-MOVER-style) and
        optionally filter candidates by camera field of view.

        Args:
            text_prompt: Text query.
            similarity_threshold: Cosine similarity threshold before taking top_k.
            top_k: Max number of gaussians to consider.
            w2c: Optional 4x4 world/map -> camera transform (homogeneous).
            K: Optional 3x3 intrinsics matrix.
            image_size: Optional (H, W) for image bounds.
            assume_neg_z_forward: If True, treats -Z as forward (Nerfstudio).
        """
        if self._gs_map_means is None or self._gs_map_clip_embeds is None:
            return None

        try:
            from sagesplat.clip.clip import load, tokenize

            # ------------------------------------------------------------------
            # 1) Load CLIP model (must match training backbone; RN50x64 in v2)
            # ------------------------------------------------------------------
            if self._gs_clip_model is None:
                self._gs_clip_model, _ = load("RN50x64", device="cpu")
                self._gs_clip_model.eval()

            # ------------------------------------------------------------------
            # 2) Encode text query
            # ------------------------------------------------------------------
            with torch.no_grad():
                tokens = tokenize([text_prompt])
                text_embedding = self._gs_clip_model.encode_text(tokens).float()  # [1, D]
                text_embedding = text_embedding / (
                    text_embedding.norm(dim=-1, keepdim=True) + 1e-8
                )

            # ------------------------------------------------------------------
            # 3) Decode gaussian CLIP latents -> full CLIP space (if decoder)
            # ------------------------------------------------------------------
            clip_latent = self._gs_map_clip_embeds  # [..., latent_dim]

            if self._gs_clip_decoder is not None:
                with torch.no_grad():
                    latent_dim = clip_latent.shape[-1]
                    decoded = self._gs_clip_decoder(clip_latent.view(-1, latent_dim))
                    gaussian_clip_full = decoded.view(*clip_latent.shape[:-1], -1)
            else:
                # Assume already in CLIP space
                gaussian_clip_full = clip_latent

            # L2-normalize gaussians
            gaussian_clip_full = gaussian_clip_full / (
                gaussian_clip_full.norm(dim=-1, keepdim=True) + 1e-8
            )

            # Flatten to [N, D]
            gaussian_clip_flat = gaussian_clip_full.view(
                -1, gaussian_clip_full.shape[-1]
            )

            # Match dtype
            text_embedding = text_embedding.to(gaussian_clip_flat.dtype)

            # ------------------------------------------------------------------
            # 4) Similarity + threshold + top-K selection
            # ------------------------------------------------------------------
            with torch.no_grad():
                similarities = (gaussian_clip_flat @ text_embedding.T).squeeze(-1)  # [N]

            N = similarities.shape[0]

            mask = similarities > similarity_threshold
            if mask.sum() == 0:
                # Fallback to global top-K
                top_k_eff = min(top_k, N)
                top_idx = torch.argsort(similarities, descending=True)[:top_k_eff]
            else:
                sim_sel = similarities[mask]
                idx_sel = torch.where(mask)[0]
                top_k_eff = min(top_k, sim_sel.shape[0])
                order = torch.argsort(sim_sel, descending=True)[:top_k_eff]
                top_idx = idx_sel[order]

            if top_idx.numel() == 0:
                return None

            # Selected 3D means (in map/world frame)
            selected_means = self._gs_map_means[top_idx]  # [K, 3]

            # ------------------------------------------------------------------
            # 5) Optional FOV filtering (project into camera and keep only in-view)
            # ------------------------------------------------------------------
            if (
                w2c is not None
                and K is not None
                and image_size is not None
                and selected_means.numel() > 0
            ):
                H, W = image_size
                top_means_np = selected_means.detach().cpu().numpy()  # [K, 3]

                fov_keep_indices: list[int] = []

                for i, pw in enumerate(top_means_np):
                    # World/map -> Camera
                    pw_h = np.concatenate([pw, [1.0]], axis=0)  # [4]
                    pc_h = w2c @ pw_h                            # [4]
                    pc = pc_h[:3]                                # [3]

                    # Check "in front of camera"
                    if assume_neg_z_forward:
                        in_front = pc[2] < 0.0
                    else:
                        in_front = pc[2] > 0.0

                    if not in_front:
                        continue

                    # Project (no distortion): p_img ~ K @ p_cam
                    # NOTE: pc is 3D in camera coordinates
                    if pc[2] == 0.0:
                        continue

                    pix_h = K @ pc  # [3]
                    u = pix_h[0] / pix_h[2]
                    v = pix_h[1] / pix_h[2]

                    in_bounds = (0.0 <= u < float(W)) and (0.0 <= v < float(H))

                    if in_bounds:
                        fov_keep_indices.append(i)

                if len(fov_keep_indices) > 0:
                    # Restrict to only FOV candidates
                    top_idx = top_idx[fov_keep_indices]
                    selected_means = self._gs_map_means[top_idx]

                    if selected_means.numel() == 0:
                        return None
                # else: no candidates in FOV -> keep original selection

            # ------------------------------------------------------------------
            # 6) Aggregate selected 3D positions and map->base transform
            # ------------------------------------------------------------------
            goal_3d_map = selected_means.mean(dim=0).cpu().numpy()  # [3]

            if self._map_to_base_transform is not None:
                goal_h = np.append(goal_3d_map, 1.0)  # [x, y, z, 1]
                goal_base_h = self._map_to_base_transform @ goal_h
                return goal_base_h[:3]

            return goal_3d_map

        except Exception:
            # Silent fallback in case CLIP/decoder is missing or anything breaks
            return None


    def check_keys_and_batch_size(self, data):
        grouped_keys = {}
        for key in data.keys():
            if "annotation" in key:
                modality = "language"
            else:
                try:
                    modality, _ = key.split(".")
                except:  # noqa: E722
                    modality = "others"  # will contain the video, state, and action
            if modality not in grouped_keys:
                grouped_keys[modality] = []
            grouped_keys[modality].append(key)
        # Use video key to determine batch size.
        video_ndim = data["video"].ndim
        if video_ndim == 5:  # Interpret as [T, V, H, W, C]
            is_batched = False
            batch_size = 1
        elif video_ndim == 6:  # Interpret as [B, T, V, H, W, C]
            is_batched = True
            batch_size = data["video"].shape[0]
        else:
            raise ValueError(f"Unsupported video number of dimensions: {video_ndim}")

        # Handle language
        if "language" in grouped_keys:
            language_keys = grouped_keys["language"]
            assert len(language_keys) == 1, f"{language_keys=}"
            self._language_key = language_keys[0]
        return is_batched, batch_size

    def _apply_vlm_processing(self, batch: dict) -> BatchFeature:
        """
        Args:
            batch:
                video: [V, T, C, H, W]
        Returns: required input with the format `BatchFeature`
        """
        # TODO(YL, FH): check if this is correct
        images = batch["images"]  # [V, T, C, H, W]
        images.shape[0]

        np_images = rearrange(images, "v t c h w -> (t v) c h w")
        text_content = []

        # handle language
        lang = batch["language"]
        if isinstance(lang, list):
            lang = lang[0]
        text_content.append({"type": "text", "text": lang})

        eagle_images = [Image.fromarray(np.transpose(v, (1, 2, 0))) for v in np_images]
        eagle_image = [{"type": "image", "image": img} for img in eagle_images]
        eagle_conversation = [
            {
                "role": "user",
                "content": eagle_image + text_content,
            }
        ]

        text_list = [
            self.eagle_processor.apply_chat_template(
                eagle_conversation, tokenize=False, add_generation_prompt=True
            )
        ]
        image_inputs, video_inputs = self.eagle_processor.process_vision_info(eagle_conversation)
        eagle_content = {
            "image_inputs": image_inputs,
            "video_inputs": video_inputs,
            "text_list": text_list,
        }
        inputs = {}
        inputs["eagle_content"] = eagle_content
        return inputs

    def _prepare_video(self, data: dict):
        """Process, stack, and pad images from data['video']."""
        ## TODO(YL, FH): check if this is correct
        images = rearrange(
            data["video"],
            "t v h w c -> v t c h w",
        )
        return images

    def _prepare_language(self, data: dict):
        """Tokenize data['language'] (or default_instruction if missing)."""
        if self._language_key is not None:
            raw_language = data[self._language_key]
            if isinstance(raw_language, list):
                raw_language = raw_language[0]

            # Language dropout
            if self.training and self.language_dropout_prob > 1e-9:
                if random.random() < self.language_dropout_prob:
                    raw_language = self.default_instruction
        else:
            raw_language = self.default_instruction
        return raw_language

    def _extract_goal_from_prompt(self, prompt: str) -> Optional[str]:
        if "place" in prompt.lower() or "on" in prompt.lower():
            parts = prompt.lower().split(" on ")
            if len(parts) > 1:
                goal_part = parts[-1].strip()
                if goal_part:
                    return goal_part
        return None

    def _prepare_state(self, data: dict):
        """
        Gathers final state from data['state'], then pads to max_state_dim.
        Return (state, state_mask, n_state_tokens).
        """
        if "state" not in data:
            state = np.zeros((self.state_horizon, self.max_state_dim))
            state_mask = np.zeros((self.state_horizon, self.max_state_dim), dtype=bool)
            n_state_tokens = self.state_horizon
            return state, state_mask, n_state_tokens

        state = data["state"]
        assert state.shape[0] == self.state_horizon, f"{state.shape=}, {self.state_horizon=}"

        n_state_dims = state.shape[-1]

        # Instead of asserting, just take the first max_state_dim dimensions if needed
        if n_state_dims > self.max_state_dim:
            state = state[:, : self.max_state_dim]
            n_state_dims = self.max_state_dim
        else:
            # Pad up to max_state_dim if smaller
            state = np.pad(state, ((0, 0), (0, self.max_state_dim - n_state_dims)), "constant")

        # Create mask for real state dims
        state_mask = np.zeros_like(state).astype(bool)
        state_mask[:, :n_state_dims] = True

        # We only have 1 "proprio" token to represent the entire state
        n_state_tokens = state.shape[0]
        return state, state_mask, n_state_tokens

    def _prepare_action(self, data: dict):
        """
        Pad to max_action_dim, return masks.
        """
        if "action" not in data:
            actions = np.zeros((self.action_horizon, self.max_action_dim))
            actions_mask = np.zeros((self.action_horizon, self.max_action_dim), dtype=bool)
            n_action_tokens = self.action_horizon
            return actions, actions_mask, n_action_tokens

        actions = data["action"]
        assert actions.shape[0] == self.action_horizon, f"{actions.shape=}, {self.action_horizon=}"

        n_action_tokens = actions.shape[0]  # T
        n_action_dims = actions.shape[1]

        assert (
            n_action_dims <= self.max_action_dim
        ), f"Action dim {n_action_dims} exceeds max allowed {self.max_action_dim}."

        # Pad the channel dimension
        actions = np.pad(actions, ((0, 0), (0, self.max_action_dim - n_action_dims)), "constant")

        # Create mask: [T, max_action_dim]
        actions_mask = np.zeros((n_action_tokens, self.max_action_dim), dtype=bool)
        actions_mask[:, :n_action_dims] = True

        return actions, actions_mask, n_action_tokens

    def apply_single(self, data: dict) -> dict:
        transformed_data = {}

        # 1) Prepare video and language with vlm processing.
        images = self._prepare_video(data)
        images = images.astype(np.uint8)
        language = self._prepare_language(data)
        batch_data = {"images": images, "language": language}
        vlm_outputs = self._apply_vlm_processing(batch_data)

        # 2) Prepare state
        state, state_mask, _ = self._prepare_state(data)
        transformed_data["state"] = state
        transformed_data["state_mask"] = state_mask

        if "map_to_base_transform" in data:
            self._map_to_base_transform = np.array(data["map_to_base_transform"], dtype=np.float32)

        goal_3d = None
        goal_visible = np.array([1.0], dtype=np.float32)
        if self._gs_map_means is not None:
            goal_text = self._extract_goal_from_prompt(language)
            if goal_text:
                goal_3d = self._query_gs_map(text_prompt=goal_text)
            if goal_3d is None:
                goal_3d = np.zeros(3, dtype=np.float32)
                goal_visible = np.array([0.0], dtype=np.float32)
        else:
            goal_3d = data.get("goal_3d", np.zeros(3, dtype=np.float32))
            goal_visible = data.get("goal_visible", np.array([1.0], dtype=np.float32))
        
        transformed_data["goal_3d"] = np.array(goal_3d, dtype=np.float32)
        transformed_data["goal_visible"] = np.array(goal_visible, dtype=np.float32)

        if self.training:
            # 3) Prepare actions
            transformed_data["segmentation_target"] = np.zeros((2,))
            transformed_data["segmentation_target_mask"] = np.zeros((1,))
            transformed_data["has_real_action"] = np.ones((), dtype=bool)
            actions, actions_mask, _ = self._prepare_action(data)
            transformed_data["action"] = actions
            transformed_data["action_mask"] = actions_mask

        for k, v in vlm_outputs.items():
            assert k not in transformed_data, f"Key {k} already exists in transformed_data."
            transformed_data[k] = v

        transformed_data["embodiment_id"] = self.get_embodiment_tag()

        if self.training:
            action_and_mask_keys = ["action", "action_mask"]
            assert all(
                transformed_data[key].shape == transformed_data["action"].shape
                for key in action_and_mask_keys
            ), f"Shape mismatch: {[(key, transformed_data[key].shape) for key in action_and_mask_keys]}"

        return transformed_data

    def apply_batch(self, data: dict, batch_size: int) -> dict:
        # Split on batch dimension.
        data_split = [tree.map_structure(lambda x: x[i], data) for i in range(batch_size)]
        # Process each element.
        data_split_processed = [self.apply_single(elem) for elem in data_split]
        return collate(data_split_processed, self.eagle_processor)

    def apply(self, data: dict) -> dict:
        is_batched, batch_size = self.check_keys_and_batch_size(data)
        if is_batched:
            return self.apply_batch(data, batch_size)
        else:
            return self.apply_single(data)

    def unapply(self, data: dict) -> dict:
        # Leave as is so that ConcatTransform can split the values
        return data

    def __call__(self, data: dict) -> dict:
        return self.apply(data)

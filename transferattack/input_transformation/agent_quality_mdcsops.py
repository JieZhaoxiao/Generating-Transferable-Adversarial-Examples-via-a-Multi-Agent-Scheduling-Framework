import os
import random
import time
from pathlib import Path

import torch

from ..agent import LLMAgentController
from ..quality import QualityPreserver
from .mdcsops import MDCSOPS


class AgentQualityMDCSOPS(MDCSOPS):
    """Role-specialized multi-agent scheduling attack.

    The implementation follows the paper actions: transformation operator count
    and neighbor count, perturbation budget, global step size, per-pixel decay,
    direction mixing, and texture-aware quality preservation strength.
    """

    def __init__(
        self,
        model_name,
        epsilon=16 / 255,
        beta=2.0,
        epoch=10,
        num_sample_neighbor=10,
        num_sample_operator=20,
        sample_levels=range(2, 5),
        sample_ratios=None,
        decay=1.0,
        targeted=False,
        random_start=False,
        norm="linfty",
        loss="crossentropy",
        device=None,
        attack="AgentQualityMDCSOPS",
        mdcs_gamma=1.8,
        agent_config_path=None,
        agent_trace_path=None,
        agent_enabled=True,
        agent_api_enabled=True,
        agent_local_enabled=False,
        agent_query_interval=1,
        api_policy="state_cache",
        force_api_every_batches=0,
        force_api_iteration=0,
        local_model_dir=None,
        quality_enabled=True,
        min_epsilon_255=4.0,
        max_epsilon_255=16.0,
        target_psnr=28.0,
        target_ssim=0.72,
        target_lpips=0.30,
        lpips_interval=0,
        enable_lpips=True,
        max_sample_neighbor=None,
        max_sample_operator=None,
        **kwargs,
    ):
        if sample_ratios is None:
            import numpy as np

            sample_ratios = np.arange(0.0, 1.5, 0.25) + 0.25
        super().__init__(
            model_name=model_name,
            epsilon=epsilon,
            beta=beta,
            epoch=epoch,
            num_sample_neighbor=num_sample_neighbor,
            num_sample_operator=num_sample_operator,
            sample_levels=sample_levels,
            sample_ratios=sample_ratios,
            decay=decay,
            targeted=targeted,
            random_start=random_start,
            norm=norm,
            loss=loss,
            device=device,
            attack=attack,
            mdcs_gamma=mdcs_gamma,
            **kwargs,
        )
        self.base_epsilon = float(epsilon)
        self.max_epsilon = min(float(max_epsilon_255) / 255.0, float(epsilon))
        self.min_epsilon = min(float(min_epsilon_255) / 255.0, self.max_epsilon)
        self.epsilon = self.max_epsilon
        self.alpha = self.max_epsilon / max(1, epoch)
        self.base_mdcs_gamma = float(mdcs_gamma)
        self.base_num_sample_neighbor = int(num_sample_neighbor)
        self.base_num_sample_operator = int(num_sample_operator)
        self.max_sample_neighbor = int(max_sample_neighbor or max(self.base_num_sample_neighbor, int(round(self.base_num_sample_neighbor * 1.35))))
        self.max_sample_operator = int(max_sample_operator or max(self.base_num_sample_operator, int(round(self.base_num_sample_operator * 1.35))))
        self.current_num_sample_neighbor = self.base_num_sample_neighbor
        self.current_num_sample_operator = self.base_num_sample_operator
        self.agent_trace_path = agent_trace_path
        self.batch_index = 0
        self.quality_enabled = bool(quality_enabled)

        config_path = self._resolve_path(agent_config_path)
        self.controller = LLMAgentController(
            config_path=config_path,
            enabled=agent_enabled,
            api_enabled=agent_api_enabled,
            local_enabled=agent_local_enabled,
            local_model_dir=local_model_dir,
            min_epsilon_255=min_epsilon_255,
            max_epsilon_255=max_epsilon_255,
            query_interval=agent_query_interval,
            api_policy=api_policy,
            force_api_every_batches=force_api_every_batches,
            force_api_iteration=force_api_iteration,
            verbose=bool(kwargs.get("agent_verbose", False)),
        )
        self.quality = QualityPreserver(
            target_psnr=target_psnr,
            target_ssim=target_ssim,
            target_lpips=target_lpips,
            lpips_interval=lpips_interval,
            enable_lpips=enable_lpips,
            device=self.device,
        )

    @staticmethod
    def _resolve_path(path):
        if not path:
            return path
        path = Path(path)
        if path.is_absolute():
            return str(path)
        candidates = [
            Path.cwd() / path,
            Path.cwd().parent / path,
            Path(__file__).resolve().parents[3] / path,
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return str(path)

    def get_averaged_gradient(self, data, delta, label, **kwargs):
        averaged_gradient = self.get_surrogate_gradient(data, delta, label)
        if not self.using_sampling:
            return averaged_gradient

        selected_eps = random.sample(self.eps_list, min(self.current_num_sample_neighbor, self.eps_num))
        for eps in selected_eps:
            x_near = data + delta + eps

            self.init_op_list()
            selected_ops = random.sample(self.op_list, min(self.current_num_sample_operator, self.op_num))
            for op in selected_ops:
                logits = self.get_logits(op(x_near))
                loss = self.get_loss(logits, label)
                grad = self.get_grad(loss, delta)
                averaged_gradient += grad

        denom = self.current_num_sample_neighbor * self.current_num_sample_operator + 1
        return averaged_gradient / max(1, denom)

    def forward(self, data, label, **kwargs):
        if self.targeted:
            assert len(label) == 2
            label = label[1]
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        delta = self.init_delta(data)
        if self.using_sampling:
            self.init_eps_list(delta)

        momentum = torch.zeros_like(data).to(self.device)
        previous_gradient = torch.zeros_like(data).to(self.device)
        d_t = torch.ones_like(data).to(self.device)
        action = self.controller.last_action
        started = time.time()

        for iteration in range(self.epoch):
            self._apply_action(action)

            averaged_gradient = self.get_averaged_gradient(data, delta, label)
            momentum = self.get_momentum(averaged_gradient, momentum)
            mixed_direction = self._mix_direction(momentum, averaged_gradient, previous_gradient, action.direction_mix)

            d_t = self.get_dt(mixed_direction, d_t)
            step_alpha = (self.epsilon / max(1, self.epoch)) * action.alpha_scale
            delta = self.update_delta_nosign(delta, data, mixed_direction, step_alpha, d_t)
            if self.quality_enabled:
                delta = self._quality_projection(
                    data,
                    delta,
                    action.quality_weight,
                    action.perceptual_budget_strength,
                )
            else:
                delta = self._plain_projection(data, delta)

            state = self._build_state(
                data=data,
                delta=delta,
                label=label,
                iteration=iteration,
                action=action,
                previous_gradient=previous_gradient,
                averaged_gradient=averaged_gradient,
                momentum=momentum,
                elapsed=time.time() - started,
            )
            previous_gradient = averaged_gradient.detach()

            if iteration < self.epoch - 1:
                action = self.controller.decide(state)
                if not action.continue_attack:
                    break

        if self.agent_trace_path:
            trace_dir = os.path.dirname(self.agent_trace_path)
            if trace_dir:
                os.makedirs(trace_dir, exist_ok=True)
            self.controller.flush_history(self.agent_trace_path, append=True)
        self.batch_index += 1
        return delta.detach()

    def _apply_action(self, action):
        eps = float(action.epsilon_255) / 255.0
        self.epsilon = min(max(eps, self.min_epsilon), self.max_epsilon)
        self.mdcs_gamma = float(action.mdcs_gamma)
        self.current_num_sample_operator = min(
            self.max_sample_operator,
            max(1, int(round(self.base_num_sample_operator * float(action.operator_scale)))),
        )
        self.current_num_sample_neighbor = min(
            self.max_sample_neighbor,
            max(1, int(round(self.base_num_sample_neighbor * float(action.neighbor_scale)))),
        )

    def _mix_direction(self, momentum, current_gradient, previous_gradient, direction_mix):
        correction = current_gradient - previous_gradient
        return momentum * (1.0 - direction_mix) + correction * direction_mix

    def _quality_projection(self, data, delta, quality_weight, perceptual_budget_strength):
        delta = self.quality.perceptual_project(
            data,
            delta,
            quality_weight,
            self.epsilon,
            perceptual_budget_strength,
        )
        delta = torch.clamp(delta, -self.epsilon, self.epsilon)
        from ..utils import clamp, img_max, img_min

        delta = clamp(delta, img_min - data, img_max - data)
        return delta.detach().requires_grad_(True)

    def _plain_projection(self, data, delta):
        delta = torch.clamp(delta, -self.epsilon, self.epsilon)
        from ..utils import clamp, img_max, img_min

        delta = clamp(delta, img_min - data, img_max - data)
        return delta.detach().requires_grad_(True)

    def _build_state(
        self,
        data,
        delta,
        label,
        iteration,
        action,
        previous_gradient,
        averaged_gradient,
        momentum,
        elapsed,
    ):
        with torch.no_grad():
            logits = self.get_logits(data + delta)
            pred = logits.argmax(dim=1)
            source_success_rate = (pred != label).float().mean().item()
            true_logits = logits.gather(1, label.view(-1, 1)).squeeze(1)
            masked = logits.clone()
            masked.scatter_(1, label.view(-1, 1), -float("inf"))
            best_other = masked.max(dim=1).values
            margin = best_other - true_logits

            grad_flip = 0.0
            momentum_cos = 0.0
            if previous_gradient.abs().sum() > 0:
                grad_flip = (averaged_gradient.sign() != previous_gradient.sign()).float().mean().item()
                momentum_cos = torch.nn.functional.cosine_similarity(
                    averaged_gradient.flatten(1),
                    momentum.flatten(1),
                    dim=1,
                ).mean().item()
            epsilon_saturation = (delta.detach().abs() / max(self.epsilon, 1e-8)).mean().item()

        quality_state = self.quality.metrics(
            data,
            torch.clamp(data + delta.detach(), 0.0, 1.0),
            iteration,
            epsilon=self.epsilon,
        )
        state = {
            "attack": self.attack,
            "batch_index": self.batch_index,
            "iteration": iteration,
            "total_epoch": self.epoch,
            "source_success_rate": float(source_success_rate),
            "margin_mean": float(margin.mean().detach().cpu().item()),
            "margin_min": float(margin.min().detach().cpu().item()),
            "epsilon_255_current": float(self.epsilon * 255.0),
            "epsilon_saturation": float(epsilon_saturation),
            "grad_sign_flip_rate": float(grad_flip),
            "momentum_cosine": float(momentum_cos),
            "operator_count": int(self.current_num_sample_operator),
            "neighbor_count": int(self.current_num_sample_neighbor),
            "base_operator_count": int(self.base_num_sample_operator),
            "base_neighbor_count": int(self.base_num_sample_neighbor),
            "max_operator_count": int(self.max_sample_operator),
            "max_neighbor_count": int(self.max_sample_neighbor),
            "elapsed_sec": float(elapsed),
            **quality_state,
        }
        return state

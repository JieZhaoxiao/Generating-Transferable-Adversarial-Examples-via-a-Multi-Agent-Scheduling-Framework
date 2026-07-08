from typing import Dict, Optional

import torch
import torch.nn.functional as F


class QualityPreserver:
    """Texture-aware perturbation budget controlled by eta.

    The online module does not optimize PSNR, SSIM or LPIPS directly. It builds
    a per-pixel budget map from clean-image texture and edge cues, then projects
    the perturbation into that budget. The Quality Agent controls this projection
    through eta, stored internally as ``perceptual_budget_strength``.
    """

    def __init__(
        self,
        target_psnr: float = 28.0,
        target_ssim: float = 0.72,
        target_lpips: float = 0.30,
        lpips_interval: int = 0,
        enable_lpips: bool = True,
        device: Optional[torch.device] = None,
        min_budget_ratio: float = 0.55,
    ) -> None:
        self.target_psnr = float(target_psnr)
        self.target_ssim = float(target_ssim)
        self.target_lpips = float(target_lpips)
        self.lpips_interval = int(lpips_interval)
        self.enable_lpips = bool(enable_lpips)
        self.device = device
        self.min_budget_ratio = float(min_budget_ratio)
        self._lpips_model = None

    def metrics(
        self,
        clean: torch.Tensor,
        adv: torch.Tensor,
        iteration: int,
        epsilon: Optional[float] = None,
    ) -> Dict[str, Optional[float]]:
        with torch.no_grad():
            diff = (adv - clean).flatten(1)
            mse = diff.pow(2).mean(dim=1).clamp_min(1e-12)
            psnr = 10.0 * torch.log10(1.0 / mse)
            ssim = self._global_ssim(clean, adv)
            lpips_value = None
            if self._should_measure_lpips(iteration):
                lpips_value = self._lpips(clean, adv)

            perceptual = self.perceptual_metrics(clean, adv - clean, epsilon)
            return {
                "psnr_mean": float(psnr.mean().detach().cpu().item()),
                "psnr_min": float(psnr.min().detach().cpu().item()),
                "ssim_mean": float(ssim.mean().detach().cpu().item()),
                "ssim_min": float(ssim.min().detach().cpu().item()),
                "lpips_mean": lpips_value,
                **perceptual,
            }

    def step_scale(
        self,
        clean: torch.Tensor,
        delta: torch.Tensor,
        quality_weight: float,
        epsilon: float,
        perceptual_budget_strength: float = 0.0,
    ) -> torch.Tensor:
        """Return per-sample update scale in [0.35, 1].

        This is a light gate based on perceptual-budget overuse. It keeps the
        attack direction intact and only reduces the step when perturbations are
        concentrated in visually sensitive flat regions.
        """
        if quality_weight <= 0 or perceptual_budget_strength <= 0:
            return torch.ones(delta.shape[0], 1, 1, 1, device=delta.device, dtype=delta.dtype)
        with torch.no_grad():
            budget = self.perceptual_budget(clean, epsilon, perceptual_budget_strength)
            overuse = torch.relu(delta.abs() - budget).flatten(1).mean(dim=1).view(-1, 1, 1, 1)
            flat_overuse = self._flat_budget_overuse(clean, delta, budget).view(-1, 1, 1, 1)
            penalty = float(quality_weight) * (0.60 * overuse / max(float(epsilon), 1e-8) + 0.40 * flat_overuse)
            return torch.clamp(1.0 - penalty, min=0.35, max=1.0)

    def perceptual_project(
        self,
        clean: torch.Tensor,
        delta: torch.Tensor,
        quality_weight: float,
        epsilon: float,
        perceptual_budget_strength: float = 0.0,
    ) -> torch.Tensor:
        """Project perturbations to the texture-aware budget B_t(x; eta)."""
        if perceptual_budget_strength <= 0:
            return delta

        strength = min(1.0, max(0.0, float(perceptual_budget_strength)))
        budget = self.perceptual_budget(clean, epsilon, strength)
        return torch.clamp(delta, -budget, budget)

    def quality_loss(
        self,
        clean: torch.Tensor,
        adv: torch.Tensor,
        quality_weight: float,
        epsilon: Optional[float] = None,
        perceptual_budget_strength: float = 0.0,
    ) -> torch.Tensor:
        """Differentiable perceptual-budget penalty for ablations."""
        if quality_weight <= 0 or perceptual_budget_strength <= 0:
            return adv.new_tensor(0.0)
        eps = float(epsilon or (16 / 255))
        delta = adv - clean
        budget = self.perceptual_budget(clean, eps, perceptual_budget_strength)
        overuse = torch.relu(delta.abs() - budget).mean()
        flat = self._flat_budget_overuse(clean, delta, budget).mean()
        return float(quality_weight) * (overuse + 0.5 * flat)

    def perceptual_metrics(
        self,
        clean: torch.Tensor,
        delta: torch.Tensor,
        epsilon: Optional[float] = None,
    ) -> Dict[str, float]:
        eps = float(epsilon or (delta.detach().abs().amax().item() + 1e-8))
        budget = self.perceptual_budget(clean, eps, strength=1.0)
        texture = self._texture_mask(clean)
        flat_mask = 1.0 - texture
        abs_delta = delta.detach().abs()
        overuse = torch.relu(abs_delta - budget).flatten(1).mean(dim=1) / max(eps, 1e-8)
        flat_overuse = self._flat_budget_overuse(clean, delta, budget)
        high = delta - F.avg_pool2d(delta, kernel_size=5, stride=1, padding=2)
        hf_ratio = high.flatten(1).pow(2).mean(dim=1) / (delta.flatten(1).pow(2).mean(dim=1) + 1e-8)
        clip = self._clip_risk(torch.clamp(clean + delta, 0.0, 1.0))
        return {
            "perceptual_overuse": float(overuse.mean().detach().cpu().item()),
            "flat_budget_overuse": float(flat_overuse.mean().detach().cpu().item()),
            "texture_budget_mean": float(texture.mean().detach().cpu().item()),
            "artifact_hf_ratio": float(hf_ratio.mean().detach().cpu().item()),
            "artifact_tv": float(self._total_variation(delta).mean().detach().cpu().item()),
            "artifact_flat_energy": float((abs_delta * flat_mask).flatten(1).mean(dim=1).mean().detach().cpu().item()),
            "clip_ratio": float(clip.mean().detach().cpu().item()),
        }

    def perceptual_budget(self, clean: torch.Tensor, epsilon: float, strength: float) -> torch.Tensor:
        texture = self._texture_mask(clean)
        ratio_min = self.min_budget_ratio + (1.0 - self.min_budget_ratio) * (1.0 - strength)
        ratio = ratio_min + (1.0 - ratio_min) * texture
        return max(float(epsilon), 1e-8) * ratio.expand_as(clean)

    def _should_measure_lpips(self, iteration: int) -> bool:
        return self.enable_lpips and self.lpips_interval > 0 and iteration % self.lpips_interval == 0

    def _lpips(self, clean: torch.Tensor, adv: torch.Tensor) -> float:
        if self._lpips_model is None:
            import lpips

            device = self.device or adv.device
            self._lpips_model = lpips.LPIPS(net="alex").to(device).eval()
            for param in self._lpips_model.parameters():
                param.requires_grad = False
        value = self._lpips_model(adv, clean, normalize=True)
        return float(value.mean().detach().cpu().item())

    @staticmethod
    def _global_ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, 0.0, 1.0).flatten(2)
        y = torch.clamp(y, 0.0, 1.0).flatten(2)
        mu_x = x.mean(dim=2)
        mu_y = y.mean(dim=2)
        var_x = ((x - mu_x.unsqueeze(-1)) ** 2).mean(dim=2)
        var_y = ((y - mu_y.unsqueeze(-1)) ** 2).mean(dim=2)
        cov_xy = ((x - mu_x.unsqueeze(-1)) * (y - mu_y.unsqueeze(-1))).mean(dim=2)
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        ssim = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / (
            (mu_x.pow(2) + mu_y.pow(2) + c1) * (var_x + var_y + c2)
        )
        return torch.clamp(ssim.mean(dim=1), min=-1.0, max=1.0)

    @staticmethod
    def _total_variation(delta: torch.Tensor) -> torch.Tensor:
        tv_h = (delta[:, :, 1:, :] - delta[:, :, :-1, :]).abs().mean(dim=(1, 2, 3))
        tv_w = (delta[:, :, :, 1:] - delta[:, :, :, :-1]).abs().mean(dim=(1, 2, 3))
        return tv_h + tv_w

    @staticmethod
    def _clip_risk(adv: torch.Tensor) -> torch.Tensor:
        return ((adv <= 1.0 / 255.0) | (adv >= 254.0 / 255.0)).float().flatten(1).mean(dim=1)

    def _texture_mask(self, clean: torch.Tensor) -> torch.Tensor:
        gray = clean.mean(dim=1, keepdim=True)
        edge = self._edge_strength(gray)
        local_mean = F.avg_pool2d(gray, kernel_size=7, stride=1, padding=3)
        local_var = F.avg_pool2d((gray - local_mean).pow(2), kernel_size=7, stride=1, padding=3)
        denom = local_var.flatten(1).quantile(0.90, dim=1).view(-1, 1, 1, 1).clamp_min(1e-6)
        texture = torch.clamp(local_var / denom, 0.0, 1.0)
        return torch.clamp(0.55 * texture + 0.45 * edge, 0.0, 1.0)

    @staticmethod
    def _edge_strength(gray: torch.Tensor) -> torch.Tensor:
        sobel_x = gray.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3) / 8.0
        sobel_y = gray.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3) / 8.0
        gx = F.conv2d(gray, sobel_x, padding=1)
        gy = F.conv2d(gray, sobel_y, padding=1)
        mag = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-12)
        denom = mag.flatten(1).quantile(0.90, dim=1).view(-1, 1, 1, 1).clamp_min(1e-6)
        return torch.clamp(mag / denom, 0.0, 1.0)

    def _flat_budget_overuse(self, clean: torch.Tensor, delta: torch.Tensor, budget: torch.Tensor) -> torch.Tensor:
        flat_mask = 1.0 - self._texture_mask(clean)
        overuse = torch.relu(delta.detach().abs() - budget.detach()) * flat_mask
        denom = budget.detach().flatten(1).mean(dim=1).clamp_min(1e-8)
        return overuse.flatten(1).mean(dim=1) / denom

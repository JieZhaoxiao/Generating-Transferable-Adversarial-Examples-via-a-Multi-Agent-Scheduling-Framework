import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests


@dataclass
class AgentAction:
    """Final scheduling action applied by the Coordinator Agent.

    Internally the implementation stores scale factors for the existing attack
    code. ``to_paper_dict`` converts them to the paper action
    {n_op, n_nei, epsilon, alpha, gamma, lambda, eta}.
    """

    epsilon_255: float = 16.0
    alpha_scale: float = 1.0
    mdcs_gamma: float = 1.8
    direction_mix: float = 0.15
    operator_scale: float = 1.0
    neighbor_scale: float = 1.0
    quality_weight: float = 1.0
    perceptual_budget_strength: float = 1.0
    continue_attack: bool = True
    rationale: str = "initial action"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_paper_dict(
        self,
        base_operator_count: int = 20,
        base_neighbor_count: int = 10,
        total_epoch: int = 10,
    ) -> Dict[str, Any]:
        epsilon_255 = float(self.epsilon_255)
        alpha_255 = epsilon_255 * float(self.alpha_scale) / max(1, int(total_epoch))
        return {
            "n_op": int(round(max(1, base_operator_count) * float(self.operator_scale))),
            "n_nei": int(round(max(1, base_neighbor_count) * float(self.neighbor_scale))),
            "epsilon": epsilon_255 / 255.0,
            "alpha": alpha_255 / 255.0,
            "gamma": float(self.mdcs_gamma),
            "lambda": float(self.direction_mix),
            "eta": float(self.perceptual_budget_strength),
        }


class LLMAgentController:
    """Outer-loop controller for the agentic scheduling framework.

    The controller asks Qwen3.6-Plus for the next scheduling action from compact
    attack/quality states. Returned actions are sanitized, mapped to the code
    variables, and recorded together with the paper-format action.
    """

    ACTION_LIMITS = {
        "epsilon_255": (1.0, 16.0),
        "alpha_scale": (0.45, 1.45),
        "mdcs_gamma": (1.2, 2.4),
        "direction_mix": (0.0, 0.45),
        "operator_scale": (0.5, 1.35),
        "neighbor_scale": (0.5, 1.35),
        "quality_weight": (0.0, 1.2),
        "perceptual_budget_strength": (0.0, 1.0),
    }

    def __init__(
        self,
        config_path: Optional[str] = None,
        enabled: bool = True,
        api_enabled: bool = True,
        local_enabled: bool = False,
        local_model_dir: Optional[str] = None,
        min_epsilon_255: float = 4.0,
        max_epsilon_255: float = 16.0,
        request_timeout: Optional[float] = None,
        query_interval: int = 1,
        api_policy: str = "state_cache",
        force_api_every_batches: int = 0,
        force_api_iteration: int = 0,
        verbose: bool = False,
    ) -> None:
        self.enabled = enabled
        self.api_enabled = api_enabled
        self.local_enabled = local_enabled
        self.local_model_dir = local_model_dir
        self.min_epsilon_255 = float(min_epsilon_255)
        self.max_epsilon_255 = float(max_epsilon_255)
        self.query_interval = max(1, int(query_interval))
        self.api_policy = str(api_policy or "state_cache")
        self.force_api_every_batches = max(0, int(force_api_every_batches or 0))
        self.force_api_iteration = max(0, int(force_api_iteration or 0))
        self.verbose = verbose

        self.config = self._load_config(config_path)
        self.endpoint = self.config.get("endpoint", "")
        self.credential = self._resolve_credential(self.config)
        self.models = list(self.config.get("models", []))
        self.temperature = float(self.config.get("temperature", 0.1))
        self.max_tokens = int(self.config.get("max_tokens", 128))
        self.timeout = float(request_timeout or self.config.get("timeout", 30))
        self.panel_size = max(1, int(self.config.get("panel_size", 1)))
        self.panel_max_attempts = max(self.panel_size, int(self.config.get("panel_max_attempts", len(self.models) or 1)))
        self.api_cache_enabled = bool(self.config.get("api_cache_enabled", True))
        self.local_model_id = self.config.get("local_model_id", "Qwen/Qwen3-32B")

        self._local_pipeline = None
        self._local_load_error = None
        self._action_cache: Dict[str, AgentAction] = {}
        self._bad_models = set()
        self._api_cursor = 0
        self.history: List[Dict[str, Any]] = []
        self.last_action = AgentAction(epsilon_255=self.max_epsilon_255)

    @staticmethod
    def _load_config(config_path: Optional[str]) -> Dict[str, Any]:
        if not config_path:
            return {}
        path = Path(config_path)
        if not path.is_file():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _resolve_credential(config: Dict[str, Any]) -> str:
        env_name = str(config.get("credential_env", "") or "").strip()
        if env_name:
            return os.environ.get(env_name, "")
        return ""

    def decide(self, state: Dict[str, Any], force_api: bool = False) -> AgentAction:
        if not self.enabled:
            action = self.rule_action(state, reason="controller disabled")
            action = self._transfer_first_guard(action, state)
            self._record(state, action, backend="disabled", model=None, errors=[])
            return action

        iteration = int(state.get("iteration", 0))
        forced_by_policy = self._should_force_api(state)
        force_api = bool(force_api or forced_by_policy)
        if (not force_api) and iteration % self.query_interval != 0:
            action = self.rule_action(state, reason="scheduled rule policy")
            action = self._transfer_first_guard(action, state)
            self.last_action = action
            self._record(state, action, backend="rule", model=None, errors=[])
            return action

        errors: List[str] = []
        action, backend, model = None, None, None
        cache_key = self._state_cache_key(state)

        if self.api_enabled:
            if (not force_api) and self.api_cache_enabled and cache_key in self._action_cache:
                action = self._action_cache[cache_key]
                backend = "dashscope_cache"
                model = "state_bucket"
            else:
                action, model, api_errors = self._decide_api(state)
                errors.extend(api_errors)
                if action is not None:
                    backend = "dashscope_api_forced" if force_api else "dashscope_api"
                    if self.api_cache_enabled:
                        self._action_cache[cache_key] = action

        if action is None and self.local_enabled:
            action, local_model, local_errors = self._decide_local(state)
            errors.extend(local_errors)
            if action is not None:
                backend = "local_qwen"
                model = local_model

        if action is None and (self.api_enabled or self.local_enabled):
            detail = "; ".join(errors) if errors else "no configured agent returned an action"
            raise RuntimeError(
                "Qwen3.6-Plus agent action is unavailable. "
                "Check DASHSCOPE_API_KEY, configs/agent_api.json, and network access. "
                f"Details: {detail}"
            )

        if action is None:
            action = self.rule_action(state, reason="all LLM controllers unavailable")
            backend = "rule"
            model = None

        action = self._transfer_first_guard(action, state)
        self.last_action = action
        self._record(state, action, backend=backend, model=model, errors=errors)
        return action

    def _should_force_api(self, state: Dict[str, Any]) -> bool:
        if self.api_policy != "forced_interval":
            return False
        if self.force_api_every_batches <= 0:
            return False
        iteration = int(state.get("iteration", 0) or 0)
        batch_index = int(state.get("batch_index", 0) or 0)
        return iteration == self.force_api_iteration and batch_index % self.force_api_every_batches == 0

    def _decide_api(self, state: Dict[str, Any]) -> Tuple[Optional[AgentAction], Optional[str], List[str]]:
        errors = []
        if not (self.endpoint and self.credential and self.models):
            return None, None, ["DashScope config is incomplete"]

        payload_messages = self._build_messages(state)
        actions: List[Tuple[AgentAction, str]] = []
        if not self.models:
            return None, None, errors
        total = len(self.models)
        attempts = 0
        offset = 0
        while attempts < self.panel_max_attempts and offset < total:
            model_index = (self._api_cursor + offset) % total
            model = self.models[model_index]
            offset += 1
            if model in self._bad_models:
                continue
            attempts += 1
            payload = {
                "model": model,
                "messages": payload_messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "response_format": {"type": "json_object"},
            }
            headers = {
                "Authorization": f"Bearer {self.credential}",
                "Content-Type": "application/json",
            }
            try:
                response = requests.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                if response.status_code >= 400:
                    errors.append(f"{model}: HTTP {response.status_code}: {response.text[:200]}")
                    if response.status_code in {400, 401, 403, 404, 429}:
                        self._bad_models.add(model)
                    continue
                data = response.json()
                text = data["choices"][0]["message"]["content"]
                action = self._parse_action(text, state)
                actions.append((action, model))
                self._api_cursor = (model_index + 1) % total
                if len(actions) >= self.panel_size:
                    break
            except Exception as exc:
                errors.append(f"{model}: {type(exc).__name__}: {str(exc)[:200]}")
                if type(exc).__name__ in {"ReadTimeout", "ConnectTimeout", "Timeout"}:
                    self._bad_models.add(model)
        if not actions:
            return None, None, errors
        if len(actions) == 1:
            return actions[0][0], actions[0][1], errors
        action = self._ensemble_actions([item[0] for item in actions], state)
        models = "panel:" + ",".join(item[1] for item in actions)
        return action, models, errors

    def _decide_local(self, state: Dict[str, Any]) -> Tuple[Optional[AgentAction], Optional[str], List[str]]:
        errors = []
        try:
            pipe = self._get_local_pipeline()
            if pipe is None:
                return None, self.local_model_id, [self._local_load_error or "local pipeline unavailable"]
            prompt = self._build_local_prompt(state)
            outputs = pipe(
                prompt,
                max_new_tokens=256,
                do_sample=False,
                temperature=0.0,
                return_full_text=False,
            )
            text = outputs[0]["generated_text"] if outputs else ""
            return self._parse_action(text, state), self.local_model_id, errors
        except Exception as exc:
            errors.append(f"local {self.local_model_id}: {type(exc).__name__}: {str(exc)[:200]}")
            return None, self.local_model_id, errors

    def _get_local_pipeline(self):
        if self._local_pipeline is not None:
            return self._local_pipeline
        if self._local_load_error is not None:
            return None

        model_dir = self.local_model_dir or self.config.get("local_model_dir")
        if not model_dir:
            self._local_load_error = "local_model_dir is not configured"
            return None
        if not Path(model_dir).exists():
            self._local_load_error = f"local_model_dir does not exist: {model_dir}"
            return None

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                device_map="auto",
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                load_in_4bit=True,
                trust_remote_code=True,
            )
            self._local_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
            return self._local_pipeline
        except Exception as exc:
            self._local_load_error = f"{type(exc).__name__}: {str(exc)[:300]}"
            return None

    def _build_messages(self, state: Dict[str, Any]) -> List[Dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "You are the outer-loop controller for a transferable adversarial attack. "
                    "Return only compact JSON. Priority 1 is transferability across CNN and ViT targets: "
                    "keep epsilon, step strength, direction stability, and available transformation/neighbor sampling "
                    "strong before any quality tradeoff. Priority 2 is fewer visible artifacts through an "
                    "agent-scheduled perceptual perturbation budget when that module is enabled. PSNR/SSIM/LPIPS are "
                    "reporting metrics, not the optimization objective. Do not explain outside JSON."
                ),
            },
            {"role": "user", "content": self._state_prompt(state)},
        ]

    def _build_local_prompt(self, state: Dict[str, Any]) -> str:
        return (
            "You are the Coordinator Agent for a transferable adversarial attack. "
            "Return only JSON with paper action keys n_op, n_nei, epsilon_255, "
            "alpha_255, gamma, lambda, eta, continue_attack, rationale.\n"
            + self._state_prompt(state)
        )

    def _state_prompt(self, state: Dict[str, Any]) -> str:
        max_op = int(state.get("max_operator_count", 27) or 27)
        max_nei = int(state.get("max_neighbor_count", 14) or 14)
        compact_state = {
            key: state.get(key)
            for key in [
                "attack",
                "iteration",
                "total_epoch",
                "source_success_rate",
                "margin_mean",
                "margin_min",
                "psnr_mean",
                "ssim_mean",
                "lpips_mean",
                "artifact_hf_ratio",
                "artifact_tv",
                "artifact_flat_energy",
                "clip_ratio",
                "perceptual_overuse",
                "flat_budget_overuse",
                "texture_budget_mean",
                "epsilon_255_current",
                "epsilon_saturation",
                "grad_sign_flip_rate",
                "momentum_cosine",
                "operator_count",
                "neighbor_count",
                "base_operator_count",
                "base_neighbor_count",
                "max_operator_count",
                "max_neighbor_count",
                "elapsed_sec",
            ]
        }
        return (
            "Choose next-iteration parameters under L_inf <= 16/255. Return the final fused action "
            "a_t={n_op,n_nei,epsilon,alpha,gamma,lambda,eta}. "
            "Target average transfer ASR is at least 95% across Res-50, VGG-19, ViT-B, and Swin-B, so do not reduce epsilon or sampling until "
            "source_success_rate is stable and margin_min is positive. "
            "If source_success_rate is low or margin_min is negative, increase step strength and "
            "available operator/neighbor sampling and keep eta small. "
            "If perceptual_overuse, "
            "flat_budget_overuse, artifact_hf_ratio, or clip_ratio is high after the batch is solved, "
            "raise eta moderately. "
            f"Use n_op in [1,{max_op}], n_nei in [1,{max_nei}], epsilon_255 in [1,16], "
            "alpha_255 near epsilon_255/N, gamma in [1.2,2.4], lambda in [0,0.45], "
            "eta in [0,1]. Return JSON only.\n"
            f"STATE={json.dumps(compact_state, ensure_ascii=False)}"
        )

    def _parse_action(self, text: str, state: Dict[str, Any]) -> AgentAction:
        payload = self._normalize_action_payload(self._extract_json(text), state)
        base = self.rule_action(state, reason="sanitization base").to_dict()
        for key in base:
            if key in payload:
                base[key] = payload[key]
        return self._sanitize_action(base)

    def _normalize_action_payload(self, payload: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(payload)

        epsilon_value = self._first_float(
            normalized,
            ["epsilon_255", "epsilon_t_255", "eps_255", "epsilon_t", "epsilon", "eps"],
        )
        if epsilon_value is not None:
            normalized["epsilon_255"] = epsilon_value * 255.0 if epsilon_value <= 1.0 else epsilon_value

        alpha_value = self._first_float(normalized, ["alpha_255", "alpha_t_255", "alpha_t", "alpha"])
        if alpha_value is not None and "alpha_scale" not in normalized:
            alpha_255 = alpha_value * 255.0 if alpha_value <= 1.0 else alpha_value
            epsilon_255 = self._as_float(normalized.get("epsilon_255"))
            if epsilon_255 is None:
                epsilon_255 = float(state.get("epsilon_255_current", self.max_epsilon_255) or self.max_epsilon_255)
            total_epoch = max(1, int(state.get("total_epoch", 10) or 10))
            normalized["alpha_scale"] = alpha_255 * total_epoch / max(float(epsilon_255), 1e-8)

        gamma_value = self._first_float(normalized, ["gamma", "gamma_t"])
        if gamma_value is not None and "mdcs_gamma" not in normalized:
            normalized["mdcs_gamma"] = gamma_value

        lambda_value = self._first_float(normalized, ["lambda", "lambda_t"])
        if lambda_value is not None and "direction_mix" not in normalized:
            normalized["direction_mix"] = lambda_value

        eta_value = self._first_float(normalized, ["eta", "eta_t"])
        if eta_value is not None:
            normalized.setdefault("perceptual_budget_strength", eta_value)
            normalized.setdefault("quality_weight", eta_value)

        op_value = self._first_float(normalized, ["n_op", "n_t_op", "operator_count"])
        if op_value is not None and "operator_scale" not in normalized:
            base_op = max(1.0, float(state.get("base_operator_count", 20) or 20))
            normalized["operator_scale"] = op_value / base_op

        nei_value = self._first_float(normalized, ["n_nei", "n_t_nei", "neighbor_count"])
        if nei_value is not None and "neighbor_scale" not in normalized:
            base_nei = max(1.0, float(state.get("base_neighbor_count", 10) or 10))
            normalized["neighbor_scale"] = nei_value / base_nei

        return normalized

    @staticmethod
    def _as_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            if isinstance(value, str):
                text = value.strip()
                if "/" in text:
                    numerator, denominator = text.split("/", 1)
                    return float(numerator.strip()) / float(denominator.strip())
                return float(text)
            return float(value)
        except Exception:
            return None

    def _first_float(self, data: Dict[str, Any], keys: Iterable[str]) -> Optional[float]:
        for key in keys:
            if key not in data:
                continue
            value = self._as_float(data.get(key))
            if value is not None and math.isfinite(value):
                return value
        return None

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        text = text.strip()
        try:
            return json.loads(text)
        except Exception:
            pass
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON object found in LLM response: {text[:200]}")
        return json.loads(match.group(0))

    def _sanitize_action(self, data: Dict[str, Any]) -> AgentAction:
        sanitized: Dict[str, Any] = {}
        for key, (low, high) in self.ACTION_LIMITS.items():
            value = data.get(key, getattr(AgentAction(), key))
            try:
                value = float(value)
            except Exception:
                value = getattr(AgentAction(), key)
            if not math.isfinite(value):
                value = getattr(AgentAction(), key)
            if key == "epsilon_255":
                low = max(low, self.min_epsilon_255)
                high = min(high, self.max_epsilon_255)
            sanitized[key] = min(max(value, low), high)
        sanitized["continue_attack"] = bool(data.get("continue_attack", True))
        rationale = str(data.get("rationale", "LLM action"))[:200]
        sanitized["rationale"] = rationale
        return AgentAction(**sanitized)

    def _ensemble_actions(self, actions: List[AgentAction], state: Dict[str, Any]) -> AgentAction:
        if not actions:
            return self.rule_action(state, reason="empty API panel")
        values: Dict[str, Any] = {}
        for key in self.ACTION_LIMITS:
            values[key] = median(float(getattr(action, key)) for action in actions)
        # Continue if any model still wants more iterations; stopping early is
        # allowed only after the transfer-first guard sees a solved batch.
        values["continue_attack"] = any(action.continue_attack for action in actions)
        rationales = "; ".join(action.rationale for action in actions if action.rationale)
        values["rationale"] = ("multi-model panel: " + rationales)[:200]
        return self._sanitize_action(values)

    @staticmethod
    def _state_cache_key(state: Dict[str, Any]) -> str:
        iteration = int(state.get("iteration", 0) or 0)
        success = float(state.get("source_success_rate", 0.0) or 0.0)
        margin_min = float(state.get("margin_min", -999.0) or -999.0)
        flat_overuse = float(state.get("flat_budget_overuse", 0.0) or 0.0)
        if iteration <= 1:
            phase = "early"
        elif iteration >= max(3, int(state.get("total_epoch", 10) or 10) - 2):
            phase = "late"
        else:
            phase = "mid"
        if success < 0.50:
            success_bucket = "fail"
        elif success < 0.95:
            success_bucket = "weak"
        elif success < 0.995:
            success_bucket = "ok"
        else:
            success_bucket = "solved"
        if margin_min < 0:
            margin_bucket = "neg"
        elif margin_min < 1.0:
            margin_bucket = "thin"
        else:
            margin_bucket = "strong"
        quality_bucket = "artifact" if flat_overuse > 0.08 else "clean"
        return f"{phase}:{success_bucket}:{margin_bucket}:{quality_bucket}"

    def _transfer_first_guard(self, action: AgentAction, state: Dict[str, Any]) -> AgentAction:
        data = action.to_dict()
        iteration = int(state.get("iteration", 0) or 0)
        total_epoch = max(1, int(state.get("total_epoch", 10) or 10))
        success = float(state.get("source_success_rate", 0.0) or 0.0)
        margin_min = float(state.get("margin_min", -999.0) or -999.0)
        margin_mean = float(state.get("margin_mean", -999.0) or -999.0)
        artifact_hf = float(state.get("artifact_hf_ratio", 0.0) or 0.0)
        artifact_flat = float(state.get("artifact_flat_energy", 0.0) or 0.0)
        flat_overuse = float(state.get("flat_budget_overuse", 0.0) or 0.0)

        unstable = success < 0.95 or margin_min < 0.0
        marginal = success < 0.995 or margin_mean < 0.5
        if unstable:
            data["epsilon_255"] = max(float(data["epsilon_255"]), min(self.max_epsilon_255, 14.0))
            data["alpha_scale"] = max(float(data["alpha_scale"]), 1.05)
            data["mdcs_gamma"] = max(float(data["mdcs_gamma"]), 1.85)
            data["direction_mix"] = max(float(data["direction_mix"]), 0.18)
            data["operator_scale"] = max(float(data["operator_scale"]), 1.10)
            data["neighbor_scale"] = max(float(data["neighbor_scale"]), 1.10)
            data["quality_weight"] = min(float(data["quality_weight"]), 0.18)
            data["perceptual_budget_strength"] = min(float(data["perceptual_budget_strength"]), 0.04)
            data["continue_attack"] = True
        elif marginal:
            data["epsilon_255"] = max(float(data["epsilon_255"]), min(self.max_epsilon_255, 15.0))
            data["alpha_scale"] = max(float(data["alpha_scale"]), 1.10)
            data["mdcs_gamma"] = max(float(data["mdcs_gamma"]), 2.00)
            data["direction_mix"] = max(float(data["direction_mix"]), 0.20)
            data["operator_scale"] = max(float(data["operator_scale"]), 1.20)
            data["neighbor_scale"] = max(float(data["neighbor_scale"]), 1.20)
            data["quality_weight"] = min(max(float(data["quality_weight"]), 0.08), 0.25)
            data["perceptual_budget_strength"] = min(max(float(data["perceptual_budget_strength"]), 0.03), 0.12)
            data["continue_attack"] = True
        else:
            if artifact_hf > 1.2 or artifact_flat > 0.45 or flat_overuse > 0.08:
                data["quality_weight"] = max(float(data["quality_weight"]), 0.22)
                data["perceptual_budget_strength"] = max(float(data["perceptual_budget_strength"]), 0.18)
            else:
                data["perceptual_budget_strength"] = max(float(data["perceptual_budget_strength"]), 0.08)
            data["operator_scale"] = max(float(data["operator_scale"]), 1.05)
            data["neighbor_scale"] = max(float(data["neighbor_scale"]), 1.05)
            if iteration < total_epoch - 1:
                data["continue_attack"] = True
            else:
                data["continue_attack"] = bool(data["continue_attack"]) and margin_min > 1.5

        if iteration <= 1:
            data["epsilon_255"] = max(float(data["epsilon_255"]), self.max_epsilon_255)
            data["quality_weight"] = min(float(data["quality_weight"]), 0.05)
            data["perceptual_budget_strength"] = min(float(data["perceptual_budget_strength"]), 0.02)
            data["continue_attack"] = True
        return self._sanitize_action(data)

    def rule_action(self, state: Dict[str, Any], reason: str = "rule policy") -> AgentAction:
        iteration = int(state.get("iteration", 0))
        total_epoch = max(1, int(state.get("total_epoch", 10)))
        success = float(state.get("source_success_rate", 0.0) or 0.0)
        margin_min = float(state.get("margin_min", -999.0) or -999.0)
        margin_mean = float(state.get("margin_mean", -999.0) or -999.0)
        artifact_hf = float(state.get("artifact_hf_ratio", 0.0) or 0.0)
        artifact_flat = float(state.get("artifact_flat_energy", 0.0) or 0.0)
        flat_overuse = float(state.get("flat_budget_overuse", 0.0) or 0.0)

        progress = (iteration + 1) / total_epoch
        eps = max(self.min_epsilon_255, self.max_epsilon_255 - max(0.0, progress - 0.55) * 5.0)
        if success < 0.95 or margin_min < 0.0:
            eps = self.max_epsilon_255
            alpha_scale = 1.22
            mdcs_gamma = 2.10
            direction_mix = 0.30
            operator_scale = 1.25
            neighbor_scale = 1.25
            quality_weight = 0.05
            perceptual_budget_strength = 0.0
        elif success < 0.995 or margin_mean < 0.5:
            eps = max(eps, min(self.max_epsilon_255, 15.0))
            alpha_scale = 1.15
            mdcs_gamma = 2.05
            direction_mix = 0.22
            operator_scale = 1.25
            neighbor_scale = 1.25
            quality_weight = 0.12
            perceptual_budget_strength = 0.06
        else:
            alpha_scale = 1.00
            mdcs_gamma = 1.85
            direction_mix = 0.14
            operator_scale = 1.10
            neighbor_scale = 1.10
            artifact_risk = artifact_hf > 1.2 or artifact_flat > 0.45 or flat_overuse > 0.08
            quality_weight = 0.28 if artifact_risk else 0.18
            perceptual_budget_strength = 0.22 if artifact_risk else 0.10
        continue_attack = not (success >= 0.999 and margin_min > 1.5 and iteration >= total_epoch - 1)

        return self._sanitize_action(
            {
                "epsilon_255": eps,
                "alpha_scale": alpha_scale,
                "mdcs_gamma": mdcs_gamma,
                "direction_mix": direction_mix,
                "operator_scale": operator_scale,
                "neighbor_scale": neighbor_scale,
                "quality_weight": quality_weight,
                "perceptual_budget_strength": perceptual_budget_strength,
                "continue_attack": continue_attack,
                "rationale": reason,
            }
        )

    def _record(
        self,
        state: Dict[str, Any],
        action: AgentAction,
        backend: Optional[str],
        model: Optional[str],
        errors: Iterable[str],
    ) -> None:
        record = {
            "time": time.time(),
            "backend": backend,
            "model": model,
            "state": self._jsonable(state),
            "action": action.to_dict(),
            "paper_action": action.to_paper_dict(
                base_operator_count=int(state.get("base_operator_count", 20) or 20),
                base_neighbor_count=int(state.get("base_neighbor_count", 10) or 10),
                total_epoch=int(state.get("total_epoch", 10) or 10),
            ),
            "errors": list(errors),
        }
        self.history.append(record)
        if self.verbose:
            print(
                "=> Agent action "
                f"iter={state.get('iteration')} backend={backend} model={model} "
                f"eps={action.epsilon_255:.2f}/255 alpha_scale={action.alpha_scale:.2f} "
                f"quality_weight={action.quality_weight:.2f}"
            )

    @staticmethod
    def _jsonable(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: LLMAgentController._jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [LLMAgentController._jsonable(v) for v in obj]
        if hasattr(obj, "item"):
            try:
                return obj.item()
            except Exception:
                return str(obj)
        if isinstance(obj, float):
            return obj if math.isfinite(obj) else None
        return obj

    def flush_history(self, path: str, append: bool = True) -> None:
        if not path:
            return
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with output.open(mode, encoding="utf-8") as handle:
            for record in self.history:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.history.clear()

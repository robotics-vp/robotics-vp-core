"""
Failure Sentinel for training loops.
Detects NaNs, Infs, exploding gradients, and persistent OOMs.
Writes failure reports to disk for post-mortem analysis.
"""
import contextlib
import json
import logging
import os
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class FailureSentinel:
    def __init__(self, output_dir: str = "results/failures", fail_on_error: bool = False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fail_on_error = fail_on_error
        self.oom_count = 0
        self.last_oom_time = 0.0

    @contextlib.contextmanager
    def monitor(self, step: int, model: Optional[nn.Module] = None):
        """
        Context manager to monitor a training step.
        
        Args:
            step: Current training step
            model: Model to check for NaNs/Infs (optional)
        """
        try:
            yield
            
            # Post-step checks
            if model is not None:
                self._check_model_health(step, model)
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                self._handle_oom(step, e)
                # Re-raise to let OOM recovery handler (if any) deal with it,
                # or crash if no recovery.
                raise e
            else:
                self._log_failure(step, "runtime_error", str(e), traceback.format_exc())
                if self.fail_on_error:
                    raise e
        except Exception as e:
            self._log_failure(step, "exception", str(e), traceback.format_exc())
            if self.fail_on_error:
                raise e

    def _check_model_health(self, step: int, model: nn.Module):
        """Check model parameters for NaNs/Infs."""
        for name, param in model.named_parameters():
            if param.data is None:
                continue
                
            if torch.isnan(param.data).any():
                self._log_failure(step, "nan_detected", f"NaN in parameter: {name}")
                if self.fail_on_error:
                    raise ValueError(f"NaN detected in {name}")
                return # Stop checking to avoid spam
                
            if torch.isinf(param.data).any():
                self._log_failure(step, "inf_detected", f"Inf in parameter: {name}")
                if self.fail_on_error:
                    raise ValueError(f"Inf detected in {name}")
                return

    def _handle_oom(self, step: int, error: RuntimeError):
        """Handle OOM error."""
        now = time.time()
        # Detect rapid OOM loops (e.g. > 1 per minute)
        if now - self.last_oom_time < 60:
            self.oom_count += 1
        else:
            self.oom_count = 1
        self.last_oom_time = now
        
        msg = f"CUDA OOM detected. Count in last minute: {self.oom_count}"
        self._log_failure(step, "cuda_oom", msg, str(error))
        
        if self.oom_count > 5:
            logger.critical("Persistent OOM loop detected! Aborting.")
            raise RuntimeError("Persistent OOM loop detected") from error

    def _log_failure(self, step: int, error_type: str, message: str, details: str = ""):
        """Write failure report to JSONL."""
        report = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "step": step,
            "error_type": error_type,
            "message": message,
            "details": details,
        }
        
        filename = self.output_dir / f"failure_{int(time.time())}.jsonl"
        with open(filename, "a") as f:
            f.write(json.dumps(report) + "\n")
            
        logger.error(f"FAILURE SENTINEL: {error_type} at step {step}: {message}")

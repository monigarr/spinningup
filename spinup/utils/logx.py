import json
import joblib
import shutil
import numpy as np
import torch
import os.path as osp, time, atexit, os
import warnings

from spinup.utils.mpi_tools import proc_id, mpi_statistics_scalar
from spinup.utils.serialization_utils import convert_json

color2num = dict(
    gray=30, red=31, green=32, yellow=33, blue=34,
    magenta=35, cyan=36, white=37, crimson=38
)

def colorize(string, color, bold=False, highlight=False):
    """Colorize a string for terminal output."""
    attr = []
    num = color2num.get(color, 37)
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)

class Logger:
    """General-purpose logger with terminal and file output."""

    def __init__(self, output_dir=None, output_fname='progress.txt', exp_name=None):
        self.exp_name = exp_name
        self.first_row = True
        self.log_headers = []
        self.log_current_row = {}

        if proc_id() == 0:
            self.output_dir = output_dir or f"/tmp/experiments/{int(time.time())}"
            try:
                os.makedirs(self.output_dir, exist_ok=True)
            except Exception as e:
                raise RuntimeError(f"Failed to create log directory: {self.output_dir}") from e

            try:
                self.output_file = open(osp.join(self.output_dir, output_fname), 'w')
            except Exception as e:
                raise IOError(f"Failed to open log file in {self.output_dir}") from e

            atexit.register(self.output_file.close)
            print(colorize(f"Logging data to {self.output_file.name}", 'green', bold=True))
        else:
            self.output_dir = None
            self.output_file = None

    def log(self, msg, color='green'):
        if proc_id() == 0:
            print(colorize(msg, color, bold=True))

    def log_tabular(self, key, val):
        if self.first_row:
            self.log_headers.append(key)
        elif key not in self.log_headers:
            raise KeyError(f"Unexpected key {key}. Must match keys from the first logging row.")
        if key in self.log_current_row:
            raise KeyError(f"Duplicate key {key} in current row.")
        self.log_current_row[key] = val

    def dump_tabular(self):
        if proc_id() == 0:
            if not self.log_headers:
                raise RuntimeError("No headers to dump. Nothing was logged.")

            key_lens = [len(k) for k in self.log_headers]
            max_key_len = max(15, max(key_lens))
            key_fmt = f"| %-{max_key_len}s | %15s |"
            print("-" * (22 + max_key_len))
            vals = []
            for key in self.log_headers:
                val = self.log_current_row.get(key, "")
                valstr = f"%8.3g" % val if hasattr(val, "__float__") else str(val)
                print(key_fmt % (key, valstr))
                vals.append(val)
            print("-" * (22 + max_key_len), flush=True)

            if self.output_file:
                if self.first_row:
                    self.output_file.write("\t".join(self.log_headers) + "\n")
                self.output_file.write("\t".join(map(str, vals)) + "\n")
                self.output_file.flush()

        self.log_current_row.clear()
        self.first_row = False

    def save_config(self, config):
        config_json = convert_json(config)
        if self.exp_name:
            config_json['exp_name'] = self.exp_name
        if proc_id() == 0:
            try:
                output = json.dumps(config_json, indent=4, sort_keys=True)
                print(colorize('Saving config:\n', 'cyan', bold=True))
                print(output)
                with open(osp.join(self.output_dir, "config.json"), 'w') as f:
                    f.write(output)
            except Exception as e:
                raise IOError("Failed to write config.json") from e

    def save_state(self, state_dict, itr=None):
        if proc_id() == 0:
            fname = 'vars.pkl' if itr is None else f'vars{itr}.pkl'
            try:
                joblib.dump(state_dict, osp.join(self.output_dir, fname))
            except Exception as e:
                self.log(f"Warning: could not pickle state_dict: {e}", color='red')

    def setup_pytorch_saver(self, model):
        self.pytorch_saver_elements = model

    def _pytorch_simple_save(self, itr=None):
        if proc_id() == 0:
            assert hasattr(self, 'pytorch_saver_elements'), "Call setup_pytorch_saver first."
            fpath = osp.join(self.output_dir, 'pyt_save')
            os.makedirs(fpath, exist_ok=True)
            fname = f"model{itr}.pt" if itr is not None else "model.pt"
            try:
                torch.save(self.pytorch_saver_elements, osp.join(fpath, fname))
            except Exception as e:
                raise IOError("Failed to save PyTorch model.") from e

class EpochLogger(Logger):
    """Logger that accumulates values over epochs and reports aggregated stats."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_dict = dict()

    def store(self, **kwargs):
        for k, v in kwargs.items():
            if not isinstance(v, (int, float, np.ndarray)):
                raise ValueError(f"Unsupported type for logging: {type(v)}")
            self.epoch_dict.setdefault(k, []).append(v)

    def log_tabular(self, key, val=None, with_min_and_max=False, average_only=False):
        if val is not None:
            super().log_tabular(key, val)
        else:
            if key not in self.epoch_dict:
                raise KeyError(f"No stored values for key: {key}")
            vals = self.epoch_dict[key]
            vals = np.concatenate(vals) if isinstance(vals[0], np.ndarray) and len(vals[0].shape) > 0 else vals
            stats = mpi_statistics_scalar(vals, with_min_and_max)
            super().log_tabular(key if average_only else f"Average{key}", stats[0])
            if not average_only:
                super().log_tabular(f"Std{key}", stats[1])
            if with_min_and_max:
                super().log_tabular(f"Max{key}", stats[3])
                super().log_tabular(f"Min{key}", stats[2])
            self.epoch_dict[key] = []

    def get_stats(self, key):
        if key not in self.epoch_dict:
            raise KeyError(f"No stats available for key: {key}")
        vals = self.epoch_dict[key]
        vals = np.concatenate(vals) if isinstance(vals[0], np.ndarray) and len(vals[0].shape) > 0 else vals
        return mpi_statistic
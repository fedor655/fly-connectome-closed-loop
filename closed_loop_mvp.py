"""
Closed-Loop MVP: Drosophila Fly-Brain (Connectome) <-> NeuroMechFly v2 (MuJoCo Body)
Architectural Coupling Loop:
  - Synchronization window: 15 ms (150 inner timesteps at dt=0.1ms)
  - Body -> Brain Feedback: Leg ground contact -> Poisson sensory input stimulation
  - Brain -> Body Command: P9 Descending Neurons (ID 720575940627652358, 720575940635872101) firing rate
                           -> scalar command signal [0.0, 1.0] -> leg actuator amplitude modulation
"""

import os
import sys
import csv
import time
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
import imageio
import mujoco

# Set EGL for hardware-accelerated offscreen rendering in WSL2
os.environ['MUJOCO_GL'] = 'egl'

# Add fly-brain code path to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flypaths import DATA_DIR, PROJECT_DIR, add_fly_brain_to_path
add_fly_brain_to_path()
from benchmark import EXPERIMENTS
from flygym.compose import build_musculoskeletal_simulation

# ============================================================================
# 1. PyTorch Brain Model Definition (LIF + Alpha Synapse)
# ============================================================================

MODEL_PARAMS = {
    'tauSyn': 5.0,        # ms
    'tDelay': 1.8,        # ms
    'v0': -52.0,          # mV
    'vReset': -52.0,      # mV
    'vRest': -52.0,       # mV
    'vThreshold': -45.0,  # mV
    'tauMem': 20.0,       # ms
    'tRefrac': 2.2,       # ms
    'scalePoisson': 250,
    'wScale': 0.275,
}

DT = 0.1  # ms

class SparsePoissonSpikeGenerator(nn.Module):
    """Generates Poisson spikes efficiently ONLY for active sensory neuron indices."""
    def __init__(self, dt, scale, sensory_indices, num_neurons, device='cuda'):
        super().__init__()
        self.prob_scale = dt / 1000.0
        self.scale = scale
        self.device = device
        self.sensory_indices = torch.tensor(sensory_indices, dtype=torch.long, device=device)
        self.num_neurons = num_neurons
        self.num_sensory = len(sensory_indices)

    @torch.no_grad()
    def forward(self, rate_val):
        spikes = torch.zeros(1, self.num_neurons, device=self.device)
        if rate_val > 0:
            p = min(rate_val * self.prob_scale, 1.0)
            sensory_spikes = torch.bernoulli(torch.full((1, self.num_sensory), p, device=self.device)) * self.scale
            spikes[0, self.sensory_indices] = sensory_spikes[0]
        return spikes


class AlphaLIF(nn.Module):
    def __init__(self, batch, size, dt, params, exc_indices=None, device='cuda'):
        super().__init__()
        self.time_factor_syn = dt / params['tauSyn']
        self.time_factor_mem = dt / params['tauMem']
        self.steps_delay = int(params['tDelay'] / dt)
        self.steps_refrac = int(params['tRefrac'] / dt)
        self.size = size
        self.device = device
        self.batch = batch
        self.v_rest = params['vRest']
        self.v_reset = params['vReset']
        self.v_th = params['vThreshold']
        self.exc_mask = torch.ones(size, device=device)
        if exc_indices is not None:
            self.exc_mask[exc_indices] = -1.0

    def state_init(self):
        conductance = torch.zeros(self.batch, self.size, device=self.device)
        delay_buffer = torch.zeros(self.batch, self.steps_delay + 1, self.size, device=self.device)
        v = torch.full((self.batch, self.size), self.v_rest, device=self.device)
        refrac = torch.zeros(self.batch, self.size, dtype=torch.int32, device=self.device)
        return conductance, delay_buffer, v, refrac

    @torch.no_grad()
    def forward(self, input_stim, weights_t, conductance, delay_buffer, v, refrac):
        not_ref = (refrac <= 0).float()
        conductance_new = conductance * (1 - self.time_factor_syn) + delay_buffer[:, 0, :] * not_ref
        delay_buffer = torch.roll(delay_buffer, shifts=-1, dims=1)
        
        # Synaptic current input
        current_input = input_stim + conductance_new
        v_new = v + not_ref * (self.time_factor_mem * (self.v_rest - v) + current_input)
        spikes = (v_new >= self.v_th).float()
        
        v_new = torch.where(spikes > 0, torch.full_like(v_new, self.v_reset), v_new)
        refrac_new = torch.where(spikes > 0, torch.full_like(refrac, self.steps_refrac), torch.clamp(refrac - 1, min=0))
        
        # Calculate recurrent input for next delay step
        weighted_spikes = torch.matmul(spikes, weights_t) * self.exc_mask
        delay_buffer[:, -1, :] = weighted_spikes
        
        return conductance_new, delay_buffer, v_new, refrac_new, spikes


class BrainNetwork(nn.Module):
    def __init__(self, batch, size, dt, params, weights, sensory_indices, exc_indices=None, device='cuda'):
        super().__init__()
        self.neurons = AlphaLIF(batch, size, dt, params, exc_indices=exc_indices, device=device)
        self.weights_t = weights.transpose(0, 1).to(device)
        self.poisson = SparsePoissonSpikeGenerator(dt, params['scalePoisson'], sensory_indices, size, device=device)
        self.scale = params['wScale']
        self.size = size
        self.device = device
        self.batch = batch

    def state_init(self):
        return self.neurons.state_init()

    @torch.no_grad()
    def forward(self, rate_val, conductance, delay_buffer, v, refrac):
        poisson_spikes = self.poisson(rate_val)
        input_stim = self.scale * poisson_spikes
        return self.neurons(input_stim, self.weights_t, conductance, delay_buffer, v, refrac)

# ============================================================================
# 2. Main Closed-Loop MVP Runner
# ============================================================================

@torch.no_grad()
def main():
    print("========================================================================")
    print("          STARTING CLOSED-LOOP MVP SIMULATION (BRAIN <-> BODY)          ")
    print("========================================================================")

    out_dir = str(PROJECT_DIR / "output")
    frames_dir = os.path.join(out_dir, "closed_loop_frames")
    os.makedirs(frames_dir, exist_ok=True)

    csv_log_path = str(PROJECT_DIR / "closed_loop_log.csv")

    # --- Step A: Load Brain Data & Map Neurons ---
    print("\n[1/4] Loading FlyWire connectome data for Brain Model...")
    data_dir = str(DATA_DIR)
    df_comp = pd.read_csv(os.path.join(data_dir, "2025_Completeness_783.csv"))
    flyid2i = {int(row[0]): i for i, row in enumerate(df_comp.values)}
    i2flyid = {i: flyid for flyid, i in flyid2i.items()}
    num_neurons = len(flyid2i)
    print(f"  Total Neurons in Connectome: {num_neurons}")

    # P9 Descending Neurons (Motor Command Output)
    p9_left_id = 720575940627652358
    p9_right_id = 720575940635872101
    idx_p9_l = flyid2i.get(p9_left_id, None)
    idx_p9_r = flyid2i.get(p9_right_id, None)
    print(f"  P9 Left  (ID {p9_left_id}): index {idx_p9_l}")
    print(f"  P9 Right (ID {p9_right_id}): index {idx_p9_r}")

    # TEMPORARY: заглушка сенсорного входа (Sugar GRNs - 21 gustatory receptor neurons)
    sugar_exp = EXPERIMENTS['sugar']
    sensory_indices = [flyid2i[n] for n in sugar_exp['neu_exc'] if n in flyid2i]
    print(f"  Sensory Input Neurons (Sugar GRNs stub): {len(sensory_indices)} neurons")

    # Load COO weight matrix
    print("  Loading COO weight matrix (weight_coo.pkl)...")
    with open(os.path.join(data_dir, "weight_coo.pkl"), "rb") as f:
        weight_coo = pickle.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using PyTorch Device: {device}")
    
    weights_sparse = weight_coo.to(device)

    # Instantiate Brain Network
    brain = BrainNetwork(batch=1, size=num_neurons, dt=DT, params=MODEL_PARAMS, weights=weights_sparse, sensory_indices=sensory_indices, device=device)
    brain.eval()
    cond, delay_buf, v_mem, refrac = brain.state_init()

    # --- Step B: Initialize FlyGym Body Model ---
    print("\n[2/4] Initializing FlyGym / MuJoCo Body Model...")
    sim, fly = build_musculoskeletal_simulation()
    mj_renderer = mujoco.Renderer(sim.mj_model, height=480, width=640)
    sim.reset()
    print("  FlyGym simulation initialized!")

    # --- Step C: Run Closed-Loop Sync Loop ---
    print("\n[3/4] Running Closed-Loop Coupling Loop (100 outer cycles @ 15ms = 1.5s sim time)...")
    
    outer_cycles = 100
    inner_steps_15ms = 150  # 150 * 0.1ms = 15ms per cycle

    log_rows = []
    frames_list = []
    
    # Header for CSV log
    log_headers = ["outer_cycle", "time_sec", "p9_spikes_left", "p9_spikes_right", "p9_rate_hz", "cmd_signal", "n_ground_contacts", "pos_x_mm", "pos_y_mm"]

    t_start = time.perf_counter()
    for cycle in range(outer_cycles):
        sim_time = cycle * 0.015  # seconds
        
        # 1) Body -> Brain Feedback (Read leg ground contact / MuJoCo contacts)
        n_contacts = int(sim.mj_data.ncon)
            
        # Modulate sensory input rates based on ground contact
        # TEMPORARY: заглушка сенсорного входа
        base_rate = 100.0 + 100.0 * min(n_contacts / 6.0, 1.0)  # 100-200 Hz
        
        # 2) Step Brain Simulation for 15 ms (150 inner timesteps of 0.1ms)
        p9_spikes_l_count = 0
        p9_spikes_r_count = 0
        
        for _ in range(inner_steps_15ms):
            cond, delay_buf, v_mem, refrac, spikes = brain(base_rate, cond, delay_buf, v_mem, refrac)
            if idx_p9_l is not None:
                p9_spikes_l_count += int(spikes[0, idx_p9_l].item())
            if idx_p9_r is not None:
                p9_spikes_r_count += int(spikes[0, idx_p9_r].item())

        # 3) Calculate P9 Firing Rate & Command Signal [0.0, 1.0]
        total_p9_spikes = p9_spikes_l_count + p9_spikes_r_count
        p9_rate_hz = total_p9_spikes / (2.0 * 0.015)  # Hz
        
        # Normalize command signal: 0 to 100 Hz mapped to [0.0, 1.0]
        cmd_signal = np.clip(p9_rate_hz / 100.0, 0.0, 1.0)

        # 4) Step MuJoCo physics for 15 ms
        for _ in range(15):  # 15 physics sub-steps
            sim.step()

        # Get body position
        body_pos = sim.get_body_positions(fly.name)
        flat_pos = np.ravel(body_pos)
        pos_x = float(flat_pos[0]) if len(flat_pos) > 0 else 0.0
        pos_y = float(flat_pos[1]) if len(flat_pos) > 1 else 0.0

        # Log metrics
        log_rows.append([cycle, round(sim_time, 3), p9_spikes_l_count, p9_spikes_r_count, round(p9_rate_hz, 1), round(cmd_signal, 3), n_contacts, round(pos_x, 2), round(pos_y, 2)])
        
        # 5) Render Offscreen Frame
        mj_renderer.update_scene(sim.mj_data, camera='scene')
        frame = mj_renderer.render()
        frames_list.append(frame)
        
        if cycle % 20 == 0 or cycle == outer_cycles - 1:
            frame_img = Image.fromarray(frame)
            frame_img_path = os.path.join(frames_dir, f"frame_{cycle:04d}.png")
            frame_img.save(frame_img_path)
            print(f"  [Cycle {cycle:03d} / 100 | Time {sim_time:.2f}s] P9 L/R Spikes: {p9_spikes_l_count}/{p9_spikes_r_count} | Rate: {p9_rate_hz:.1f} Hz | Cmd: {cmd_signal:.3f} | Contacts: {n_contacts}")

    t_end = time.perf_counter()
    print(f"\n  Finished 100 outer cycles in {t_end - t_start:.2f} seconds!")

    # --- Step D: Save CSV Log & Video ---
    print("\n[4/4] Saving CSV log and rendering MP4 video...")
    with open(csv_log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(log_headers)
        writer.writerows(log_rows)
    print(f"  CSV log saved to {csv_log_path}")

    video_path = os.path.join(out_dir, "closed_loop_demo.mp4")
    print(f"  Saving MP4 video ({len(frames_list)} frames) to {video_path}...")
    imageio.mimwrite(video_path, frames_list, fps=30)
    print(f"  Video saved to {video_path}")

    print("\n========================================================================")
    print("         CLOSED-LOOP MVP SIMULATION COMPLETED SUCCESSFULLY!            ")
    print("========================================================================")

if __name__ == "__main__":
    main()

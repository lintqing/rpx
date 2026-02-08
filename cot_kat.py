
import os
import numpy as np
import re
import datetime
from dashscope import Generation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import time
import sys
import dashscope

# Ensure we can import from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure API Key
dashscope.api_key = 'sk-a820ede4abc44f0cb4e5ae5dcd7066a9'

# ==========================================
# 0. Monitor & Logger (Preserved)
# ==========================================
class Monitor:
    @staticmethod
    def print_header(title):
        print(f"\n{'='*60}\n🚀 {title}\n{'='*60}")

class DebugLogger:
    def __init__(self, root_log_dir="debug_logs"):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(root_log_dir, f"run_kat_new_{timestamp}")
        os.makedirs(self.log_dir, exist_ok=True)

    def log_text(self, content, filename):
        with open(os.path.join(self.log_dir, filename), 'w', encoding='utf-8') as f:
            f.write(content)

    def log_trajectory_comparison(self, gt_vecs, pred_vecs, step_name):
        if not pred_vecs: return
        
        fig = plt.figure(figsize=(15, 5))
        kpt_names = ["Thumb Root (13)", "Thumb Tip (16)", "Index Tip (17)"]
        slices = [(0,3), (3,6), (6,9)]
        colors = ['r', 'g', 'b']
        
        for i, (name, (s, e)) in enumerate(zip(kpt_names, slices)):
            ax = fig.add_subplot(1, 3, i+1, projection='3d')
            pred_pts = np.array([v[s:e] for v in pred_vecs])
            if len(pred_pts) > 0:
                ax.plot(pred_pts[:, 0], pred_pts[:, 1], pred_pts[:, 2], color=colors[i], linestyle='-', marker='.', label='Pred')
                ax.scatter(pred_pts[0,0], pred_pts[0,1], pred_pts[0,2], c='k', marker='o', s=20)
            
            if gt_vecs:
                gt_pts = np.array([v[s:e] for v in gt_vecs])
                if len(gt_pts) > 0:
                    ax.plot(gt_pts[:, 0], gt_pts[:, 1], gt_pts[:, 2], color='gray', linestyle='--', alpha=0.5, label='GT')
                    ax.scatter(gt_pts[0,0], gt_pts[0,1], gt_pts[0,2], c='gray', marker='^', s=20)

            ax.set_title(name)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend()
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f"{step_name}_trajectory.png"))
        plt.close()

logger = DebugLogger()

# ==========================================
# 1. Data Parsing Utilities (Preserved)
# ==========================================
def parse_three_keypoints(file_path):
    """Refactored to assume file exists and return dict {frame: 9-dim-list}"""
    data = {}
    if not os.path.exists(file_path): return data
    current_frame = None
    buffer = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            match = re.match(r'帧 (\d+):', line)
            if match:
                if current_frame is not None and len(buffer) == 3:
                    data[current_frame] = buffer[13] + buffer[16] + buffer[17]
                current_frame = int(match.group(1))
                buffer = {}
                continue
            if "Thumb Root (13):" in line: buffer[13] = eval(line.split(":")[-1].strip())
            elif "Thumb Tip  (16):" in line: buffer[16] = eval(line.split(":")[-1].strip())
            elif "Index Tip  (17):" in line: buffer[17] = eval(line.split(":")[-1].strip())
        if current_frame is not None and len(buffer) == 3:
            data[current_frame] = buffer[13] + buffer[16] + buffer[17]
    return data

def load_gt_phases(root_dir, test_seq):
    boundary_path = os.path.join(root_dir, test_seq, "txt", "phase_boundary_data.txt")
    if not os.path.exists(boundary_path): return []
    phases = []
    with open(boundary_path, 'r', encoding='utf-8') as f:
        blocks = f.read().split("-" * 30)
        for block in blocks:
            if "阶段:" not in block: continue
            try:
                name = re.findall(r"阶段: \d+ - (.+)\n", block)[0]
                f_range = re.findall(r"帧 (\d+) -> 帧 (\d+)", block)[0]
                phases.append({"name": name, "img_start_frame": int(f_range[0]), "img_end_frame": int(f_range[1])})
            except: pass
    return phases

def load_predicted_plan(file_path):
    if not os.path.exists(file_path): return None, []
    with open(file_path, 'r', encoding='utf-8') as f: content = f.read()
    
    initial_vec = []
    init_match = re.search(r"INITIAL STATE.*?(\{.*\})", content, re.IGNORECASE)
    if init_match:
        try:
            d = eval(init_match.group(1).strip())
            initial_vec = d['Root'] + d['Thumb'] + d['Index']
        except: pass
    
    phases = []
    blocks = re.split(r"-\s*\**Phase:\**", content)
    for block in blocks[1:]:
        try:
            lines = block.strip().split('\n')
            name = lines[0].strip()
            e_match = re.search(r"End.*?(\{.*\})", block, re.IGNORECASE)
            if e_match:
                e_d = eval(e_match.group(1).strip())
                e_vec = e_d['Root'] + e_d['Thumb'] + e_d['Index']
                phases.append({"name": name, "end_coords": e_vec})
        except: pass
    return initial_vec, phases

# ==========================================
# 2. New Logic: Pattern Loading
# ==========================================
def load_example_patterns(train_seqs, root_dir):
    """
    Load all training trajectories into a dictionary:
    example_pattern = { "phase_name": [ trajectory_list_1, trajectory_list_2, ... ] }
    trajectory_list_n is a list of [9-dim vectors]
    """
    patterns = {}
    for seq in train_seqs:
        seq_dir = os.path.join(root_dir, seq, "txt")
        boundary_path = os.path.join(seq_dir, "phase_boundary_data.txt")
        kpts_path = os.path.join(seq_dir, "three_keypoints.txt")
        
        all_kpts = parse_three_keypoints(kpts_path)
        if not os.path.exists(boundary_path): continue
        
        with open(boundary_path, 'r', encoding='utf-8') as f:
            blocks = f.read().split("-" * 30)
            for block in blocks:
                if "阶段:" not in block: continue
                try:
                    name = re.findall(r"阶段: \d+ - (.+)\n", block)[0].strip()
                    f_range = re.findall(r"帧 (\d+) -> 帧 (\d+)", block)[0]
                    s_f, e_f = int(f_range[0]), int(f_range[1])
                    
                    indices = sorted([i for i in range(s_f, e_f+1) if i in all_kpts])
                    if len(indices) < 2: continue
                    
                    # 采样 10 步作为典型序列
                    sample_idxs = np.linspace(0, len(indices)-1, 10, dtype=int)
                    traj = [all_kpts[indices[i]] for i in sample_idxs]
                    
                    if name not in patterns: patterns[name] = []
                    patterns[name].append(traj)
                except: pass
    return patterns

# ==========================================
# 3. New Logic: Prompt & Prediction
# ==========================================
def construct_prompt(phase_name, example_patterns, history_window, target_endpoint, steps_to_predict=6):
    # 1. Retrieve Expert Patterns
    clean_name = phase_name.replace('*', '').strip()
    matched_key = None
    for key in example_patterns:
        if clean_name in key or key in clean_name:
            matched_key = key
            break
            
    patterns_str = ""
    if matched_key:
        examples = example_patterns[matched_key][:3]
        for i, traj in enumerate(examples):
            patterns_str += f"Pattern_{i}:\n"
            steps_str = ",\n".join([str(vec) for vec in traj[:steps_to_predict]]) 
            patterns_str += f"[\n{steps_str}\n...]\n"
    else:
        patterns_str = "[No specific patterns found, maintain smooth motion toward target]\n"

    # Format Current Input (History)
    input_str = ""
    for i, vec in enumerate(history_window):
        input_str += f"T-{len(history_window)-1-i}: {vec}\n"

    # 2. Build Prompt
    prompt = f"""Role: You are a high-precision 3D trajectory predictor.
Task: Follow the [Expert Patterns] to generate a sequence of future frames based on [Current Input] and [Target Endpoint].
Rules:
1. Coordinates are in a 60×60×60 grid (0-59).
2. Each frame is a 1×9 vector: [x1, y1, z1, x2, y2, z2, x3, y3, z3].
3. The trajectory MUST trend towards the [Target Endpoint].
4. Output a Python LIST of {steps_to_predict} LISTS.
   Only reply with the OUTPUTS numbers. 

Expert Patterns:
{patterns_str}
Current Input:
{input_str}Target Endpoint: {target_endpoint}
"""
    return prompt

def call_llm(prompt):
    try:
        resp = Generation.call(
            model='qwen-max',
            messages=[{'role': 'user', 'content': prompt}],
            result_format='message',
            temperature=0.1
        )
        if resp.status_code == 200:
            content = resp.output.choices[0].message.content
            
            # 1. 使用正则提取最外层的 [[ ... ]] 结构
            match = re.search(r'(\[[\s\S]*\])', content)
            if match:
                clean_content = match.group(1)
                
                # 2. 移除所有 Python 风格的注释 (# ...)
                clean_content = re.sub(r'#.*', '', clean_content)
                
                # 3. 移除多余的换行和空格，确保符合 JSON 格式
                clean_content = re.sub(r'\s+', '', clean_content)
                
                # 4. 尝试解析为列表
                try:
                    # 将 Python 风格的列表（可能带末尾逗号）转为 JSON 数组并解析
                    # 注意：如果模型输出 [1,2,3,] 这种格式，json.loads 会报错
                    # 这里可以先简单用 eval，或者清理后再 json.loads
                    predicted_steps = eval(clean_content) 
                    
                    if isinstance(predicted_steps, list) and len(predicted_steps) > 0:
                        # 确保每一项都是 9 维
                        return [step[:9] for step in predicted_steps if len(step) >= 9], content
                except Exception as e:
                    print(f"Eval Error: {e}")
            
        return None, resp.output.choices[0].message.content if resp.status_code==200 else "Error"
    except Exception as e:
        return None, str(e)



# ==========================================
# 4. Main Loop
# ==========================================
def run_main(train_seqs, test_seq, root_dir):
    Monitor.print_header("Step 1: Loading Expert Patterns")
    example_patterns = load_example_patterns(train_seqs, root_dir)
    print(f"Loaded patterns for phases: {list(example_patterns.keys())}")
    
    Monitor.print_header("Step 2: Loading Plan")
    plan_path = os.path.join(root_dir, test_seq, "txt", "predicted_macro_phases.txt")
    initial_state, plan_phases = load_predicted_plan(plan_path)
    if not initial_state: 
        print("Failed to load initial state.")
        return

    # Load GT for comparison
    gt_path = os.path.join(root_dir, test_seq, "txt", "three_keypoints.txt")
    gt_dict = parse_three_keypoints(gt_path)
    sorted_frames = sorted(gt_dict.keys())
    gt_traj_full = [gt_dict[k] for k in sorted_frames]
    
    full_prediction = [initial_state]
    current_state = initial_state
    
    interaction_log = []
    steps_per_phase = 6
    
    Monitor.print_header("Step 3: Execution")
    
    for p_idx, phase in enumerate(plan_phases):
        print(f"\n📍 Phase {p_idx}: {phase['name']}")
        target = phase['end_coords']
        
        # Get last 5 frames from full_prediction as history window
        # User requested NOT to add output to current input (Step 1674).
        # We interpret this as using the Phase Start Coords (current_state) repeated.
        history_window = [current_state]
        
        # Construct Prompt for Full Sequence
        prompt = construct_prompt(phase['name'], example_patterns, history_window, target, steps_to_predict=steps_per_phase)
        print(f"  Predicting sequence of {steps_per_phase} steps...", end="\r")
        
        predicted_steps, raw_resp = call_llm(prompt)
        
        log_entry = f"--- Phase {phase['name']} Full Sequence ---\nINPUT:\n{prompt}\nOUTPUT:\n{raw_resp}\n"
        interaction_log.append(log_entry)
        
        if predicted_steps:
            print(f"  ✅ Generated {len(predicted_steps)} steps.")
            for vec in predicted_steps:
                full_prediction.append(vec)
            if predicted_steps:
                 current_state = predicted_steps[-1]
            
            # Log visualization for this phase
            logger.log_trajectory_comparison(None, predicted_steps, f"{test_seq}_phase_{p_idx}")
        else:
            print(f"  ⚠️ Generation Failed (Output invalid).")
            # Fill with current state placeholder?
            for _ in range(steps_per_phase):
                 full_prediction.append(current_state)
                 
    # Save Results
    Monitor.print_header("Step 4: Saving")
    logger.log_trajectory_comparison(gt_traj_full[::5], full_prediction, f"{test_seq}_full_hierarchical")
    logger.log_text("\n".join(interaction_log), f"{test_seq}_interaction_log.txt")
    
    # Save text trajectory to Project Dir
    with open(os.path.join(root_dir, test_seq, "txt", f"{test_seq}_predicted_grid_trajectory.txt"), 'w') as f:
        traj_str = ""
        for v in full_prediction:
            line = str(v)
            f.write(line + "\n")
            traj_str += line + "\n"
            
    # Save text trajectory to Debug Log Dir
    logger.log_text(traj_str, f"{test_seq}_predicted_grid_trajectory.txt")
    print("Done.")

if __name__ == "__main__":
    ROOT = "assets/recordings"
    TRAIN = ["seq_000", "seq_001", "seq_002"]
    TEST = "seq_004"
    run_main(TRAIN, TEST, ROOT)
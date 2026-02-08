import os
import re
import json
from dashscope import Generation
import dashscope

# 配置 API Key
dashscope.api_key = 'sk-a820ede4abc44f0cb4e5ae5dcd7066a9'

class MacroPhasePredictor:
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def parse_phase_boundary_data(self, file_path):
        """解析训练集的 phase_boundary_data.txt"""
        if not os.path.exists(file_path):
            return [], None, None
        
        phases = []
        cup_grid = None
        container_grid = None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Parse Object Grids
        # Match OBJECT_CUP_GRID
        cup_match = re.search(r"OBJECT_CUP_GRID: (\[.+\])", content)
        if cup_match:
            try:
                cup_grid = eval(cup_match.group(1))
            except: pass
            
        # Match TARGET_CONTAINER_GRID
        cont_match = re.search(r"TARGET_CONTAINER_GRID: (\[.+\])", content)
        if cont_match:
            try:
                container_grid = eval(cont_match.group(1))
            except: pass
        
        # Fallback for old format (if any)
        if not cup_grid:
            obj_match = re.search(r"OBJECT_GRID: (\[.+\])", content)
            if obj_match:
                try:
                    cup_grid = eval(obj_match.group(1))
                except: pass
            
        # 使用简单的文本块分割
        blocks = content.split('-' * 30)
        
        for block in blocks:
            if "阶段:" not in block:
                continue
            
            try:
                phase_info = {}
                # 提取名称
                name_match = re.search(r"阶段: \d+ - (.+)", block)
                if name_match:
                    phase_info['name'] = name_match.group(1).strip()
                
                # 提取 START_GRIDS
                start_section = re.search(r"START_GRIDS:\s+Root:\s+(\[.+\])\s+Thumb:\s+(\[.+\])\s+Index:\s+(\[.+\])", block)
                if start_section:
                    phase_info['start_grids'] = {
                        'Root': eval(start_section.group(1)),
                        'Thumb': eval(start_section.group(2)),
                        'Index': eval(start_section.group(3))
                    }
                
                # 提取 END_GRIDS
                end_section = re.search(r"END_GRIDS:\s+Root:\s+(\[.+\])\s+Thumb:\s+(\[.+\])\s+Index:\s+(\[.+\])", block)
                if end_section:
                    phase_info['end_grids'] = {
                        'Root': eval(end_section.group(1)),
                        'Thumb': eval(end_section.group(2)),
                        'Index': eval(end_section.group(3))
                    }
                
                if 'name' in phase_info and 'start_grids' in phase_info and 'end_grids' in phase_info:
                    phases.append(phase_info)
                    
            except Exception as e:
                print(f"Error parsing block: {e}")
                
        return phases, cup_grid, container_grid

    def get_first_frame_grids(self, file_path):
        """从 three_keypoints.txt 提取第一帧的网格坐标"""
        if not os.path.exists(file_path):
            return None
            
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # 寻找第一个 "帧 X:"
        start_grids = {}
        found_frame = False
        
        for i, line in enumerate(lines):
            if re.match(r"帧 \d+:", line):
                found_frame = True
                # 接下来的三行应该是坐标
                try:
                    root_line = lines[i+1]
                    thumb_line = lines[i+2]
                    index_line = lines[i+3]
                    
                    if "Thumb Root" in root_line:
                        start_grids['Root'] = eval(root_line.split(":")[-1].strip())
                    if "Thumb Tip" in thumb_line:
                        start_grids['Thumb'] = eval(thumb_line.split(":")[-1].strip())
                    if "Index Tip" in index_line:
                        start_grids['Index'] = eval(index_line.split(":")[-1].strip())
                        
                    return start_grids
                except:
                    pass
                break
                
        return None

    def construct_prompt(self, train_samples, test_init_grids, test_cup_grid=None, test_container_grid=None):
        """构建 Few-Shot Prompt"""
        prompt = "You are a high-level robot task planner.\n"
        prompt += "Task: Pouring Water (Grab bottle, Open bottle, Pour liquid, Close bottle).\n"
        prompt += "Given the initial hand keypoint grid coordinates (Root, Thumb, Index), the Object Cup location, and the Target Container location, predict the sequence of action phases required to complete the task.\n"
        prompt += "The grid resolution is 60x60x60 (values 0-59).\n"
        prompt += "For each phase, specify the phase name, start grids, and end grids.\n\n"
        
        prompt += "Reference Examples:\n"
        
        for sample in train_samples:
            prompt += f"Task Example:\n"
            if not sample['phases']: continue
            
            # Initial State (Start of Phase 0)
            init_state = sample['phases'][0]['start_grids']
            prompt += f"INITIAL STATE: {init_state}\n"
            if sample.get('cup_grid'):
                prompt += f"OBJECT CUP (Grab Target): {sample['cup_grid']}\n"
            if sample.get('container_grid'):
                prompt += f"TARGET CONTAINER (Pour Target): {sample['container_grid']}\n"
            
            prompt += f"PLAN:\n"
            
            for p in sample['phases']:
                prompt += f"  - Phase: {p['name']}\n"
                prompt += f"    Start: {p['start_grids']}\n"
                prompt += f"    End:   {p['end_grids']}\n"
            prompt += "\n"
            
        prompt += "-" * 30 + "\n"
        prompt += "NEW TASK TO PREDICT:\n"
        prompt += f"INITIAL STATE: {test_init_grids}\n"
        
        if test_cup_grid:
            prompt += f"OBJECT CUP (Grab Target): {test_cup_grid}\n"
        else:
            prompt += f"OBJECT CUP (Grab Target): [Unknown]\n"
            
        if test_container_grid:
            prompt += f"TARGET CONTAINER (Pour Target): {test_container_grid}\n"
        else:
            prompt += f"TARGET CONTAINER (Pour Target): [Unknown]\n"
            
        prompt += "PREDICTED PLAN (Please strictly follow the format):\n"
        
        return prompt

    def run_prediction(self, train_seqs, test_seq):
        print(f"🚀 Running Macro Phase Prediction for {test_seq}...")
        
        # 1. 加载训练数据
        train_samples = []
        for seq in train_seqs:
            path = os.path.join(self.root_dir, seq, "txt", "phase_boundary_data.txt")
            phases, cup_grid, cont_grid = self.parse_phase_boundary_data(path)
            if phases:
                train_samples.append({
                    'seq': seq, 
                    'phases': phases, 
                    'cup_grid': cup_grid,
                    'container_grid': cont_grid
                })
        print(f"  - Loaded {len(train_samples)} training sequences.")

        # 2. 加载测试集初始状态
        test_kpts_path = os.path.join(self.root_dir, test_seq, "txt", "three_keypoints.txt")
        test_init_grids = self.get_first_frame_grids(test_kpts_path)
        
        # 加载测试集目标物体位置
        test_boundary_path = os.path.join(self.root_dir, test_seq, "txt", "phase_boundary_data.txt")
        _, test_cup_grid, test_cont_grid = self.parse_phase_boundary_data(test_boundary_path)
        
        if not test_init_grids:
            print(f"  ❌ Failed to load initial grids for {test_seq}")
            return

        print(f"  - Test Initial State: {test_init_grids}")
        print(f"  - Test Cup Grid: {test_cup_grid}")
        print(f"  - Test Container Grid: {test_cont_grid}")

        # 3. 构建 Prompt
        prompt = self.construct_prompt(train_samples, test_init_grids, test_cup_grid, test_cont_grid)
        
        # 4. 调用 API
        try:
            resp = Generation.call(
                model='qwen-max',
                messages=[{'role': 'user', 'content': prompt}],
                result_format='message'
            )
            
            if resp.status_code == 200:
                prediction = resp.output.choices[0].message.content
                print("  ✅ Prediction received.")
                
                # 5. 保存结果
                output_path = os.path.join(self.root_dir, test_seq, "txt", "predicted_macro_phases.txt")
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(f"MACRO PHASE PREDICTION FOR {test_seq}\n")
                    f.write("="*50 + "\n\n")
                    # 显式写入初始状态，确保 cot_kat 能读取
                    f.write(f"**INITIAL STATE:** {test_init_grids}\n")
                    if test_cup_grid:
                         f.write(f"**OBJECT CUP GRID:** {test_cup_grid}\n")
                    if test_cont_grid:
                         f.write(f"**TARGET CONTAINER GRID:** {test_cont_grid}\n\n")
                    f.write(prediction)
                print(f"  - Saved to {output_path}")
                
            else:
                print(f"  ❌ API Error: {resp.code} - {resp.message}")
                
        except Exception as e:
            print(f"  ❌ Execution Error: {e}")

if __name__ == "__main__":
    ROOT_DIR = "assets/recordings"
    TRAIN_SEQS = ["seq_000", "seq_001", "seq_002", ] # 使用部分数据作为训练
    TEST_SEQ = "seq_004" # 预测 seq_004
    
    # 简单的自动发现逻辑 (可选)
    # all_seqs = sorted([d for d in os.listdir(ROOT_DIR) if d.startswith('seq_')])
    # TRAIN_SEQS = all_seqs[:-1]
    # TEST_SEQ = all_seqs[-1]
    
    predictor = MacroPhasePredictor(ROOT_DIR)
    predictor.run_prediction(TRAIN_SEQS, TEST_SEQ)

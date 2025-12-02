import os
import re

def simple_find_chinese(folder_path):
    """简化版：直接在控制台输出结果"""
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                matches = list(chinese_pattern.finditer(content))
                if matches:
                    print(f"\n=== {file_path} ===")
                    print(f"找到 {len(matches)} 个中文字符:")
                    
                    for i, match in enumerate(matches, 1):
                        # 计算行号和位置
                        lines_before = content[:match.start()].split('\n')
                        line_num = len(lines_before)
                        position = len(lines_before[-1]) + 1
                        
                        print(f"  {i}. 行{line_num}位{position}: '{match.group()}'")
                        
            except:
                continue

# 使用
simple_find_chinese("/root/code/vat")

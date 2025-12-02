import os
import re
import ast
import tokenize
from io import StringIO
import chardet

def remove_chinese_comments(file_path):
    """
    删除Python文件中的中文注释
    """
    try:
        # 检测文件编码
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            encoding = chardet.detect(raw_data)['encoding'] or 'utf-8'
        
        # 读取文件内容
        with open(file_path, 'r', encoding=encoding) as f:
            content = f.read()
        
        # 使用AST解析Python代码，精确识别注释
        try:
            tree = ast.parse(content)
        except SyntaxError:
            print(f"文件 {file_path} 存在语法错误，尝试使用正则表达式方法")
            return remove_chinese_comments_regex(file_path)
        
        # 获取所有注释的位置
        comments = []
        with open(file_path, 'rb') as f:
            tokens = tokenize.tokenize(f.readline)
            for token in tokens:
                if token.type == tokenize.COMMENT:
                    comments.append({
                        'start': token.start[1],  # 列号
                        'end': token.end[1],
                        'line': token.start[0] - 1,  # 行号（从0开始）
                        'text': token.string
                    })
        
        # 检查注释是否包含中文
        chinese_pattern = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]')
        comments_to_remove = []
        
        for comment in comments:
            if chinese_pattern.search(comment['text']):
                comments_to_remove.append(comment)
        
        if not comments_to_remove:
            return False  # 没有需要删除的中文注释
        
        # 按行处理内容
        lines = content.split('\n')
        modified = False
        
        # 从后往前删除，避免行号变化影响
        for comment in sorted(comments_to_remove, key=lambda x: x['line'], reverse=True):
            line_num = comment['line']
            if line_num < len(lines):
                line_content = lines[line_num]
                comment_text = comment['text'].strip()
                
                # 删除行内注释
                if '#' in line_content:
                    # 找到注释开始位置
                    comment_start = line_content.find('#')
                    comment_content = line_content[comment_start:]
                    
                    if chinese_pattern.search(comment_content):
                        # 如果整行都是注释，删除整行
                        if line_content.strip().startswith('#'):
                            del lines[line_num]
                        else:
                            # 只删除注释部分，保留代码
                            lines[line_num] = line_content[:comment_start].rstrip()
                        modified = True
        
        if modified:
            # 写入修改后的内容
            new_content = '\n'.join(lines)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
        
        return False
        
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return False

def remove_chinese_comments_regex(file_path):
    """
    使用正则表达式方法删除中文注释（备选方案）
    """
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            encoding = chardet.detect(raw_data)['encoding'] or 'utf-8'
        
        with open(file_path, 'r', encoding=encoding) as f:
            content = f.read()
        
        # 匹配中文注释的正则表达式
        # 单行注释：从#开始到行尾，且包含中文
        chinese_pattern = re.compile(r'#.*[\u4e00-\u9fff]+.*$', re.MULTILINE)
        
        # 多行字符串注释（docstring），谨慎处理
        # 这里我们只处理明显的多行注释模式
        multiline_comment_pattern = re.compile(r'(\'\'\'[\s\S]*?[\u4e00-\u9fff]+[\s\S]*?\'\'\'|\"\"\"[\s\S]*?[\u4e00-\u9fff]+[\s\S]*?\"\"\")')
        
        # 先备份原内容
        backup_content = content
        
        # 删除多行注释（谨慎使用）
        content = multiline_comment_pattern.sub('', content)
        
        # 删除单行中文注释
        def remove_chinese_comment(match):
            comment_text = match.group(0)
            # 检查是否真的包含中文
            if re.search(r'[\u4e00-\u9fff]', comment_text):
                return ''
            return comment_text
        
        content = chinese_pattern.sub(remove_chinese_comment, content)
        
        # 清理空行（可选）
        lines = content.split('\n')
        non_empty_lines = [line for line in lines if line.strip() != '']
        content = '\n'.join(non_empty_lines)
        
        if content != backup_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"正则表达式方法处理文件 {file_path} 时出错: {e}")
        return False

def process_python_files_in_folder(folder_path, backup=True):
    """
    处理文件夹中的所有Python文件
    
    Args:
        folder_path: 文件夹路径
        backup: 是否创建备份文件
    """
    python_files = []
    
    # 收集所有Python文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    print(f"找到 {len(python_files)} 个Python文件")
    
    processed_count = 0
    backup_dir = os.path.join(folder_path, 'backup_chinese_comments')
    
    if backup and not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    
    for file_path in python_files:
        if backup:
            # 创建备份
            backup_path = os.path.join(backup_dir, os.path.basename(file_path))
            with open(file_path, 'rb') as src, open(backup_path, 'wb') as dst:
                dst.write(src.read())
        
        try:
            if remove_chinese_comments(file_path):
                print(f"已处理: {file_path}")
                processed_count += 1
            else:
                print(f"无需处理: {file_path}（无中文注释）")
        except Exception as e:
            print(f"处理失败: {file_path} - 错误: {e}")
    
    print(f"\n处理完成！共处理了 {processed_count} 个文件")
    if backup:
        print(f"备份文件保存在: {backup_dir}")

# 使用示例
if __name__ == "__main__":
    folder_path = input("请输入要处理的文件夹路径: ").strip()
    
    if os.path.exists(folder_path):
        # 确认操作
        confirm = input("这将删除所有Python文件中的中文注释。是否继续？(y/n): ").strip().lower()
        if confirm == 'y':
            backup_choice = input("是否创建备份文件？(y/n): ").strip().lower()
            backup = backup_choice == 'y'
            
            process_python_files_in_folder(folder_path, backup)
        else:
            print("操作已取消")
    else:
        print("文件夹路径不存在！")
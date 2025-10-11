# 配置文件路径（需替换为实际路径）
input_path = "DrugBank.txt"    # 输入文件路径
output_path = "DrugBank1.txt"  # 输出文件路径

try:
    with open(input_path, 'r', encoding='utf-8') as f_in, open(output_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            parts = line.strip().split()  # 按空格分割内容
            if len(parts) > 2:
                f_out.write(' '.join(parts[2:]) + '\n')  # 保留第3部分及之后的内容
            else:
                f_out.write(line)  # 不足两部分时保留原行
    print(f"处理完成，结果已保存至：{output_path}")
except Exception as e:
    print(f"处理出错：{e}，请检查文件路径和编码")

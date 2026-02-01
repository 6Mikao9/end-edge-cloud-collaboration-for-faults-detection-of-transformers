# 修复脚本：清理 cloud_server.py 中的旧代码
import re

with open('cloud_server.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 找到新的 dashboard 函数结束位置（以 </html>\n    """ 结尾）
# 然后删除从 # ================= 到 # ================= 守护任务 ================= 之间的所有内容

pattern = r'(</html>\n    """)\n\n# =================.*?\n(            overflow: hidden;.*?)</html>\n    """\n\n(# ================= 守护任务 =================)'

replacement = r'\1\n\n\3'

new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

if new_content != content:
    with open('cloud_server.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("文件已修复！")
else:
    print("未找到需要修复的内容，可能文件已经是正确的格式。")

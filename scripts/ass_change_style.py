import os
from tqdm import tqdm
import concurrent.futures

ass_dir = '/hdd/srt11/data_8844/YouTube/ass'
style_block = """[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,微软雅黑,18,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,3,2,2,20,20,30,1
"""

ass_files = [f for f in os.listdir(ass_dir) if f.endswith('.ass')]

def process(i):
    fname = ass_files[i]
    path = os.path.join(ass_dir, fname)
    with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    with open(path, 'w', encoding='utf-8') as f:
        in_style = False
        for line in lines:
            if line.strip() == '[V4+ Styles]':
                in_style = True
                f.write(style_block + '\n')
            elif in_style and line.startswith('Format:'):
                continue  # 跳过原Format行
            elif in_style and line.startswith('Style:'):
                continue  # 跳过原Style行
            elif in_style and line.strip() == '':
                in_style = False
            elif not in_style:
                f.write(line)

with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
    responses = list(tqdm(executor.map(process, range(len(ass_files)))))

import re

with open('generate_research_figures.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Increase global rcParams sizes
content = re.sub(r'("font\.size"\s*:\s*)(\d+)', lambda m: m.group(1) + str(int(int(m.group(2))*1.6)), content)
content = re.sub(r'("axes\.titlesize"\s*:\s*)(\d+)', lambda m: m.group(1) + str(int(int(m.group(2))*1.6)), content)
content = re.sub(r'("axes\.labelsize"\s*:\s*)(\d+)', lambda m: m.group(1) + str(int(int(m.group(2))*1.6)), content)
content = re.sub(r'("xtick\.labelsize"\s*:\s*)(\d+)', lambda m: m.group(1) + str(int(int(m.group(2))*1.6)), content)
content = re.sub(r'("ytick\.labelsize"\s*:\s*)(\d+)', lambda m: m.group(1) + str(int(int(m.group(2))*1.6)), content)
content = re.sub(r'("legend\.fontsize"\s*:\s*)(\d+)', lambda m: m.group(1) + str(int(int(m.group(2))*1.6)), content)

# Increase explicitly defined fontsize and size attributes
content = re.sub(r'fontsize=(\d+(?:\.\d+)?)', lambda m: 'fontsize=' + str(int(float(m.group(1))*1.6)), content)
content = re.sub(r'labelsize=(\d+(?:\.\d+)?)', lambda m: 'labelsize=' + str(int(float(m.group(1))*1.6)), content)
content = re.sub(r'\'size\':\s*(\d+(?:\.\d+)?)', lambda m: "'size': " + str(int(float(m.group(1))*1.6)), content)

with open('generate_research_figures.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Successfully scaled up all font sizes by ~1.6x.')

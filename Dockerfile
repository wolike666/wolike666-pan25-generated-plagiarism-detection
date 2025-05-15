# 使用官方轻量 Python 3.12 镜像
FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 如果有依赖文件，可以先复制并安装；否则可省略下面两行
# COPY requirements.txt /app/
# RUN pip install --no-cache-dir -r requirements.txt

# 安装（可选）TIRA Python 客户端，如果在容器内需要调用 tira-cli
# RUN pip install --no-cache-dir 'git+https://github.com/tira-io/tira.git@main#egg=tira&subdirectory=python-client'

# 复制你的检测脚本到容器
COPY pan12-text-alignment-baseline.py /app/pan12-text-alignment-baseline.py

# 确保脚本可执行（可选）
RUN chmod +x /app/pan12-text-alignment-baseline.py

# 默认入口：运行脚本并接收两个参数：<输入数据目录> <输出目录>
ENTRYPOINT ["python", "/app/pan12-text-alignment-baseline.py"]

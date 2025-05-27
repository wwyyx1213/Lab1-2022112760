# Lab1 实验：基于大模型的编程与Git实战  可视化界面

import warnings

warnings.filterwarnings("ignore")
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import re
import random
import sys
from collections import defaultdict, deque
import matplotlib.pyplot as plt  # 可视化图表
import networkx as nx  # 有向图
from matplotlib.widgets import Button  # 按钮
import matplotlib.patches as patches  # 箭头
import numpy as np  # 数值计算
import tkinter as tk  # 图形界面
from tkinter import simpledialog, messagebox, filedialog, scrolledtext  # 对话框


# =========================
# 有向图类，包含图的构建、可视化、分析等功能
# =========================
class DirectedGraph:
    def __init__(self):
        # 使用嵌套defaultdict存储有向图，支持多重边
        self.graph = defaultdict(lambda: defaultdict(int))
        self.fig = None  # matplotlib图形对象
        self.ax = None  # matplotlib坐标轴对象
        self.pos = None  # 节点坐标布局
        self.G = None  # networkx的DiGraph对象
        self.dragging_node = None  # 当前被拖动的节点
        self.drag_start_pos = None  # 拖动起始位置
        self.node_size = 2000  # 节点大小

    def add_edge(self, from_node, to_node):
        # 添加一条有向边，支持多重边计数
        self.graph[from_node][to_node] += 1

    def get_nodes(self):
        # 获取所有节点
        return list(self.graph.keys())

    def get_edges(self):
        # 获取所有边及权重
        edges = []
        for u in self.graph:
            for v in self.graph[u]:
                edges.append((u, v, self.graph[u][v]))
        return edges

    def showDirectedGraph(self):
        # 构建networkx有向图对象
        self.G = nx.DiGraph()
        for u in self.graph:
            for v, w in self.graph[u].items():
                self.G.add_edge(u, v, weight=w)
        # 创建matplotlib窗口，设置较小尺寸
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.fig.subplots_adjust(bottom=0.12, top=0.92, left=0.06, right=0.98)
        # 使用spring_layout自动布局节点
        self.pos = nx.spring_layout(self.G, k=1, iterations=50)
        self.ax.clear()
        # 绘制节点
        self.node_collection = nx.draw_networkx_nodes(
            self.G,
            self.pos,
            node_color="lightblue",
            node_size=self.node_size,
            alpha=0.9,
            ax=self.ax,
        )
        # 绘制自定义边（箭头不被节点覆盖）
        self._draw_edges_with_arrows()
        # 绘制节点标签
        nx.draw_networkx_labels(
            self.G, self.pos, font_size=10, font_weight="bold", ax=self.ax
        )
        # 绘制边权重标签
        self._draw_edge_labels()
        # 设置标题
        self.ax.set_title("有向图可视化", fontsize=18, pad=18, fontweight="bold")
        plt.axis("off")
        # 绑定鼠标事件：拖动节点、缩放视图
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        # 添加保存按钮，放在底部中间
        save_ax = self.fig.add_axes([0.42, 0.01, 0.16, 0.055])
        save_button = Button(save_ax, "保存图形", color="#e0e0e0", hovercolor="#b0b0b0")
        save_button.on_clicked(self.save_graph)
        plt.show()

    def _draw_edges_with_arrows(self):
        # 手动绘制每条边，使箭头两端都不在节点圆形范围内
        node_radius = np.sqrt(self.node_size / np.pi) / 1000  # 经验缩放，适配layout
        for u, v, data in self.G.edges(data=True):
            x1, y1 = self.pos[u]
            x2, y2 = self.pos[v]
            dx, dy = x2 - x1, y2 - y1
            dist = np.hypot(dx, dy)
            if dist == 0:
                continue
            # 两端都缩进节点半径，保证箭头完全不与节点圆形重叠
            shrink_ratio = node_radius / dist
            start_x = x1 + dx * shrink_ratio
            start_y = y1 + dy * shrink_ratio
            end_x = x2 - dx * shrink_ratio
            end_y = y2 - dy * shrink_ratio
            # 再次缩进，保证更美观
            start_x = start_x + dx * shrink_ratio
            start_y = start_y + dy * shrink_ratio
            end_x = end_x - dx * shrink_ratio
            end_y = end_y - dy * shrink_ratio
            # 绘制箭头
            arrow = patches.FancyArrowPatch(
                (start_x, start_y),
                (end_x, end_y),
                arrowstyle="-|>",
                color="gray",
                mutation_scale=20,
                linewidth=1.5,
                connectionstyle="arc3,rad=0.1",
                zorder=1,
            )
            self.ax.add_patch(arrow)

    def _draw_edge_labels(self):
        # 绘制边权重标签
        edge_labels = nx.get_edge_attributes(self.G, "weight")
        for (u, v), label in edge_labels.items():
            x1, y1 = self.pos[u]
            x2, y2 = self.pos[v]
            label_x = (x1 + x2) / 2
            label_y = (y1 + y2) / 2
            self.ax.text(
                label_x,
                label_y,
                str(label),
                color="black",
                fontsize=9,
                ha="center",
                va="center",
                backgroundcolor="white",
            )

    def on_press(self, event):
        # 鼠标按下事件，判断是否点击了节点
        if event.inaxes != self.ax:
            return
        for node, (x, y) in self.pos.items():
            if abs(event.xdata - x) < 0.1 and abs(event.ydata - y) < 0.1:
                self.dragging_node = node
                self.drag_start_pos = (event.xdata, event.ydata)
                break

    def on_release(self, event):
        # 鼠标释放事件，结束拖动
        self.dragging_node = None
        self.drag_start_pos = None

    def on_motion(self, event):
        # 鼠标移动事件，拖动节点并重绘
        if self.dragging_node is None or event.inaxes != self.ax:
            return
        self.pos[self.dragging_node] = (event.xdata, event.ydata)
        self.ax.clear()
        self.node_collection = nx.draw_networkx_nodes(
            self.G,
            self.pos,
            node_color="lightblue",
            node_size=self.node_size,
            alpha=0.9,
            ax=self.ax,
        )
        self._draw_edges_with_arrows()
        nx.draw_networkx_labels(
            self.G, self.pos, font_size=10, font_weight="bold", ax=self.ax
        )
        self._draw_edge_labels()
        self.ax.set_title("有向图可视化", fontsize=18, pad=18, fontweight="bold")
        plt.axis("off")
        self.fig.canvas.draw()

    def on_scroll(self, event):
        # 鼠标滚轮缩放事件，动态缩放视图
        base_scale = 1.2
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        xdata = event.xdata  # 鼠标指针位置
        ydata = event.ydata
        if xdata is None or ydata is None:
            return
        if event.button == "up":
            scale_factor = 1 / base_scale
        elif event.button == "down":
            scale_factor = base_scale
        else:
            scale_factor = 1
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_xlim[0]) * scale_factor
        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_xlim[0])
        self.ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
        self.ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])
        self.fig.canvas.draw_idle()

    def save_graph(self, event):
        # 保存当前图形为图片
        try:
            plt.savefig("graph.png", format="png", dpi=300, bbox_inches="tight")
            print("图形已保存为 graph.png")
        except Exception as e:
            print(f"保存图片时出错: {e}")

    def queryBridgeWords(self, word1, word2):
        # 查询桥接词功能
        word1, word2 = word1.lower(), word2.lower()
        if word1 not in self.graph:
            return f"错误：单词 '{word1}' 不在图中"
        if word2 not in self.graph:
            return f"错误：单词 '{word2}' 不在图中"
        bridges = [w for w in self.graph[word1] if word2 in self.graph[w]]
        if not bridges:
            return f"在 '{word1}' 和 '{word2}' 之间没有桥接词"
        elif len(bridges) == 1:
            return f"从 '{word1}' 到 '{word2}' 的桥接词是: {bridges[0]}"
        else:
            return f"从 '{word1}' 到 '{word2}' 的桥接词有: {', '.join(bridges)}"

    def generateNewText(self, inputText):
        # 生成新文本，自动插入桥接词
        words = preprocess_text(inputText)
        if len(words) < 2:
            return "错误：输入文本太短，至少需要两个单词"
        new_text = []
        for i in range(len(words) - 1):
            new_text.append(words[i])
            bridges = [w for w in self.graph[words[i]] if words[i + 1] in self.graph[w]]
            if bridges:
                new_text.append(random.choice(bridges))
        new_text.append(words[-1])
        return " ".join(new_text)

    def calcShortestPath(self, word1, word2=None):
        # 计算最短路径，支持单个单词到所有其他单词的路径计算
        word1 = word1.lower()
        if word1 not in self.graph:
            return f"错误：起始单词 '{word1}' 不在图中"

        # 如果只输入一个单词，计算到所有其他单词的最短路径
        if word2 is None:
            all_paths = []
            dist = defaultdict(lambda: float("inf"))
            prev = {}
            dist[word1] = 0
            q = deque([word1])

            # BFS计算最短路径
            while q:
                u = q.popleft()
                for v in self.graph[u]:
                    if dist[u] + self.graph[u][v] < dist[v]:
                        dist[v] = dist[u] + self.graph[u][v]
                        prev[v] = u
                        q.append(v)

            # 收集所有可达节点的路径
            for target in self.graph.keys():
                if target != word1 and dist[target] != float("inf"):
                    path = []
                    cur = target
                    while cur != word1:
                        path.append(cur)
                        cur = prev[cur]
                    path.append(word1)
                    path.reverse()
                    all_paths.append((target, path, dist[target]))

            # 按路径长度排序
            all_paths.sort(key=lambda x: x[2])

            # 格式化输出
            if not all_paths:
                return f"从 '{word1}' 无法到达任何其他单词"

            result = [f"从 '{word1}' 出发的所有最短路径："]
            for target, path, distance in all_paths:
                result.append(f"\n到 '{target}':")
                result.append(f"路径: {' -> '.join(path)}")
                result.append(f"总权重: {distance:.2f}")
            return "\n".join(result)

        # 如果输入两个单词，计算它们之间的最短路径
        word2 = word2.lower()
        if word2 not in self.graph:
            return f"错误：目标单词 '{word2}' 不在图中"

        dist = defaultdict(lambda: float("inf"))
        prev = {}
        dist[word1] = 0
        q = deque([word1])

        while q:
            u = q.popleft()
            for v in self.graph[u]:
                if dist[u] + self.graph[u][v] < dist[v]:
                    dist[v] = dist[u] + self.graph[u][v]
                    prev[v] = u
                    q.append(v)

        if word2 not in dist or dist[word2] == float("inf"):
            return f"从 '{word1}' 到 '{word2}' 不存在路径"

        path = []
        cur = word2
        while cur != word1:
            path.append(cur)
            cur = prev[cur]
        path.append(word1)
        path.reverse()

        return f"从 '{word1}' 到 '{word2}' 的最短路径是:\n{' -> '.join(path)}\n总权重: {dist[word2]:.2f}"

    def calPageRank(self, d=0.85, max_iter=100, tol=1e-6):
        # 计算PageRank，处理悬挂节点（出度为0的节点）
        nodes = set(self.graph.keys()) | {v for u in self.graph for v in self.graph[u]}
        N = len(nodes)
        pr = dict.fromkeys(nodes, 1.0 / N)
        for _ in range(max_iter):
            new_pr = dict.fromkeys(nodes, (1 - d) / N)
            # 计算所有出度为0节点的PR总和
            dangling_sum = sum(pr[u] for u in nodes if len(self.graph[u]) == 0)
            for u in self.graph:
                out_sum = sum(self.graph[u].values())
                if out_sum > 0:
                    for v in self.graph[u]:
                        new_pr[v] += d * pr[u] * self.graph[u][v] / out_sum
            # 悬挂节点贡献均分给所有节点
            for v in nodes:
                new_pr[v] += d * dangling_sum / N
            if max(abs(new_pr[n] - pr[n]) for n in nodes) < tol:
                break
            pr = new_pr
        return pr

    def randomWalk(self):
        # 随机游走，结果写入文件
        if not self.graph:
            return "错误：图为空"
        visited = set()
        current = random.choice(list(self.graph.keys()))
        walk = [current]
        while True:
            if not self.graph[current]:
                break
            next_node = random.choice(list(self.graph[current].keys()))
            edge = (current, next_node)
            if edge in visited:
                break
            visited.add(edge)
            walk.append(next_node)
            current = next_node
        result = " ".join(walk)
        try:
            with open("random_walk.txt", "w", encoding="utf-8") as f:
                f.write(result)
        except Exception as e:
            print(f"警告：保存结果到文件时出错: {e}")
        return result

    def print_graph_cli(self):
        # 返回有向图结构字符串（用于界面展示）
        lines = []
        for u in self.graph:
            outs = []
            for v in self.graph[u]:
                outs.append(f"{u}→{v}")
            if outs:
                lines.append("    ".join(outs))
        return "有向图结构：\n" + "\n".join(lines)


# =========================
# 文本预处理与图加载
# =========================
def preprocess_text(text):
    # 文本预处理：去除非字母字符，转小写，分词
    if not isinstance(text, str):
        raise ValueError("输入必须是字符串类型")
    text = re.sub(r"[^A-Za-z\s]", " ", text)
    words = text.lower().split()
    return [w for w in words if w]  # 移除空字符串


def load_graph_from_file(filepath):
    # 从文件加载文本并构建有向图
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        if not text.strip():
            raise ValueError("文件为空")
        graph = DirectedGraph()
        words = preprocess_text(text)
        if len(words) < 2:
            raise ValueError("文本太短，无法构建图")
        for i in range(len(words) - 1):
            graph.add_edge(words[i], words[i + 1])
        return graph
    except FileNotFoundError:
        raise FileNotFoundError(f"找不到文件: {filepath}")
    except Exception as e:
        raise Exception(f"处理文件时出错: {str(e)}")


# =========================
# 主图形化界面
# =========================
def main():
    # 文件选择对话框
    root = tk.Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename(
        title="请选择输入文本文件",
        filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
    )
    if not filepath:
        messagebox.showerror("错误", "未选择文件，程序退出。")
        return
    try:
        graph = load_graph_from_file(filepath)
    except Exception as e:
        messagebox.showerror("加载文件出错", str(e))
        return
    # 主窗口
    app = tk.Tk()
    app.title("文本图形分析系统")
    app.minsize(700, 500)
    app.geometry("900x540")
    # 标题
    title_label = tk.Label(app, text="文本图形分析系统", font=("微软雅黑", 18, "bold"))
    title_label.pack(pady=(18, 8))
    # 主Frame分区
    main_frame = tk.Frame(app)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    # 结果显示区Frame（左侧）
    result_frame = tk.Frame(main_frame, bd=2, relief=tk.GROOVE)
    result_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=5)
    # 按钮区Frame（右侧）
    btn_frame = tk.Frame(main_frame)
    btn_frame.pack(side=tk.RIGHT, fill=tk.Y, expand=True, padx=(15, 0), pady=5)
    # 结果显示区
    result_text = scrolledtext.ScrolledText(
        result_frame, width=60, height=18, state="normal", font=("Consolas", 11)
    )
    result_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, graph.print_graph_cli())
    result_text.config(state="disabled")

    # 结果区显示函数
    def show_result(msg):
        result_text.config(state="normal")
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, msg)
        result_text.config(state="disabled")

    # 各功能按钮回调
    def btn_show_graph():
        graph.showDirectedGraph()

    def btn_query_bridge():
        w1 = simpledialog.askstring("查询桥接词", "请输入第一个单词:")
        w2 = simpledialog.askstring("查询桥接词", "请输入第二个单词:")
        if not w1 or not w2:
            messagebox.showerror("错误", "单词不能为空")
            return
        res = graph.queryBridgeWords(w1, w2)
        show_result(res)

    def btn_generate_text():
        text = simpledialog.askstring("生成新文本", "请输入要处理的文本:")
        if not text:
            messagebox.showerror("错误", "输入文本不能为空")
            return
        res = graph.generateNewText(text)
        show_result("生成的新文本:\n" + res)

    def btn_shortest_path():
        w1 = simpledialog.askstring("最短路径", "请输入起始单词:")
        if not w1:
            messagebox.showerror("错误", "起始单词不能为空")
            return
        w2 = simpledialog.askstring(
            "最短路径", "请输入目标单词（可选，留空则计算到所有单词的路径）:"
        )
        res = graph.calcShortestPath(w1, w2 if w2 else None)
        show_result(res)

    def btn_pagerank():
        pr = graph.calPageRank()
        res = "PageRank结果 (按重要度降序排列):\n"
        for k, v in sorted(pr.items(), key=lambda x: -x[1])[:10]:
            res += f"{k}: {v:.4f}\n"
        show_result(res)

    def btn_random_walk():
        result = graph.randomWalk()
        show_result("随机游走结果:\n" + result + "\n\n结果已保存到 random_walk.txt")

    def btn_exit():
        app.quit()
        sys.exit(0)

    # 关闭窗口时直接退出程序
    app.protocol("WM_DELETE_WINDOW", btn_exit)
    # 按钮区美化，自适应布局
    btns = [
        ("显示有向图", btn_show_graph),
        ("查询桥接词", btn_query_bridge),
        ("生成新文本", btn_generate_text),
        ("计算最短路径", btn_shortest_path),
        ("计算节点重要度(PageRank)", btn_pagerank),
        ("随机游走", btn_random_walk),
        ("退出程序", btn_exit),
    ]
    for text, cmd in btns:
        btn = tk.Button(btn_frame, text=text, font=("微软雅黑", 12), command=cmd)
        btn.pack(fill=tk.BOTH, expand=True, pady=8, padx=2)
    app.mainloop()


# 程序入口
if __name__ == "__main__":
    main()

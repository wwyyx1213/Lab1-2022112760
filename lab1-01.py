def showDirectedGraph(self):
        G = nx.DiGraph()
        for u in self.graph:
            for v, w in self.graph[u].items():
                G.add_edge(u, v, weight=w)
        
        # 在命令行界面显示有向图
        self.showDirectedGraphCLI()
        
        # 创建图形窗口
        plt.figure(figsize=(12, 8))
        
        # 使用spring_layout初始化节点位置
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # 创建交互式图形
        fig = plt.gcf()
        ax = plt.gca()
        
        # 绘制节点
        nodes = nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                                     node_size=2000, alpha=0.9)
        
        # 绘制边和箭头
        edges = nx.draw_networkx_edges(G, pos, edge_color='gray',
                                     arrowsize=20, width=1.5,
                                     connectionstyle='arc3,rad=0.1')  # 添加弧度避免箭头重叠
        
        # 添加标签
        labels = nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        
        # 添加边权重标签
        edge_labels = nx.get_edge_attributes(G, 'weight')
        edge_label_pos = {}
        for (u, v), w in edge_labels.items():
            # 计算边的中点位置
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            # 调整标签位置，使其不覆盖箭头
            edge_label_pos[(u, v)] = ((x1 + x2) / 2, (y1 + y2) / 2)
        
        edge_labels_obj = nx.draw_networkx_edge_labels(G, edge_label_pos,
                                                     edge_labels=edge_labels,
                                                     font_size=8)
        
        # 设置标题和其他属性
        plt.title("交互式有向图 (可拖动节点)", fontsize=16, pad=20)
        plt.axis('off')
        
        # 添加交互式功能
        def on_press(event):
            if event.inaxes != ax:
                return
            for node, (x, y) in pos.items():
                if abs(event.xdata - x) < 0.1 and abs(event.ydata - y) < 0.1:
                    ax._drag_node = node
                    break
        
        def on_motion(event):
            if event.inaxes != ax or not hasattr(ax, '_drag_node'):
                return
            node = ax._drag_node
            pos[node] = (event.xdata, event.ydata)
            
            # 更新图形
            plt.cla()
            nodes = nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                                         node_size=2000, alpha=0.9)
            edges = nx.draw_networkx_edges(G, pos, edge_color='gray',
                                         arrowsize=20, width=1.5,
                                         connectionstyle='arc3,rad=0.1')
            labels = nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
            
            # 更新边权重标签位置
            edge_label_pos = {}
            for (u, v), w in edge_labels.items():
                x1, y1 = pos[u]
                x2, y2 = pos[v]
                edge_label_pos[(u, v)] = ((x1 + x2) / 2, (y1 + y2) / 2)
            
            edge_labels_obj = nx.draw_networkx_edge_labels(G, edge_label_pos,
                                                         edge_labels=edge_labels,
                                                         font_size=8)
            
            plt.title("交互式有向图 (可拖动节点)", fontsize=16, pad=20)
            plt.axis('off')
            fig.canvas.draw_idle()
        
        def on_release(event):
            if hasattr(ax, '_drag_node'):
                del ax._drag_node
        
        # 连接事件处理器
        fig.canvas.mpl_connect('button_press_event', on_press)
        fig.canvas.mpl_connect('motion_notify_event', on_motion)
        fig.canvas.mpl_connect('button_release_event', on_release)
        
        # 保存图片
        try:
            plt.savefig("graph.png", format="png", dpi=300, bbox_inches='tight')
            print("图形已保存为 graph.png")
        except Exception as e:
            print(f"保存图片时出错: {e}")
        
        # 显示图形
        plt.show()
        plt.close()

    def showDirectedGraphCLI(self):
        """在命令行界面显示有向图的结构"""
        print("\n=== 有向图结构 ===")
        
        # 获取所有节点
        nodes = sorted(self.graph.keys())
        
        # 为每个节点创建其出边的表示
        for node in nodes:
            if node in self.graph and self.graph[node]:
                # 获取当前节点的所有出边
                edges = []
                for target, weight in self.graph[node].items():
                    edge = f"{node}→{target}"
                    if weight > 1:
                        edge += f"({weight})"
                    edges.append(edge)
                
                # 打印当前节点的所有出边
                print("  ".join(edges))
        
        print("=" * 20)
        print("说明：")
        print("- 箭头(→)表示边的方向")
        print("- 括号中的数字表示边的权重")
        print("=" * 20)
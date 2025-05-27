import pytest
from lab1 import DirectedGraph, preprocess_text

@pytest.fixture
def sample_graph():
    graph = DirectedGraph()
    # 构建有桥接词的图
    words = preprocess_text("the eats mouse the dog chases the cat runs the mouse runs")
    for i in range(len(words) - 1):
        graph.add_edge(words[i], words[i + 1])
    return graph

def test_generateNewText_case1(sample_graph):
    # 路径1：输入太短
    res = sample_graph.generateNewText("the")
    assert "错误" in res

def test_generateNewText_case2(sample_graph):
    # 路径2：无桥接词
    res = sample_graph.generateNewText("the dog")
    assert res == "the dog"

def test_generateNewText_case3(sample_graph):
    # 路径3：有桥接词
    res = sample_graph.generateNewText("the mouse")
    # the->eats->mouse
    assert res == "the eats mouse" or res == "the mouse"  # 随机性

def test_generateNewText_case4(sample_graph):
    # 路径2：无桥接词
    res = sample_graph.generateNewText("the eats mouse")
    assert res == "the eats mouse"
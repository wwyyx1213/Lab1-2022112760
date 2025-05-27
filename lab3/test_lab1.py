import pytest
from lab1 import DirectedGraph, preprocess_text

@pytest.fixture
def sample_graph():
    graph = DirectedGraph()
    # 新测试数据，确保有桥接词
    words = preprocess_text("the eats mouse the dog chases the cat runs the mouse runs")
    for i in range(len(words) - 1):
        graph.add_edge(words[i], words[i + 1])
    return graph


def test_case1(sample_graph):
    """the->mouse，桥接词为eats"""
    res = sample_graph.queryBridgeWords("the", "mouse")
    assert "eats" in res and "桥接词" in res

def test_case2(sample_graph):
    """the->dog，无桥接词（直接相连）"""
    res = sample_graph.queryBridgeWords("the", "dog")
    assert "没有桥接词" in res

def test_case3(sample_graph):
    """the->runs，桥接词为cat,mouse """
    res = sample_graph.queryBridgeWords("the", "runs")
    assert "cat" in res and "mouse" in res and "桥接词" in res

def test_case4(sample_graph):
    """word1不在图中"""
    res = sample_graph.queryBridgeWords("notexist", "cat")
    assert "不在图中" in res and "notexist" in res and "桥接词" not in res and "cat" not in res

def test_case5(sample_graph):
    """word2不在图中"""
    res = sample_graph.queryBridgeWords("the", "notexist")
    assert "不在图中" in res and "notexist" in res and "桥接词" not in res 

def test_case6(sample_graph):
    """word1、word2均不在图中"""
    res = sample_graph.queryBridgeWords("notexist1", "notexist2")
    assert "不在图中" in res and "notexist1" in res and "桥接词" not in res 

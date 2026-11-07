"""Тема 6. Графи: модель транспортної мережі, DFS/BFS, Дейкстра."""

import heapq
from collections import deque

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

# ---------- Завдання 1: модель мережі (спрощена трамвайна мережа Львова) ----------
# Ваги - приблизний час у дорозі між зупинками, хв.
EDGES = [
    ("Вокзал", "Підвальна", 6),
    ("Вокзал", "Сихів", 18),
    ("Вокзал", "Левандівка", 9),
    ("Підвальна", "Площа Ринок", 3),
    ("Площа Ринок", "Оперний театр", 4),
    ("Оперний театр", "Політехніка", 5),
    ("Оперний театр", "Личаків", 8),
    ("Політехніка", "Стрийський парк", 6),
    ("Стрийський парк", "Сихів", 12),
    ("Личаків", "Погулянка", 4),
    ("Підвальна", "Личаків", 7),
    ("Левандівка", "Рясне", 10),
    ("Політехніка", "Сихів", 15),
]

G = nx.Graph()
G.add_weighted_edges_from(EDGES)


def analyze(graph: nx.Graph) -> None:
    print("=== Завдання 1: аналіз графа ===")
    print(f"Вершин: {graph.number_of_nodes()}")
    print(f"Ребер:  {graph.number_of_edges()}")
    print(f"Зв'язний: {nx.is_connected(graph)}")
    print(f"Середній ступінь: {sum(d for _, d in graph.degree()) / graph.number_of_nodes():.2f}")
    print("Ступені вершин:")
    for node, deg in sorted(graph.degree(), key=lambda x: -x[1]):
        print(f"  {node}: {deg}")


def draw(graph: nx.Graph, path: str = "topic6_graph.png") -> None:
    pos = nx.spring_layout(graph, seed=42)
    plt.figure(figsize=(10, 7))
    nx.draw(graph, pos, with_labels=True, node_color="skyblue", node_size=1800, font_size=9)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=nx.get_edge_attributes(graph, "weight"))
    plt.title("Трамвайна мережа (спрощена модель)")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Граф збережено у {path}")


# ---------- Завдання 2: DFS і BFS ----------
def dfs_path(graph: nx.Graph, start: str, goal: str) -> list[str] | None:
    stack = [(start, [start])]
    visited = set()
    while stack:
        node, path = stack.pop()
        if node == goal:
            return path
        if node in visited:
            continue
        visited.add(node)
        # reversed, щоб обхід ішов у порядку сусідів, як у рекурсивному DFS
        for neighbor in reversed(list(graph.neighbors(node))):
            if neighbor not in visited:
                stack.append((neighbor, path + [neighbor]))
    return None


def bfs_path(graph: nx.Graph, start: str, goal: str) -> list[str] | None:
    queue = deque([(start, [start])])
    visited = {start}
    while queue:
        node, path = queue.popleft()
        if node == goal:
            return path
        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return None


def compare_dfs_bfs(graph: nx.Graph, start: str, goal: str) -> None:
    print(f"\n=== Завдання 2: шлях {start} -> {goal} ===")
    d = dfs_path(graph, start, goal)
    b = bfs_path(graph, start, goal)
    print(f"DFS ({len(d) - 1} переходів): {' -> '.join(d)}")
    print(f"BFS ({len(b) - 1} переходів): {' -> '.join(b)}")
    print(
        "Пояснення: DFS заходить якомога глибше по першому ж сусіду і повертається лише "
        "у глухому куті, тому знаходить якийсь шлях, але не обов'язково найкоротший. "
        "BFS обходить граф рівнями (спочатку всі сусіди, потім сусіди сусідів), тому "
        "перший знайдений шлях має мінімальну кількість ребер."
    )


# ---------- Завдання 3: Дейкстра ----------
def dijkstra(graph: nx.Graph, start: str) -> tuple[dict, dict]:
    distances = {node: float("inf") for node in graph.nodes}
    previous = {node: None for node in graph.nodes}
    distances[start] = 0
    heap = [(0, start)]
    while heap:
        dist, node = heapq.heappop(heap)
        if dist > distances[node]:
            continue
        for neighbor, attrs in graph[node].items():
            new_dist = dist + attrs["weight"]
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                previous[neighbor] = node
                heapq.heappush(heap, (new_dist, neighbor))
    return distances, previous


def restore_path(previous: dict, target: str) -> list[str]:
    path = []
    while target is not None:
        path.append(target)
        target = previous[target]
    return path[::-1]


def all_shortest_paths(graph: nx.Graph) -> None:
    print("\n=== Завдання 3: найкоротші шляхи (Дейкстра) ===")
    nodes = list(graph.nodes)
    for i, src in enumerate(nodes):
        distances, previous = dijkstra(graph, src)
        for dst in nodes[i + 1:]:
            path = restore_path(previous, dst)
            # звірка з вбудованою реалізацією networkx
            assert distances[dst] == nx.dijkstra_path_length(graph, src, dst)
            print(f"{src} -> {dst}: {distances[dst]} хв | {' -> '.join(path)}")


if __name__ == "__main__":
    analyze(G)
    draw(G)
    compare_dfs_bfs(G, "Вокзал", "Сихів")
    compare_dfs_bfs(G, "Рясне", "Погулянка")
    all_shortest_paths(G)

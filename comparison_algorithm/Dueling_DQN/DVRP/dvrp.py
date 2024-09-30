from itertools import permutations


def calculate_distance(point1, point2):
    # 计算两个点之间的欧氏距离
    return ((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2) ** 0.5


def calculate_total_distance(route, points):
    # 计算路径的总距离
    total_distance = 0
    for i in range(len(route) - 1):
        point1 = points[route[i]]
        point2 = points[route[i + 1]]
        total_distance += calculate_distance(point1, point2)
    return total_distance


def dvrp(points):
    num_points = len(points)
    best_route = None
    best_distance = float('inf')

    # 生成所有点的排列组合
    all_routes = permutations(range(1, num_points))

    for route in all_routes:
        # 在路径的开头和结尾添加起始点和结束点
        route = [0] + list(route) + [0]
        total_distance = calculate_total_distance(route, points)

        # 更新最优路径
        if total_distance < best_distance:
            best_distance = total_distance
            best_route = route

    return best_route, best_distance


# 示例点集
points = [(0, 0), (1, 2), (3, 5), (7, 1), (9, 4)]

best_route, best_distance = dvrp(points)
print("Best route:", best_route)
print("Best distance:", best_distance)

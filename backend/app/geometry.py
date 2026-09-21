def point_in_rect(px, py, x1, y1, x2, y2):
    return x1 <= px <= x2 and y1 <= py <= y2


def point_in_polygon(px, py, points):
    inside = False
    n = len(points)
    x1, y1 = points[-1]
    for i in range(n):
        x2, y2 = points[i]
        if (y1 > py) != (y2 > py):
            x_intersect = (x2 - x1) * (py - y1) / (y2 - y1) + x1
            if px < x_intersect:
                inside = not inside
        x1, y1 = x2, y2
    return inside

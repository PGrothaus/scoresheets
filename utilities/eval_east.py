import shapely.geometry


def create_polygon(x1, y1, x2, y2, x3, y3, x4, y4):
    p1 = x1, y1
    p2 = x2, y2
    p3 = x3, y3
    p4 = x4, y4
    return shapely.geometry.Polygon([p1, p2, p3, p4])


def get_iou_for_two_polygons(p1, p2):
    return p1.intersection(p2).area / p1.union(p2).area


def read_polygon_file(fp):
    with open(fp, 'r') as f:
        lines = f.readlines()

    polygon_coordinates = []
    for line in lines:
        line = line.strip('\n').strip('\r')
        elements = line.split(',')
        coordinates = elements[:8]
        coordinates = map(int, coordinates)
        polygon_coordinates.append(coordinates)

    return [create_polygon(*c) for c in polygon_coordinates]


def read_result_and_model_polygons(fp_testset, fp_model):
    polygons_test = read_polygon_file(fp_testset)
    polygons_model = read_polygon_file(fp_model)
    return polygons_model, polygons_test


def process_model_polygon(polygon, polygons_test, thr=0.8):
    ious = [get_iou_for_two_polygons(polygon, p0) for p0 in polygons_test]
    max_iou = max(ious)
    print(max_iou)
    if max_iou >= thr:
        return 1, ious.index(max_iou)
    else:
        return 0, None


def analyse_polygons(polygons_model, polygons_test, thr=0.8):
    pos_det_count = 0
    detected_polyons = []
    for p0 in polygons_model:
        pos_det, index = process_model_polygon(p0, polygons_test, thr=thr)
        pos_det_count += pos_det
        if index is not None:
            detected_polyons.append(index)

    print(pos_det_count, len(set(detected_polyons)), thr)
    precision = 1. * pos_det_count / len(polygons_model)
    recall = 1. * len(set(detected_polyons)) / len(polygons_test)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)

    return precision, recall, f1


def analyse_test_example(fp_result_file, fp_ground_truth, thr=0.8):
    polygons_model, polygons_test = read_result_and_model_polygons(
        fp_ground_truth, fp_result_file
    )
    p, r, f1 = analyse_polygons(polygons_model, polygons_test, thr=thr)
    print("precision: {}\nrecall: {}\nf1: {}\n".format(p, r, f1))


if "__main__" == __name__:
    import sys
    fp_result_file = sys.argv[1]
    fp_ground_truth = sys.argv[2]
    thr = float(sys.argv[3])

    analyse_test_example(fp_result_file, fp_ground_truth, thr=thr)

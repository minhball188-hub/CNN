# file: import_mesh_to_flux.py
# Import lưới từ file .TRA vào FLUX và gán vật liệu

from flux_connector import FluxConnection

# Tolerance để so sánh tọa độ (mm)
COORD_TOLERANCE = 0.05  # Tăng lên 0.05mm để match tốt hơn

# Bán kính trong và ngoài của rotor (mm)
R_IN = 16.0
R_OUT = 39.0
R_TOLERANCE = 0.1  # Tolerance để check điểm nằm trên đường tròn

def is_on_rin(x, y):
    """Check xem điểm (x,y) có nằm trên đường tròn Rin không"""
    r_squared = x*x + y*y
    rin_squared = R_IN * R_IN  # 256
    return abs(r_squared - rin_squared) < R_TOLERANCE * R_IN * 2

def is_on_rout(x, y):
    """Check xem điểm (x,y) có nằm trên đường tròn Rout không"""
    r_squared = x*x + y*y
    rout_squared = R_OUT * R_OUT  # 1521
    return abs(r_squared - rout_squared) < R_TOLERANCE * R_OUT * 2

def is_on_boundary_circle(x, y):
    """Check xem điểm có nằm trên Rin hoặc Rout không"""
    return is_on_rin(x, y) or is_on_rout(x, y)

# Số lượng points đã có trong project gốc (0 đến 146 = 147 points)
# FLUX sẽ tự động gán ID mới bắt đầu từ 147
EXISTING_POINT_COUNT = 147

# Các point biên đã có sẵn trong project gốc (hardcode vì FLUX không export được)
# Format: {(x, y): point_id}
BOUNDARY_POINTS = {
    (0.0, 39.0): 7,
    (11.62, 37.23): 6,      # (11.6229, 37.2278)
    (27.58, 27.58): 10,     # (27.5772, 27.5772)
    (37.23, 11.62): 33,     # (37.2278, 11.6229)
    (39.0, 0.0): 34,
    (0.0, 16.0): 24,
    (11.31, 11.31): 5,      # (11.3137, 11.3137)
    (16.0, 0.0): 23,
    (7.51, 28.72): 4,       # (7.50806, 28.7213)
    (20.24, 20.24): 9,      # (20.236, 20.236)
    (30.84, 9.63): 2,       # (30.8426, 9.62938)
    (9.63, 30.84): 1,       # (9.62938, 30.8426)
    (18.11, 18.11): 8,      # (18.1147, 18.1147)
    (28.72, 7.51): 3,       # (28.7213, 7.50806)
}

def normalize_coord(value):
    """Normalize các giá trị rất nhỏ về 0"""
    if abs(value) < COORD_TOLERANCE:
        return 0.0
    return value

def find_matching_point(x, y, existing_coords, tolerance=COORD_TOLERANCE):
    """Tìm point có tọa độ gần với (x, y) trong tolerance"""
    for (ex, ey), point_id in existing_coords.items():
        if abs(x - ex) < tolerance and abs(y - ey) < tolerance:
            return point_id
    return None

def parse_tra_file(tra_path):
    """Đọc file .TRA và trích xuất nodes và elements"""
    nodes = {}  # {node_id: (x, y)}
    elements = []  # [(n1, n2, n3), ...] - chỉ lấy 3 đỉnh của tam giác

    with open(tra_path, 'r') as f:
        lines = f.readlines()

    # Tìm số nodes và elements
    num_nodes = 0
    num_elements = 0
    for line in lines:
        if 'Number of nodes' in line:
            num_nodes = int(line.split()[0])
        if 'Number of surface elements' in line:
            num_elements = int(line.split()[0])

    print(f"Số nodes: {num_nodes}")
    print(f"Số elements: {num_elements}")

    # Tìm vị trí bắt đầu của tọa độ nodes
    node_start = 0
    for i, line in enumerate(lines):
        if 'Coordinates of the nodes' in line:
            node_start = i + 1
            break

    # Đọc tọa độ nodes
    for i in range(node_start, node_start + num_nodes):
        parts = lines[i].split()
        node_id = int(parts[0])
        x = float(parts[1])
        y = float(parts[2])
        # Chuyển từ m sang mm và giữ 2 chữ số thập phân
        x_mm = round(x * 1000, 2)
        y_mm = round(y * 1000, 2)
        # Normalize các giá trị rất nhỏ về 0
        x_mm = normalize_coord(x_mm)
        y_mm = normalize_coord(y_mm)
        nodes[node_id] = (x_mm, y_mm)

    # Tìm vị trí bắt đầu của elements
    elem_start = 0
    for i, line in enumerate(lines):
        if 'Description of elements' in line:
            elem_start = i + 1
            break

    # Đọc elements (mỗi element có 2 dòng: header và node indices)
    for i in range(num_elements):
        idx = elem_start + i * 2 + 1  # Dòng chứa node indices
        parts = lines[idx].split()
        # 6-node triangle: lấy 3 đỉnh đầu (corner nodes)
        n1, n2, n3 = int(parts[0]), int(parts[1]), int(parts[2])
        elements.append((n1, n2, n3))

    return nodes, elements


def main():
    tra_path = r"D:\Minh Tuan\Projects\VS Code\CNN_test\Day 3\rotor_3.TRA"
    flu_path = r"D:/Minh Tuan/Projects/VS Code/CNN_test/Day 3/IPM_mesh.FLU"
    output_path = r"D:/Minh Tuan/Projects/VS Code/CNN_test/Day 3/IPM_mesh_with_rotor.FLU"
    log_path = r"D:\Minh Tuan\Projects\VS Code\CNN_test\Day 3\import_log.txt"

    # Mở file log
    log = open(log_path, 'w', encoding='utf-8')
    def log_print(msg):
        print(msg)
        log.write(msg + '\n')
        log.flush()

    log_print("--- BẮT ĐẦU IMPORT LƯỚI VÀO FLUX ---")

    # Parse file TRA
    log_print("Đang đọc file TRA...")
    nodes, elements = parse_tra_file(tra_path)
    log_print(f"Đã đọc {len(nodes)} nodes và {len(elements)} elements")

    # In ra vài node đầu để kiểm tra
    log_print("\n--- Kiểm tra 5 nodes đầu tiên ---")
    for i, node_id in enumerate(sorted(nodes.keys())[:5]):
        x, y = nodes[node_id]
        log_print(f"  Node {node_id}: ({x}, {y}) mm")

    # In ra vài element đầu để kiểm tra
    log_print("\n--- Kiểm tra 5 elements đầu tiên ---")
    for i, (n1, n2, n3) in enumerate(elements[:5]):
        log_print(f"  Element {i+1}: nodes ({n1}, {n2}, {n3})")

    # Face bắt đầu từ đâu (Point ID sẽ được tự động xác định)
    START_FACE_ID = 46

    try:
        with FluxConnection() as flux:
            flux.start(mode='FLUX2D')

            # Load project
            log_print(f"\n>>> Đang load project: {flu_path}")
            flux.run_command(f'loadProject("{flu_path}")')
            log_print(">>> Load xong!")

            # Query các point hiện có trong project
            log_print("\n>>> Đang đọc các point hiện có trong project...")
            existing_points_file = r"D:/Minh Tuan/Projects/VS Code/CNN_test/Day 3/existing_points.txt"

            # Dùng 1 lệnh duy nhất - list comprehension + join
            flux.run_command(f"open('{existing_points_file}', 'w').write('\\n'.join([str(p.id) + ' ' + str(round(p.coordinate[0], 2)) + ' ' + str(round(p.coordinate[1], 2)) for p in Point.getAll()]))")

            # Đọc file và tạo dict existing_coords
            existing_coords = {}  # {(x, y): point_id} - các point đã có trong project
            import time
            time.sleep(0.5)  # Đợi file được ghi xong
            try:
                with open(existing_points_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            point_id = int(parts[0])
                            x = normalize_coord(float(parts[1]))
                            y = normalize_coord(float(parts[2]))
                            existing_coords[(x, y)] = point_id
                log_print(f"    Đã đọc {len(existing_coords)} points từ FLUX")
            except Exception as e:
                log_print(f"    Lỗi đọc existing points: {e}")

            # Merge BOUNDARY_POINTS vào existing_coords (ưu tiên BOUNDARY_POINTS)
            log_print(f"    Thêm {len(BOUNDARY_POINTS)} boundary points từ hardcode")
            for coord, point_id in BOUNDARY_POINTS.items():
                existing_coords[coord] = point_id
            log_print(f"    Tổng cộng: {len(existing_coords)} points đã biết")

            # Log các boundary points
            log_print("\n--- Boundary points (hardcode) ---")
            for (x, y), pid in BOUNDARY_POINTS.items():
                log_print(f"    Point[{pid}]: ({x}, {y})")

            # FLUX sẽ tự động gán ID mới bắt đầu từ EXISTING_POINT_COUNT
            # Project gốc có points 0-146, nên point mới sẽ là 147, 148, ...
            current_point_id = EXISTING_POINT_COUNT
            log_print(f"    Project gốc có {EXISTING_POINT_COUNT} points (0-{EXISTING_POINT_COUNT-1})")
            log_print(f"    FLUX sẽ tạo point mới bắt đầu từ Point[{current_point_id}]")

            # Tạo mapping unique points
            unique_coords = dict(existing_coords)  # Bắt đầu với các point đã có
            node_to_point = {}  # {old_node_id: new_point_id}
            duplicate_count = 0
            existing_reuse_count = 0

            log_print(f"\n>>> Đang tạo {len(nodes)} điểm...")

            # DEBUG: In ra 15 node đầu tiên để verify
            log_print("\n--- DEBUG: 15 nodes đầu tiên từ TRA file ---")
            for node_id in sorted(nodes.keys())[:15]:
                x, y = nodes[node_id]
                log_print(f"    Node {node_id}: ({x}, {y})")

            new_points_created = 0
            for node_id in sorted(nodes.keys()):
                x, y = nodes[node_id]
                coord_key = (x, y)

                # Bước 1: Check exact match trong unique_coords
                if coord_key in unique_coords:
                    existing_point = unique_coords[coord_key]
                    node_to_point[node_id] = existing_point

                    if coord_key in existing_coords:
                        existing_reuse_count += 1
                        if existing_reuse_count <= 20:
                            log_print(f"    REUSE (exact): Node {node_id} ({x}, {y}) -> Point[{existing_point}]")
                    else:
                        duplicate_count += 1
                        if duplicate_count <= 20:
                            log_print(f"    DUPLICATE (TRA): Node {node_id} ({x}, {y}) -> Point[{existing_point}]")
                else:
                    # Bước 2: Check tolerance-based match trong existing_coords (project gốc)
                    matched_point = find_matching_point(x, y, existing_coords)

                    if matched_point is not None:
                        # Tìm thấy point gần trong project gốc
                        node_to_point[node_id] = matched_point
                        unique_coords[coord_key] = matched_point  # Cache để các node sau có thể dùng
                        existing_reuse_count += 1
                        if existing_reuse_count <= 20:
                            log_print(f"    REUSE (tolerance): Node {node_id} ({x}, {y}) -> Point[{matched_point}]")
                    else:
                        # Bước 3: Check tolerance-based match trong unique_coords (TRA đã tạo)
                        matched_new = find_matching_point(x, y, unique_coords)

                        if matched_new is not None:
                            node_to_point[node_id] = matched_new
                            duplicate_count += 1
                            if duplicate_count <= 20:
                                log_print(f"    DUPLICATE (tolerance): Node {node_id} ({x}, {y}) -> Point[{matched_new}]")
                        else:
                            # Tạo điểm mới
                            cmd = f"PointCoordinates(coordSys=CoordSys['XY1'], uvw=['{x}', '{y}'], nature=Nature['STANDARD'])"
                            flux.run_command(cmd)

                            # DEBUG: Log 15 points đầu tiên
                            if new_points_created < 15:
                                log_print(f"    TẠO: Node {node_id} ({x}, {y}) -> Point[{current_point_id}]")

                            unique_coords[coord_key] = current_point_id
                            node_to_point[node_id] = current_point_id
                            current_point_id += 1
                            new_points_created += 1

            log_print(f">>> Đã tạo {new_points_created} điểm mới")
            log_print(f">>> Dùng lại {existing_reuse_count} điểm từ project gốc")
            log_print(f">>> Bỏ qua {duplicate_count} điểm trùng trong TRA")
            log_print(f">>> Points mới từ {EXISTING_POINT_COUNT} đến {current_point_id - 1}")

            # Tạo dict lưu tọa độ của mỗi point để check boundary
            point_coords = {}  # {point_id: (x, y)}
            for node_id, (x, y) in nodes.items():
                point_id = node_to_point[node_id]
                point_coords[point_id] = (x, y)

            # DEBUG: Log các điểm nằm trên Rin và Rout
            log_print("\n--- DEBUG: Các điểm nằm trên Rin (R=16mm) ---")
            rin_points = [(pid, x, y) for pid, (x, y) in point_coords.items() if is_on_rin(x, y)]
            for pid, x, y in rin_points[:10]:
                log_print(f"    Point[{pid}]: ({x}, {y}) - R²={x*x+y*y:.2f}")
            if len(rin_points) > 10:
                log_print(f"    ... và {len(rin_points) - 10} điểm khác")

            log_print("\n--- DEBUG: Các điểm nằm trên Rout (R=39mm) ---")
            rout_points = [(pid, x, y) for pid, (x, y) in point_coords.items() if is_on_rout(x, y)]
            for pid, x, y in rout_points[:10]:
                log_print(f"    Point[{pid}]: ({x}, {y}) - R²={x*x+y*y:.2f}")
            if len(rout_points) > 10:
                log_print(f"    ... và {len(rout_points) - 10} điểm khác")

            # Tạo lines cho mỗi tam giác
            log_print(f"\n>>> Đang tạo lines cho {len(elements)} tam giác...")

            # Log lệnh Line đầu tiên
            n1, n2, n3 = elements[0]
            p1, p2, p3 = node_to_point[n1], node_to_point[n2], node_to_point[n3]
            log_print(f"    Lệnh Line đầu tiên: LineSegment(defPoint=[Point[{p1}], Point[{p2}]], nature=Nature['STANDARD'])")

            line_cache = set()  # Để tránh tạo line trùng
            skipped_boundary_lines = 0  # Đếm số lines bị skip vì cả 2 đều là boundary

            # DEBUG: Log 10 elements đầu tiên với mapping
            log_print("\n--- DEBUG: 10 elements đầu tiên (node -> point) ---")
            for i, (n1, n2, n3) in enumerate(elements[:10]):
                p1, p2, p3 = node_to_point[n1], node_to_point[n2], node_to_point[n3]
                log_print(f"    Element {i+1}: nodes({n1},{n2},{n3}) -> points({p1},{p2},{p3})")

            for elem_idx, (n1, n2, n3) in enumerate(elements):
                # Lấy point IDs tương ứng
                p1 = node_to_point[n1]
                p2 = node_to_point[n2]
                p3 = node_to_point[n3]

                # Tạo 3 lines cho tam giác (nếu chưa có)
                edges = [(p1, p2), (p2, p3), (p3, p1)]

                for edge in edges:
                    # Sắp xếp để tránh trùng lặp (p1, p2) vs (p2, p1)
                    edge_key = tuple(sorted(edge))
                    if edge_key not in line_cache:
                        # Lấy tọa độ của 2 điểm
                        x1, y1 = point_coords.get(edge[0], (0, 0))
                        x2, y2 = point_coords.get(edge[1], (0, 0))

                        # Skip nếu CẢ HAI điểm đều nằm trên CÙNG đường tròn (Rin hoặc Rout)
                        both_on_rin = is_on_rin(x1, y1) and is_on_rin(x2, y2)
                        both_on_rout = is_on_rout(x1, y1) and is_on_rout(x2, y2)

                        if both_on_rin or both_on_rout:
                            skipped_boundary_lines += 1
                            line_cache.add(edge_key)  # Vẫn thêm vào cache để không check lại
                            if skipped_boundary_lines <= 10:
                                circle = "Rin" if both_on_rin else "Rout"
                                log_print(f"    SKIP ({circle}): Point[{edge[0]}] ({x1},{y1}) - Point[{edge[1]}] ({x2},{y2})")
                            continue

                        cmd = f"LineSegment(defPoint=[Point[{edge[0]}], Point[{edge[1]}]], nature=Nature['STANDARD'])"
                        flux.run_command(cmd)
                        line_cache.add(edge_key)

                        # DEBUG: Log 20 lines đầu tiên
                        if len(line_cache) - skipped_boundary_lines <= 20:
                            log_print(f"    Line: Point[{edge[0]}] - Point[{edge[1]}]")

                # Debug: in ra mỗi 500 element
                if (elem_idx + 1) % 500 == 0:
                    log_print(f"    Đã xử lý {elem_idx + 1} elements, tạo {len(line_cache)} lines")

            log_print(f">>> Đã tạo {len(line_cache) - skipped_boundary_lines} lines")
            log_print(f">>> Bỏ qua {skipped_boundary_lines} lines (cả 2 điểm đều trên Rin/Rout)")

            # Tạm bỏ buildFaces để test save
            # log_print("\n>>> Đang build faces...")
            # flux.run_command("buildFaces()")
            # log_print(">>> Build faces xong!")

            # Lưu file
            log_print(f"\n>>> Đang lưu project: {output_path}")
            flux.run_command(f'saveProjectAs("{output_path}")')
            log_print(">>> Lưu xong!")

        log_print("\n--- KẾT THÚC ---")

        # Check file sau khi đóng Flux
        import os
        if os.path.exists(output_path):
            size = os.path.getsize(output_path)
            log_print(f"FILE TỒN TẠI: {output_path}")
            log_print(f"Kích thước: {size} bytes")
        else:
            log_print(f"!!! FILE KHÔNG TỒN TẠI: {output_path}")

    except Exception as e:
        log_print(f"\n!!! LỖI: {type(e).__name__}: {e}")
        import traceback
        log_print(traceback.format_exc())

    finally:
        log.close()


if __name__ == "__main__":
    main()

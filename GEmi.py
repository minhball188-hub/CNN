import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import ezdxf

# --- IMPORT TRỰC TIẾP HÀM CẦN DÙNG ---
from skimage.measure import find_contours, approximate_polygon

# ==========================================
# PHẦN 1: SINH HÌNH (NGNET + MAGNET)
# ==========================================

def ngnet_generate_shape(weights, centers, grid_x, grid_y, sigma):
    function_value = np.zeros_like(grid_x)
    sum_gaussian = np.zeros_like(grid_x)
    for w, center in zip(weights, centers):
        dist_sq = (grid_x - center[0])**2 + (grid_y - center[1])**2
        g = np.exp(-dist_sq / (2 * sigma**2))
        function_value += w * g
        sum_gaussian += g
    sum_gaussian[sum_gaussian == 0] = 1e-10
    return function_value / sum_gaussian

# --- THIẾT LẬP ---
R_out = 50.0; R_in = 20.0; resolution = 200
x = np.linspace(0, R_out, resolution)
y = np.linspace(0, R_out, resolution)
GX, GY = np.meshgrid(x, y)

pixel_size_x = R_out / (resolution - 1)
pixel_size_y = R_out / (resolution - 1)

radius = np.sqrt(GX**2 + GY**2)
angle = np.arctan2(GY, GX) * 180 / np.pi
mask_rotor_region = (radius >= R_in) & (radius <= R_out) & (angle >= 0) & (angle <= 45)

# --- TẠO NAM CHÂM (Parametric) ---
mag_center_x = 35; mag_center_y = 12
mag_width = 18; mag_thick = 4; mag_angle = 15

dX = GX - mag_center_x; dY = GY - mag_center_y
rotated_X = dX * np.cos(np.deg2rad(mag_angle)) + dY * np.sin(np.deg2rad(mag_angle))
rotated_Y = -dX * np.sin(np.deg2rad(mag_angle)) + dY * np.cos(np.deg2rad(mag_angle))

mask_magnet = (np.abs(rotated_X) <= mag_width/2) & \
              (np.abs(rotated_Y) <= mag_thick/2) & \
              mask_rotor_region

# --- TẠO NGNET (Topology) ---
centers = []
for r in np.linspace(R_in+2, R_out-2, 6):
    for theta in np.linspace(2, 43, 6):
        centers.append([r * np.cos(np.deg2rad(theta)), r * np.sin(np.deg2rad(theta))])

weights = np.random.uniform(-2.0, 2.0, len(centers)) 
phi_value = ngnet_generate_shape(weights, centers, GX, GY, sigma=4.0)

# --- TỔNG HỢP ---
final_design = np.full_like(GX, -1)
final_design[mask_rotor_region] = 1
flux_barrier_mask = (phi_value < 0) & mask_rotor_region
final_design[flux_barrier_mask] = 0
final_design[mask_magnet] = 2

# ==========================================
# PHẦN 2: XUẤT DXF (ĐÃ FIX LỖI & LÀM TRƠN)
# ==========================================

def export_contours_to_dxf(matrix, value_to_extract, layer_name, dxf_modelspace):
    # Tạo mask nhị phân
    binary_mask = (matrix == value_to_extract).astype(float)
    
    # Tìm đường bao (Sửa lỗi: dùng trực tiếp hàm find_contours)
    contours = find_contours(binary_mask, level=0.5)
    
    count = 0
    for contour in contours:
        # --- LÀM TRƠN ĐƯỜNG BAO ---
        # tolerance=0.8 giúp giảm số điểm thừa, làm đường thẳng mượt hơn
        simplified_contour = approximate_polygon(contour, tolerance=0.8)
        
        real_points = []
        for p in simplified_contour:
            r, c = p[0], p[1]
            real_x = c * pixel_size_x
            real_y = r * pixel_size_y
            real_points.append((real_x, real_y))
        
        # Lọc bỏ các vụn quá nhỏ (nhiễu)
        if len(real_points) > 3: 
            dxf_modelspace.add_lwpolyline(real_points, close=True, dxfattribs={'layer': layer_name})
            count += 1
    return count

# Khởi tạo file DXF
doc = ezdxf.new()
msp = doc.modelspace()
doc.layers.add(name="MAGNET", color=1)
doc.layers.add(name="AIR_HOLE", color=4)

# Xuất dữ liệu
num_mags = export_contours_to_dxf(final_design, 2, "MAGNET", msp)
num_holes = export_contours_to_dxf(final_design, 0, "AIR_HOLE", msp)

filename = "rotor_design_smoothed.dxf"
doc.saveas(filename)

print(f"--- KẾT QUẢ ---")
print(f"Đã xuất file '{filename}' thành công!")
print(f"Số lượng nam châm: {num_mags}")
print(f"Số lượng lỗ khí NGnet: {num_holes}")

# ==========================================
# PHẦN 3: KIỂM TRA (ĐÃ SỬA LỖI NAME ERROR)
# ==========================================
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
cmap = ListedColormap(['white', 'cyan', 'gray', 'red'])
plt.title("Gốc (Pixel)")
plt.pcolormesh(GX, GY, final_design, cmap=cmap, vmin=-1, vmax=2, shading='auto')
plt.axis('equal')

plt.subplot(1, 2, 2)
plt.title("Vector (Đã làm trơn)")
for val, color in zip([2, 0], ['red', 'blue']):
    # SỬA LỖI Ở ĐÂY: Xóa "measure." đi
    binary_mask = (final_design == val).astype(float)
    contours = find_contours(binary_mask, level=0.5) 
    
    for contour in contours:
        # Làm trơn trước khi vẽ để giống hệt DXF
        simp = approximate_polygon(contour, tolerance=0.8)
        plt.plot(simp[:, 1] * pixel_size_x, simp[:, 0] * pixel_size_y, color=color, linewidth=2)

plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.show()
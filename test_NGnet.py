import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from skimage import measure
from skimage.measure import approximate_polygon
from scipy import ndimage
import ezdxf

# ==========================================
# PHẦN 1: SINH HÌNH
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
np.random.seed(1)  # Fixed seed để kiểm tra mirror
R_out = 39; R_in = 16.0; resolution = 1000
x = np.linspace(0, R_out, resolution)
y = np.linspace(0, R_out, resolution)
GX, GY = np.meshgrid(x, y)

pixel_size_x = R_out / (resolution - 1)
pixel_size_y = R_out / (resolution - 1)

radius = np.sqrt(GX**2 + GY**2)
angle = np.arctan2(GY, GX) * 180 / np.pi
mask_rotor_region = (radius >= R_in) & (radius <= R_out) & (angle >= 0) & (angle <= 90)

# Mask cho vùng NGnet - có margin ở biên 0° và 90° để air KHÔNG chạm biên cắt của Ansys
ANGLE_MARGIN = 0.1  # độ - khoảng cách từ biên 0° và 90°
mask_ngnet_region = (radius >= R_in) & (radius <= R_out) & \
                    (angle >= ANGLE_MARGIN) & (angle <= 90 - ANGLE_MARGIN)

# ==========================================
# NAM CHÂM
# ==========================================
def generate_random_magnet_config(R_in, R_out, num_magnets=1):
    magnets = []
    for _ in range(num_magnets):
        margin = 3.0
        r_center = np.random.uniform(R_in + margin, R_out - margin)
        theta_center = np.random.uniform(2, 88)
        center_x = r_center * np.cos(np.deg2rad(theta_center))
        center_y = r_center * np.sin(np.deg2rad(theta_center))
        width = np.random.uniform(12, 20)
        thickness = np.random.uniform(2.5, 5.0)
        ang = np.random.uniform(0, 90)
        magnets.append({'center_x': center_x, 'center_y': center_y, 
                       'width': width, 'thickness': thickness, 'angle': ang})
    return magnets

def create_magnet_mask(GX, GY, center_x, center_y, width, thickness, angle, region_mask):
    dX = GX - center_x
    dY = GY - center_y
    angle_rad = np.deg2rad(angle)
    rotated_X = dX * np.cos(angle_rad) + dY * np.sin(angle_rad)
    rotated_Y = -dX * np.sin(angle_rad) + dY * np.cos(angle_rad)
    return (np.abs(rotated_X) <= width/2) & (np.abs(rotated_Y) <= thickness/2) & region_mask

magnets_config = generate_random_magnet_config(R_in, R_out, num_magnets=1)
mask_magnet = np.zeros_like(GX, dtype=bool)
for m in magnets_config:
    mask_magnet |= create_magnet_mask(GX, GY, m['center_x'], m['center_y'],
                                       m['width'], m['thickness'], m['angle'], mask_rotor_region)

print(f"Nam châm: pos=({magnets_config[0]['center_x']:.1f}, {magnets_config[0]['center_y']:.1f})")

# ==========================================
# NGNET 0-45° + MIRROR
# ==========================================
centers = []
for r in np.linspace(R_in+2, R_out-2, 6):
    for theta in np.linspace(2, 43, 6):
        centers.append([r * np.cos(np.deg2rad(theta)), r * np.sin(np.deg2rad(theta))])

weights = np.random.uniform(-2.0, 2.0, len(centers))
phi_value = ngnet_generate_shape(weights, centers, GX, GY, sigma=4.0)

# Mirror
phi_mirrored = phi_value.copy()
ang = np.arctan2(GY, GX) * 180 / np.pi
mask_45_90 = ang > 45
phi_mirrored[mask_45_90] = phi_value.T[mask_45_90]

# Tổng hợp
# Dùng mask_ngnet_region cho flux barrier để KHÔNG chạm biên 0° và 90°
final_design = np.full_like(GX, -1)
final_design[mask_rotor_region] = 1  # Toàn bộ rotor là sắt
final_design[(phi_mirrored < 0) & mask_ngnet_region] = 0  # Air chỉ trong vùng có margin
final_design[mask_magnet] = 0  # Vùng nam châm cũng là air (khoét lỗ cho nam châm)

# Tạo riêng mask cho nam châm để export riêng
magnet_design = np.full_like(GX, -1)
magnet_design[mask_magnet] = 2

# ==========================================
# NỘI SUY PIXEL - TRACE BOUNDARY
# ==========================================

def trace_boundary_pixels(binary_mask):
    """
    Trace boundary của vùng binary mask
    Dùng find_contours với padding để không bị cắt ở biên
    """
    from skimage import measure
    from scipy import ndimage
    
    # Label các vùng riêng biệt TRƯỚC
    labeled, num_features = ndimage.label(binary_mask)
    
    all_contours = []
    
    for label_id in range(1, num_features + 1):
        region = (labeled == label_id)
        
        # Pad từng region riêng
        padded = np.pad(region, pad_width=2, mode='constant', constant_values=False)
        
        # Tìm contour
        contours = measure.find_contours(padded.astype(float), level=0.5)
        
        for contour in contours:
            # Trừ padding offset và làm tròn về pixel
            adjusted = [(p[0] - 2, p[1] - 2) for p in contour]
            if len(adjusted) >= 3:
                all_contours.append(adjusted)
    
    return all_contours

def downsample_contour(contour, step=5):
    """Giảm số điểm trong contour bằng cách lấy mỗi step điểm"""
    if len(contour) <= step * 2:
        return contour
    return contour[::step]

def export_air_holes_pixel_trace(matrix, GX, GY, pixel_size_x, pixel_size_y, msp, layer_name, downsample_step=3):
    """
    Export air holes bằng cách trace boundary pixels
    """
    binary_mask = (matrix == 0).astype(bool)
    
    contours = trace_boundary_pixels(binary_mask)
    
    count = 0
    for contour in contours:
        if len(contour) < 10:
            continue
        
        # Downsample để giảm số điểm (tránh file quá nặng)
        contour_ds = downsample_contour(contour, step=downsample_step)
        
        # Chuyển sang mm
        points_mm = []
        for row, col in contour_ds:
            x_mm = col * pixel_size_x
            y_mm = row * pixel_size_y
            points_mm.append((x_mm, y_mm))
        
        if len(points_mm) < 4:
            continue
        
        # Đóng contour
        if points_mm[0] != points_mm[-1]:
            points_mm.append(points_mm[0])
        
        msp.add_lwpolyline(points_mm, close=True, dxfattribs={'layer': layer_name})
        count += 1
    
    return count, contours  # Trả về contours để vẽ

def export_magnet_contours(matrix, pixel_size_x, pixel_size_y, msp, layer_name, tolerance=0.8):
    """Export nam châm bằng find_contours (giữ nguyên)"""
    binary_mask = (matrix == 2).astype(float)
    contours = measure.find_contours(binary_mask, level=0.5)
    
    count = 0
    for contour in contours:
        simplified = approximate_polygon(contour, tolerance=tolerance)
        points = [(p[1] * pixel_size_x, p[0] * pixel_size_y) for p in simplified]
        
        if len(points) > 3:
            msp.add_lwpolyline(points, close=True, dxfattribs={'layer': layer_name})
            count += 1
    
    return count

# ==========================================
# XUẤT DXF
# ==========================================
doc = ezdxf.new()
msp = doc.modelspace()
doc.layers.add(name="MAGNET", color=1)
doc.layers.add(name="AIR_HOLE", color=4)

num_mags = export_magnet_contours(magnet_design, pixel_size_x, pixel_size_y, msp, "MAGNET")
num_holes, air_contours = export_air_holes_pixel_trace(final_design, GX, GY, pixel_size_x, pixel_size_y, 
                                                        msp, "AIR_HOLE", downsample_step=5)

filename = "rotor_design.dxf"
doc.saveas(filename)
print(f"\nFile: {filename} | Nam châm: {num_mags} | Lỗ khí: {num_holes}")

# ==========================================
# VISUALIZATION
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax1 = axes[0]
cmap = ListedColormap(['white', 'cyan', 'gray', 'red'])
# Tạo design để hiển thị (có cả nam châm)
display_design = final_design.copy()
display_design[mask_magnet] = 2  # Thêm nam châm vào để hiển thị
ax1.pcolormesh(GX, GY, display_design, cmap=cmap, vmin=-1, vmax=2, shading='auto')
ax1.set_title("Ma trận Pixel (Nam châm nằm trong lỗ)")
ax1.plot([0, R_out], [0, R_out], 'g--', lw=2)
ax1.set_aspect('equal')

ax2 = axes[1]
ax2.set_title("Contours (Pixel trace - răng cưa)")

# Biên rotor
theta_arc = np.linspace(0, np.pi/2, 100)
ax2.plot(R_out * np.cos(theta_arc), R_out * np.sin(theta_arc), 'k-', lw=1)
ax2.plot(R_in * np.cos(theta_arc), R_in * np.sin(theta_arc), 'k-', lw=1)
ax2.plot([R_in, R_out], [0, 0], 'k-', lw=1)
ax2.plot([0, 0], [R_in, R_out], 'k-', lw=1)

# Air holes từ pixel trace
for contour in air_contours:
    contour_ds = downsample_contour(contour, step=5)
    xs = [c[1] * pixel_size_x for c in contour_ds] + [contour_ds[0][1] * pixel_size_x]
    ys = [c[0] * pixel_size_y for c in contour_ds] + [contour_ds[0][0] * pixel_size_y]
    ax2.plot(xs, ys, 'b-', lw=1)

# Magnet - dùng magnet_design
mag_contours = measure.find_contours((magnet_design == 2).astype(float), level=0.5)
for c in mag_contours:
    ax2.plot(c[:, 1] * pixel_size_x, c[:, 0] * pixel_size_y, 'r-', lw=1.5)

ax2.set_aspect('equal')
ax2.set_xlim(-2, R_out + 2)
ax2.set_ylim(-2, R_out + 2)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("rotor_design.png", dpi=200)
print(f"Đã lưu: rotor_design.png")
plt.show()
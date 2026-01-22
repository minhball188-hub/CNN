import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from skimage import measure
from skimage.measure import approximate_polygon
from scipy.interpolate import interp1d, UnivariateSpline, splprep, splev
from scipy.ndimage import gaussian_filter, gaussian_filter1d
import ezdxf
import os

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
R_out = 39; R_in = 16.0; resolution = 1000 # Tăng độ phân giải lên 600 cho mịn hơn (có thể tăng lên 800 nếu máy đủ mạnh)
x = np.linspace(0, R_out, resolution)
y = np.linspace(0, R_out, resolution)
GX, GY = np.meshgrid(x, y)

# Tính tỷ lệ quy đổi từ Pixel sang mm (Quan trọng để xuất DXF đúng kích thước)
pixel_size_x = R_out / (resolution - 1)
pixel_size_y = R_out / (resolution - 1)

# Mask Rotor
radius = np.sqrt(GX**2 + GY**2)
angle = np.arctan2(GY, GX) * 180 / np.pi
mask_rotor_region = (radius >= R_in) & (radius <= R_out) & (angle >= 0) & (angle <= 90)

# ==========================================
# TẠO NAM CHÂM NGẪU NHIÊN (GIỐNG NGNET)
# ==========================================
def generate_random_magnet_config(R_in, R_out, num_magnets=1):
    """
    Tạo cấu hình nam châm ngẫu nhiên trong vùng rotor
    
    Parameters:
    - R_in, R_out: Bán kính trong và ngoài của rotor
    - num_magnets: Số lượng nam châm cần tạo
    
    Returns:
    - List các dictionary chứa tham số nam châm
    """
    magnets = []
    for _ in range(num_magnets):
        # Tạo vị trí tâm ngẫu nhiên trong vùng rotor (dạng cực)
        # R: từ R_in+margin đến R_out-margin
        # Theta: từ 0 đến 90 độ
        margin = 3.0  # Margin để đảm bảo nam châm không sát biên
        r_center = np.random.uniform(R_in + margin, R_out - margin)
        theta_center = np.random.uniform(2, 88)  # Độ, tránh sát biên góc
        
        # Chuyển sang tọa độ Descartes
        center_x = r_center * np.cos(np.deg2rad(theta_center))
        center_y = r_center * np.sin(np.deg2rad(theta_center))
        
        # Kích thước ngẫu nhiên (có thể điều chỉnh phạm vi)
        width = np.random.uniform(12, 20)        # Chiều rộng: 12-20 mm
        thickness = np.random.uniform(2.5, 5.0)  # Độ dày: 2.5-5 mm
        
        # Góc nghiêng ngẫu nhiên
        angle = np.random.uniform(0, 90)         # Góc: 0-90 độ
        
        magnets.append({
            'center_x': center_x,
            'center_y': center_y,
            'width': width,
            'thickness': thickness,
            'angle': angle
        })
    
    return magnets

# Sinh ngẫu nhiên nam châm (giống như NGnet random)
# Có thể thay đổi num_magnets để tạo nhiều nam châm
magnets_config = generate_random_magnet_config(R_in, R_out, num_magnets=1)

def create_magnet_mask(GX, GY, center_x, center_y, width, thickness, angle, region_mask):
    """
    Tạo mask cho một nam châm hình chữ nhật với các tham số có thể điều chỉnh
    
    Parameters:
    - GX, GY: Lưới tọa độ
    - center_x, center_y: Vị trí tâm nam châm (mm)
    - width: Chiều rộng nam châm (mm)
    - thickness: Độ dày nam châm (mm)
    - angle: Góc nghiêng (độ)
    - region_mask: Mask vùng cho phép (ví dụ: mask_rotor_region)
    
    Returns:
    - mask: Boolean mask cho nam châm
    """
    # Chuyển tọa độ về hệ tọa độ tâm nam châm
    dX = GX - center_x
    dY = GY - center_y
    
    # Xoay hệ tọa độ theo góc nghiêng
    angle_rad = np.deg2rad(angle)
    rotated_X = dX * np.cos(angle_rad) + dY * np.sin(angle_rad)
    rotated_Y = -dX * np.sin(angle_rad) + dY * np.cos(angle_rad)
    
    # Tạo mask hình chữ nhật trong hệ tọa độ đã xoay
    mask = (np.abs(rotated_X) <= width/2) & \
           (np.abs(rotated_Y) <= thickness/2) & \
           region_mask
    
    return mask

# --- TẠO NAM CHÂM TỪ CẤU HÌNH ---
mask_magnet = np.zeros_like(GX, dtype=bool)
print(f"--- THÔNG TIN NAM CHÂM (NGẪU NHIÊN) ---")
for i, magnet in enumerate(magnets_config):
    print(f"Nam châm {i+1}:")
    print(f"  - Vị trí: ({magnet['center_x']:.2f}, {magnet['center_y']:.2f}) mm")
    print(f"  - Chiều rộng: {magnet['width']:.2f} mm")
    print(f"  - Độ dày: {magnet['thickness']:.2f} mm")
    print(f"  - Góc nghiêng: {magnet['angle']:.2f} độ")
    
    mag_mask = create_magnet_mask(
        GX, GY,
        magnet['center_x'],
        magnet['center_y'],
        magnet['width'],
        magnet['thickness'],
        magnet['angle'],
        mask_rotor_region
    )
    mask_magnet = mask_magnet | mag_mask
print()

# --- TẠO NGNET (Topology) ---
centers = []
for r in np.linspace(R_in+2, R_out-2, 6):
    for theta in np.linspace(2, 88, 6):
        centers.append([r * np.cos(np.deg2rad(theta)), r * np.sin(np.deg2rad(theta))])

weights = np.random.uniform(-2.0, 2.0, len(centers)) 
phi_value = ngnet_generate_shape(weights, centers, GX, GY, sigma=4.0)

# --- TỔNG HỢP ---
# 0: Khí, 1: Sắt, 2: Nam châm
# QUAN TRỌNG: Thứ tự gán giá trị phải đúng để nam châm không bị cắt bởi sắt
# Nam châm chỉ bị cắt bởi bán kính giới hạn (R_in, R_out), không bị cắt bởi sắt

final_design = np.full_like(GX, -1) # -1 là nền

# BƯỚC 1: Gán nam châm TRƯỚC (ưu tiên cao nhất)
# Nam châm có thể bị cắt bởi bán kính giới hạn (mask_rotor_region đã xử lý điều này)
# Nhưng nam châm KHÔNG bị cắt bởi sắt hay flux barrier
final_design[mask_magnet] = 2  # Nam châm

# BƯỚC 2: Gán flux barrier (không khí) - nhưng KHÔNG ghi đè lên nam châm
flux_barrier_mask = (phi_value < 0) & mask_rotor_region & (final_design != 2)
final_design[flux_barrier_mask] = 0  # Khí (chỉ ở vùng không phải nam châm)

# BƯỚC 3: Gán sắt - nhưng KHÔNG ghi đè lên nam châm và flux barrier
# Sắt chỉ được gán ở vùng rotor, trừ vùng nam châm và flux barrier
iron_mask = mask_rotor_region & (final_design == -1)
final_design[iron_mask] = 1  # Sắt (chỉ ở vùng không phải nam châm và không phải flux barrier)

# ==========================================
# PHẦN 2: XUẤT DXF (QUAN TRỌNG)
# ==========================================

def smooth_contour_simple(contour_points, num_points=None):
    """
    Làm mịn đơn giản: chỉ resample với nhiều điểm hơn, không dùng spline phức tạp
    Giữ nguyên hình dạng, chỉ làm mịn bằng cách tăng số điểm
    Đây là phương pháp an toàn nhất, không gây biến dạng
    """
    if len(contour_points) < 4:
        return contour_points
    
    points = np.array(contour_points)
    if not np.allclose(points[0], points[-1]):
        points = np.vstack([points, points[0:1]])
    
    # Tính khoảng cách tích lũy
    diffs = np.diff(points, axis=0)
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    cumdist = np.concatenate([[0], np.cumsum(dists)])
    
    if cumdist[-1] < 1e-6:
        return contour_points
    
    cumdist_norm = cumdist / cumdist[-1]
    
    # Xác định số điểm mới
    if num_points is None:
        num_points = max(len(points) * 2, 100)
    
    # Chỉ dùng linear interpolation - đơn giản và an toàn, KHÔNG gây lòi ra ngoài
    u_new = np.linspace(0, 1, num_points)
    f_x = interp1d(cumdist_norm, points[:, 0], kind='linear', 
                   bounds_error=False, fill_value=(points[0, 0], points[-1, 0]),
                   assume_sorted=True)
    f_y = interp1d(cumdist_norm, points[:, 1], kind='linear',
                   bounds_error=False, fill_value=(points[0, 1], points[-1, 1]),
                   assume_sorted=True)
    
    u_new_safe = np.clip(u_new, cumdist_norm[0], cumdist_norm[-1])
    x_new = f_x(u_new_safe)
    y_new = f_y(u_new_safe)
    
    # Đóng đường bao
    x_new[-1] = x_new[0]
    y_new[-1] = y_new[0]
    
    return np.column_stack([x_new[:-1], y_new[:-1]]).tolist()

def smooth_contour_bspline_safe(contour_points, smoothing_strength='medium'):
    """
    Làm mịn bằng B-spline an toàn với validation để tránh lòi ra ngoài
    
    Parameters:
    - contour_points: Array các điểm (x, y) của đường bao
    - smoothing_strength: 'light' (giữ nguyên shape), 'medium' (cân bằng), 'strong' (mịn hơn)
    """
    if len(contour_points) < 4:
        return contour_points
    
    points = np.array(contour_points)
    if not np.allclose(points[0], points[-1]):
        points = np.vstack([points, points[0:1]])
    
    # QUAN TRỌNG: Tính bounding box của contour gốc để giới hạn phạm vi
    x_min_orig = np.min(points[:, 0])
    x_max_orig = np.max(points[:, 0])
    y_min_orig = np.min(points[:, 1])
    y_max_orig = np.max(points[:, 1])
    
    # Thêm margin nhỏ (1% mỗi phía) để tránh clip quá chặt
    x_range = x_max_orig - x_min_orig
    y_range = y_max_orig - y_min_orig
    margin_x = max(x_range * 0.01, 0.1)
    margin_y = max(y_range * 0.01, 0.1)
    
    x_min_clip = x_min_orig - margin_x
    x_max_clip = x_max_orig + margin_x
    y_min_clip = y_min_orig - margin_y
    y_max_clip = y_max_orig + margin_y
    
    # Xác định tham số smoothing
    if smoothing_strength == 'light':
        s = 0.0  # Interpolation (giữ nguyên shape hoàn toàn)
        num_mult = 2
    elif smoothing_strength == 'medium':
        s = len(points) * 0.0005  # Giảm s xuống để tránh ngoại suy
        num_mult = 2
    else:  # strong
        s = len(points) * 0.001  # Giảm s xuống
        num_mult = 2.5
    
    try:
        # B-spline với periodic boundary (đóng đường bao)
        tck, u = splprep([points[:, 0], points[:, 1]], s=s, k=3, per=True)
        u_new = np.linspace(0, 1, int(len(points) * num_mult))
        x_new, y_new = splev(u_new, tck)
        
        # QUAN TRỌNG: Clip các điểm về phạm vi hợp lệ để tránh lòi ra ngoài
        x_new = np.clip(x_new, x_min_clip, x_max_clip)
        y_new = np.clip(y_new, y_min_clip, y_max_clip)
        
        # Đóng đường bao
        x_new[-1] = x_new[0]
        y_new[-1] = y_new[0]
        
        result = np.column_stack([x_new[:-1], y_new[:-1]])
        
        # Validation bổ sung: kiểm tra xem có điểm nào bị clip quá nhiều không
        # Nếu phạm vi mới nhỏ hơn phạm vi gốc quá nhiều, có thể có vấn đề
        x_range_new = np.max(result[:, 0]) - np.min(result[:, 0])
        y_range_new = np.max(result[:, 1]) - np.min(result[:, 1])
        
        if x_range > 0 and (x_range_new / x_range < 0.5):
            # Nếu bị shrink quá nhiều, dùng simple method
            return smooth_contour_simple(contour_points)
        if y_range > 0 and (y_range_new / y_range < 0.5):
            return smooth_contour_simple(contour_points)
        
        return result.tolist()
    except Exception as e:
        # Fallback về simple nếu B-spline fail
        return smooth_contour_simple(contour_points)

def smooth_contour(contour_points, smoothing_factor=2.0, num_points=None, method='bspline'):
    """
    Làm mịn đường bao với nhiều phương pháp
    
    Parameters:
    - contour_points: Array các điểm (x, y) của đường bao
    - smoothing_factor: Độ mịn (chỉ dùng với method='advanced')
    - num_points: Số điểm sau khi làm mịn (None = tự động)
    - method: 'bspline' (B-spline an toàn - KHUYẾN NGHỊ), 
              'simple' (chỉ resample, an toàn), 
              'none' (không làm mịn), 
              'advanced' (spline phức tạp, có thể gây biến dạng)
    """
    if method == 'none':
        return contour_points
    
    if method == 'bspline':
        return smooth_contour_bspline_safe(contour_points, smoothing_strength='medium')
    
    if method == 'simple':
        return smooth_contour_simple(contour_points, num_points)
    
    # method == 'advanced' - phương pháp phức tạp hơn (có thể gây biến dạng)
    # Khuyến nghị: dùng 'bspline' thay vì 'advanced'
    return smooth_contour_simple(contour_points, num_points)

def export_contours_to_dxf(matrix, value_to_extract, layer_name, dxf_modelspace, 
                           use_spline=False, tolerance=0.8):
    """
    Hàm trích xuất đường bao của một giá trị cụ thể trong ma trận và vẽ vào DXF
    Sử dụng approximate_polygon để làm mịn - phương pháp đơn giản và hiệu quả
    
    Parameters:
    - use_spline: Nếu True, xuất ra SPLINE thay vì LWPOLYLINE (tốt cho Ansys Maxwell)
                  LƯU Ý: KHÔNG nên dùng spline cho hình chữ nhật (nam châm) vì sẽ làm biến dạng
    - tolerance: Độ tolerance cho approximate_polygon 
                 - Hình chữ nhật (nam châm): dùng 0.2-0.3 để giữ góc vuông
                 - Đường cong (lỗ khí): dùng 0.5-1.5 để làm mịn
                 Càng lớn càng mịn nhưng có thể mất chi tiết và dịch chuyển vị trí
    """
    # Tạo mask nhị phân
    binary_mask = (matrix == value_to_extract).astype(float)
    
    # Tìm đường bao (contours) với subpixel accuracy
    contours = measure.find_contours(binary_mask, level=0.5)
    
    count = 0
    for contour in contours:
        # --- LÀM TRƠN ĐƯỜNG BAO BẰNG approximate_polygon ---
        # Phương pháp Douglas-Peucker: giảm số điểm thừa, làm đường thẳng mượt hơn
        # tolerance càng lớn càng mịn, nhưng có thể mất chi tiết nhỏ
        simplified_contour = approximate_polygon(contour, tolerance=tolerance)
        
        # Chuyển đổi sang tọa độ thực (mm)
        real_points = []
        for p in simplified_contour:
            r, c = p[0], p[1]
            real_x = c * pixel_size_x
            real_y = r * pixel_size_y
            real_points.append((real_x, real_y))
        
        # Lọc bỏ các vụn quá nhỏ (nhiễu)
        if len(real_points) > 3:
            if use_spline and len(real_points) >= 4:
                # Xuất ra SPLINE (mịn hơn, tốt cho Ansys Maxwell)
                try:
                    dxf_modelspace.add_spline(
                        control_points=real_points,
                        degree=3,
                        dxfattribs={'layer': layer_name}
                    )
                except:
                    # Fallback về polyline nếu spline không được hỗ trợ
                    dxf_modelspace.add_lwpolyline(real_points, close=True, 
                                                  dxfattribs={'layer': layer_name})
            else:
                # Xuất ra LWPOLYLINE
                dxf_modelspace.add_lwpolyline(real_points, close=True, 
                                              dxfattribs={'layer': layer_name})
            count += 1
    return count

# Khởi tạo file DXF
doc = ezdxf.new()
msp = doc.modelspace()

# Tạo các Layer để quản lý màu sắc
# Lưu ý: KHÔNG tạo layer AIR_HOLE vì vùng không khí không được xuất ra DXF
doc.layers.add(name="MAGNET", color=1)   # Màu đỏ (Red)
# doc.layers.add(name="AIR_HOLE", color=4) # KHÔNG dùng - vùng air không được xuất

# --- XUẤT VÙNG SẮT (Giá trị 1) ---
# FIX: Xuất vùng sắt với các lỗ khí bên trong được xử lý đúng cách
# QUAN TRỌNG: Loại bỏ phần sắt bị che bởi nam châm (nam châm cắt sắt, không phải ngược lại)
# Tạo matrix riêng cho sắt, loại bỏ phần nam châm
iron_design = final_design.copy()
iron_design[iron_design == 2] = -1  # Loại bỏ nam châm khỏi matrix sắt

doc.layers.add(name="IRON", color=7)  # Màu trắng/xám cho sắt
num_iron = export_contours_to_dxf(iron_design, 1, "IRON", msp,
                                   use_spline=True, tolerance=0.8)

# --- XUẤT NAM CHÂM (Giá trị 2) ---
# FIX: tolerance nhỏ hơn (0.2-0.3) để giữ góc vuông của hình chữ nhật
# FIX: KHÔNG dùng spline cho nam châm vì spline làm biến dạng hình chữ nhật
# Dùng polyline để giữ nguyên hình dạng chữ nhật
num_mags = export_contours_to_dxf(final_design, 2, "MAGNET", msp, 
                                   use_spline=False, tolerance=0.25)

# --- KHÔNG XUẤT LỖ KHÍ (Giá trị 0) ---
# FIX: Vùng không khí (value=0) KHÔNG được xuất ra DXF như một vật thể riêng
# Các lỗ khí sẽ tự động được xử lý như lỗ rỗng bên trong vùng sắt
# khi Ansys import và thực hiện boolean operations
# num_holes = export_contours_to_dxf(final_design, 0, "AIR_HOLE", msp,
#                                     use_spline=True, tolerance=0.8)
num_holes = 0  # Không xuất vùng air

# Lưu file
filename = "rotor_design.dxf"
doc.saveas(filename)

print(f"--- KẾT QUẢ ---")
print(f"Đã xuất file '{filename}' thành công!")
print(f"Số lượng vùng sắt: {num_iron}")
print(f"Số lượng nam châm: {num_mags}")
print(f"Lưu ý: Vùng không khí (air) KHÔNG được xuất ra DXF để tránh lỗi cắt vật thể")
print(f"      Các lỗ khí sẽ được xử lý như lỗ rỗng bên trong vùng sắt khi import vào Ansys")

# ==========================================
# PHẦN 3: KIỂM TRA (VẼ LẠI DXF LÊN MÀN HÌNH)
# ==========================================
plt.figure(figsize=(10, 5))

# Hình 1: Ma trận gốc (Pixel)
plt.subplot(1, 2, 1)
cmap = ListedColormap(['white', 'cyan', 'gray', 'red'])
plt.title("Ma trận Pixel (Gốc)")
plt.pcolormesh(GX, GY, final_design, cmap=cmap, vmin=-1, vmax=2, shading='auto')
plt.axis('equal')

# Hình 2: Đường bao Vector (Cái sẽ vào Ansys)
plt.subplot(1, 2, 2)
plt.title("Đường bao Vector (Sẽ xuất DXF)")
# Vẽ lại nền rotor cho dễ nhìn
plt.plot([0, R_out], [0, 0], 'k--', alpha=0.3)
plt.plot([0, R_out*np.cos(np.pi/2)], [0, R_out*np.sin(np.pi/2)], 'k--', alpha=0.3)

# Vẽ lại các contour tìm được (đã được làm mịn bằng approximate_polygon)
# FIX: Dùng tolerance giống như khi export để đảm bảo khớp nhau
# FIX: Chỉ vẽ vùng sắt (1) và nam châm (2), KHÔNG vẽ vùng air (0)
for val, color, tol in zip([1, 2], ['gray', 'red'], [0.8, 0.25]):
    binary_mask = (final_design == val).astype(float)
    contours = measure.find_contours(binary_mask, level=0.5)
    
    for contour in contours:
        # Làm trơn bằng approximate_polygon với tolerance tương ứng
        # Vùng sắt (val=1): tolerance=0.8 để làm mịn đường cong
        # Nam châm (val=2): tolerance=0.25 để giữ góc vuông
        simplified_contour = approximate_polygon(contour, tolerance=tol)
        
        # Chuyển đổi sang tọa độ thực và vẽ
        plt.plot(simplified_contour[:, 1] * pixel_size_x, 
                simplified_contour[:, 0] * pixel_size_y, 
                color=color, linewidth=2, alpha=0.8)

plt.axis('equal')
plt.grid(True, alpha=0.3)

# Tự động lưu ảnh sau mỗi lần chạy (ghi đè file cũ)
image_filename = "rotor_design.png"
plt.savefig(image_filename, dpi=300, bbox_inches='tight')
print(f"Đã lưu ảnh: '{image_filename}'")

plt.show()
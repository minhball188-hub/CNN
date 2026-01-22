import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
import ezdxf
from scipy.interpolate import splprep, splev

# ==========================================
# PHẦN 1: SINH HÌNH ROTOR IPM (DẠNG THANH CHUẨN)
# ==========================================
# Thay vì dùng Gaussian (tròn), ta dùng hàm hình học để tạo thanh nam châm chữ nhật
def generate_flat_ipm_field(grid_x, grid_y, R_out, R_in):
    # Khởi tạo ma trận vật liệu (1: Thép)
    field = np.ones_like(grid_x)
    
    # 1. Tạo lỗ trục (R_in)
    r = np.sqrt(grid_x**2 + grid_y**2)
    field[r < R_in] = 1 # Trục là thép
    field[r > R_out] = 0 # Ngoài là khí

    # 2. Định nghĩa nam châm (Hình chữ nhật)
    # Tham số nam châm IPM (bạn có thể chỉnh kích thước ở đây cho giống Motor-CAD)
    mag_width = 24.0   # Chiều dài thanh nam châm
    mag_thick = 6.0    # Độ dày nam châm
    mag_pos_y = 30.0   # Khoảng cách từ tâm ra nam châm
    
    # Tạo mask hình chữ nhật cho nam châm
    # Logic: |x| < width/2  VÀ  |y - pos| < thick/2
    mask_magnet = (np.abs(grid_x) < mag_width/2) & \
                  (np.abs(grid_y - mag_pos_y) < mag_thick/2)
    
    # Gán giá trị nam châm (2)
    field[mask_magnet] = 2 
    
    # 3. Tạo cầu từ (Flux bridge) - phần thép nhỏ giữ nam châm không bay ra
    # Nếu muốn thêm lỗ khí 2 bên đầu nam châm (Air pocket), ta đục thêm
    pocket_width = 4.0
    mask_pocket_left = (grid_x < -mag_width/2) & (grid_x > -mag_width/2 - pocket_width) & \
                       (np.abs(grid_y - mag_pos_y) < mag_thick/2)
    mask_pocket_right = (grid_x > mag_width/2) & (grid_x < mag_width/2 + pocket_width) & \
                        (np.abs(grid_y - mag_pos_y) < mag_thick/2)
    
    field[mask_pocket_left] = 0 # Khí
    field[mask_pocket_right] = 0 # Khí

    return field

# --- THIẾT LẬP LƯỚI ---
R_out = 50.0; R_in = 15.0; resolution = 300
# Lưu ý: Tôi mở rộng vùng x từ -50 đến 50 để vẽ trọn vẹn 1 cực từ (chứ không chỉ góc 1/4)
x = np.linspace(-50, 50, resolution) 
y = np.linspace(0, 60, resolution)   
GX, GY = np.meshgrid(x, y)
pixel_size = 100 / (resolution - 1) # Ước lượng kích thước pixel

# Sinh hình IPM chuẩn
final_design = generate_flat_ipm_field(GX, GY, R_out, R_in)

# ==========================================
# PHẦN 2: XUẤT DXF SẮC NÉT (Dùng Polygon Simplification)
# ==========================================
def clean_and_export(matrix, val_target, layer_name, dxf_space):
    # Tìm biên dạng pixel
    contours = measure.find_contours(matrix == val_target, 0.5)
    count = 0
    for contour in contours:
        # BƯỚC QUAN TRỌNG: approximate_polygon
        # tolerance=0.5 -> Giữ cạnh thẳng tắp, loại bỏ bậc thang pixel
        poly = measure.approximate_polygon(contour, tolerance=0.8)
        
        # Với hình cơ khí (IPM), ta KHÔNG CẦN spline làm cong mềm
        # Ta chỉ cần nối các điểm chốt lại là thành hình chữ nhật đẹp
        y_final = poly[:, 0]
        x_final = poly[:, 1]

        # Chuyển đổi toạ độ pixel sang mm thực
        # Lưu ý map lại toạ độ theo lưới linspace bên trên
        x_real = (x_final / (resolution-1)) * 100 - 50 # Map về -50..50
        y_real = (y_final / (resolution-1)) * 60       # Map về 0..60
        
        points_mm = np.column_stack((x_real, y_real))
        
        # Vẽ nét liền (LWPolyline)
        dxf_space.add_lwpolyline(points_mm, close=True, dxfattribs={'layer': layer_name, 'color': 7})
        count += 1
    return count

# Khởi tạo DXF
doc = ezdxf.new()
msp = doc.modelspace()

# Xuất Nam châm (Magnet)
clean_and_export(final_design, 2, "MAGNET", msp)
# Xuất Lỗ khí (Air) - bao gồm cả pocket 2 đầu
clean_and_export(final_design, 0, "AIR", msp)

filename = "rotor_IPM_standard.dxf"
doc.saveas(filename)

# ==========================================
# VẼ MINH HỌA
# ==========================================
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Mô hình Pixel (Đầu vào)")
plt.pcolormesh(GX, GY, final_design, cmap='jet', shading='auto')
plt.axis('equal')

plt.subplot(1, 2, 2)
plt.title("Kết quả DXF (Sẽ nhận được)")
# Mô phỏng lại logic vector hóa để hiển thị
contours = measure.find_contours(final_design == 2, 0.5)
for c in contours:
    poly = measure.approximate_polygon(c, tolerance=0.8)
    x_real = (poly[:,1] / (resolution-1)) * 100 - 50
    y_real = (poly[:,0] / (resolution-1)) * 60
    plt.plot(x_real, y_real, 'r-', linewidth=2, label='Magnet')

contours_air = measure.find_contours(final_design == 0, 0.5)
for c in contours_air:
    poly = measure.approximate_polygon(c, tolerance=0.8)
    x_real = (poly[:,1] / (resolution-1)) * 100 - 50
    y_real = (poly[:,0] / (resolution-1)) * 60
    # Chỉ vẽ những lỗ khí nằm trong vùng rotor
    if np.mean(y_real) < R_out and np.mean(y_real) > 0: 
        plt.plot(x_real, y_real, 'b-', linewidth=2, label='Air')

plt.xlim(-50, 50); plt.ylim(0, 60)
plt.gca().set_aspect('equal')
plt.grid(True, linestyle='--')
plt.show()

print(f"Đã xuất file '{filename}'. Bây giờ hình sẽ thẳng và sắc nét đúng kiểu IPM.")
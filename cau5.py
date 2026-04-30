"""
Bài toán nén ảnh sử dụng PCA có thể được giải thích như sau:
Khi ta chia bước ảnh xám thành các vector hàng, mỗi vector đó đại diện cho một thông tin trong bức ảnh. 
Nếu bức ảnh có những chi tiết lớn và quan trọng, nhiều vector sẽ có phương sai lớn, tức chúng phân bổ gần nhau theo một hướng nào đó.
Ngược lại, những chi tiết nhỏ và khong quan trọng kéo những vector đó lệch khỏi hướng chính, tạo ra những vector có phương sai nhỏ.
PCA sẽ tìm ra các hướng (eigenvectors) mà dữ liệu phân bố nhiều nhất (có phương sai lớn nhất).
Khi ta giữ lại một số lượng thành phần chính nhất định (k), PCA sẽ loại bỏ những thành phần có phương sai nhỏ, tức là loại bỏ những chi tiết nhỏ và nhiễu trong ảnh.
Điều này giúp giảm kích thước dữ liệu (nén ảnh) và đồng thời loại bỏ những chi tiết không quan trọng, từ đó giúp khôi phục lại ảnh gốc tốt hơn khi tái tạo lại ảnh từ không gian giảm chiều.
"""



import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# from sklearn.decomposition import PCA

from cau2 import *


# a) Tiền xử lý
# Chuyển ảnh RGB về ảnh grayscale -> np.array 2D
image = Image.open("dataset/panda_image_0032.jpg").convert("L")
X = np.array(image, dtype=np.float64)

# Chuẩn hóa dữ liệu
mean_vec = np.mean(X, axis=0)
X_centered = standardize_data(X)
cov_matrix = compute_covariance_matrix(X_centered)
eigenvalues, eigenvectors = compute_eigenvalues_and_eigenvectors(cov_matrix)

# b) Áp dụng PCA
K = [2, 5, 10, 20, 50, 100]

# Khởi tạo Figure chứa lưới 2 hàng x 4 cột
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
# Hàm ravel() biến mảng axes 2D thành 1D (danh sách 8 ô liên tiếp) để tiện gọi bằng vòng lặp
axes = axes.ravel() 

# Hiển thị ảnh gốc ở ô đầu tiên
axes[0].imshow(X, cmap="gray")
axes[0].set_title("Ảnh gốc", fontsize=14)
axes[0].axis("off")

for i, k in enumerate(K):
    # Chọn k thành phần chính
    selected_eigenvectors, _ = select_principal_components(eigenvalues, eigenvectors, k)
    
    # Giảm chiều dữ liệu
    X_reduced = project_data(X_centered, selected_eigenvectors)
    
    # Để tái tạo ảnh, ta nhân ngược lại với ma trận eigenvectors chuyển vị, rồi cộng lại mean_vec
    X_approx = np.dot(X_reduced, selected_eigenvectors.T) + mean_vec

    # Lấy phần thực để loại bỏ sai số do tính toán số phức
    X_approx = np.real(X_approx)
    
    
    
    # Đánh giá (MSE)
    mse = np.mean((X - X_approx) ** 2)
    print(f"K={k}, MSE: {mse:.2f}")
    
    # Trực quan hóa
    ax = axes[i + 1]
    ax.imshow(X_approx, cmap="gray")
    ax.set_title(f"K={k} (MSE: {mse:.2f})", fontsize=12)
    ax.axis("off")

axes[7].axis("off")
plt.tight_layout()
plt.show()

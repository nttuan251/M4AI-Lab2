import matplotlib.pyplot as plt
from sklearn import datasets

from cau2 import *

def compute_PCA(X: np.ndarray, n_components: int):
    mean_vec = np.mean(X, axis=0)
    Xhat = standardize_data(X)
    cov_matrix = compute_covariance_matrix(Xhat)
    eigenvalues, eigenvectors = compute_eigenvalues_and_eigenvectors(cov_matrix)
    selected_eigenvectors, selected_eigenvalues = select_principal_components(eigenvalues, eigenvectors, n_components)
    projected_data = project_data(Xhat, selected_eigenvectors)
    return mean_vec, projected_data, selected_eigenvalues, selected_eigenvectors, eigenvalues

if __name__ == "__main__":
    iris = datasets.load_iris()

    X = iris.data
    

    X_original = X[:, :3] # Chỉ lấy 3 chiều đầu tiên để trực quan hóa cho dễ (Sepal Length, Sepal Width, Petal Length)
    mean_vec, projected_data, selected_eigenvalues, selected_eigenvectors, eigenvalues = compute_PCA(X_original, n_components=2)

    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(X_original[:, 0], X_original[:, 1], X_original[:, 2], c="blue", marker="o")
    
    
    colors_3d = ["red", "purple"]
    labels_3d = ["PC1", "PC2"]
    
    for i in range(2):
        v = selected_eigenvectors[:, i]
        val = selected_eigenvalues[i]
        
        # Độ dài mũi tên = 2 * độ lệch chuẩn
        length = np.sqrt(val) * 2 

        dx = v[0] * length
        dy = v[1] * length
        dz = v[2] * length
        
        ax1.quiver(mean_vec[0], mean_vec[1], mean_vec[2], 
                   dx, dy, dz, 
                   color=colors_3d[i], linewidth=3, arrow_length_ratio=0.1, label=labels_3d[i])

    ax1.set_title("Original Data (3D) & Eigenvectors")
    ax1.set_xlabel("Sepal Length")
    ax1.set_ylabel("Sepal Width")
    ax1.set_zlabel("Petal Length")
    ax1.legend(loc="upper right")


    ax2 = fig.add_subplot(122)
    ax2.scatter(projected_data[:, 0], projected_data[:, 1], c="green", marker="o", alpha=0.6)
    
    # Vẽ vector riêng ở không gian 2D 
    ax2.arrow(0, 0, np.sqrt(selected_eigenvalues[0]) * 2, 0, 
              color="red", width=0.05, head_width=0.2, label="PC1")
    ax2.arrow(0, 0, 0, np.sqrt(selected_eigenvalues[1]) * 2, 
              color="purple", width=0.05, head_width=0.2, label="PC2")

    ax2.set_title("Projected Data (2D) & Eigenvectors")
    ax2.set_xlabel("Principal Component 1 (PC1)")
    ax2.set_ylabel("Principal Component 2 (PC2)")
    ax2.legend(loc="upper right")
    

    plt.axis("equal")
    plt.show()


    # Tính toán tỉ lệ dữ liệu được giữ lại
    total_variance = np.sum(selected_eigenvalues)
    explained_variance_ratio = total_variance / np.sum(eigenvalues)
    print(f"Explained Variance Ratio: {explained_variance_ratio:.4f}")
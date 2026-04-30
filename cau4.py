import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import datasets

from cau2 import * 

def custom_pca(X, n_components):
    X_centered = standardize_data(X)  
    
    cov_matrix = compute_covariance_matrix(X_centered)
    
    eigenvalues, eigenvectors = compute_eigenvalues_and_eigenvectors(cov_matrix)
    
    W, sorted_eigenvalues = select_principal_components(eigenvalues, eigenvectors, n_components)
    
    X_projected = project_data(X_centered, W)
    
    return X_projected, sorted_eigenvalues[:n_components], W

if __name__ == "__main__":
    iris = datasets.load_iris()

    X = iris.data
    y_labels = np.array([f"Iris-{name}" for name in iris.target_names[iris.target]])

    
    n_comps = 2

    # Chạy và đo thời gian cho PCA tự cài đặt
    start_time = time.perf_counter()
    X_custom, eigvals_custom, W_custom = custom_pca(X, n_components=n_comps)
    time_custom = time.perf_counter() - start_time

    # Chạy và đo thời gian cho PCA của sklearn
    start_time = time.perf_counter()
    pca_sklearn = PCA(n_components=n_comps)
    X_sklearn = pca_sklearn.fit_transform(X)
    time_sklearn = time.perf_counter() - start_time


    # So sánh và đánh giá
    print("--- So sánh Eigenvalues---")
    print(f"Tự cài đặt: {eigvals_custom}")
    print(f"Scikit-learn: {pca_sklearn.explained_variance_}")
    
    print("\n--- Đánh giá tốc độ chạy ---")
    print(f"Tự cài đặt: {time_custom:.6f} giây")
    print(f"Scikit-learn: {time_sklearn:.6f} giây")
    
    print("\n--- Đánh giá sai số giữa 2 cách ---")
    # Vector riêng có thể bị lật dấu (nhân với -1) tùy vào thuật toán giải.
    # Do đó, để so sánh sai số chiếu, ta dùng trị tuyệt đối.
    error = np.mean(np.abs(np.abs(X_custom) - np.abs(X_sklearn)))
    print(f"Sai số trung bình tuyệt đối: {error:.10f}")



    # Trực quan hóa
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    color_map = {"Iris-setosa": "blue", "Iris-versicolor": "orange", "Iris-virginica": "green"}

    for species in np.unique(y_labels):
        mask = y_labels == species
        # Plot tự cài
        ax1.scatter(X_custom[mask, 0], X_custom[mask, 1], 
                    color=color_map[species], alpha=0.7, edgecolors="w", s=60, label=species)
        
        # Plot sklearn
        ax2.scatter(X_sklearn[mask, 0], X_sklearn[mask, 1], 
                    color=color_map[species], alpha=0.7, edgecolors="w", s=60, label=species)
        

    # Vẽ eigenvectors
    ax1.arrow(0, 0, np.sqrt(eigvals_custom[0]) * 2, 0, 
            color="red", width=0.05, head_width=0.2, label="PC1")
    ax1.arrow(0, 0, 0, np.sqrt(eigvals_custom[1]) * 2, 
            color="purple", width=0.05, head_width=0.2, label="PC2")
    
    ax2.arrow(0, 0, np.sqrt(pca_sklearn.explained_variance_[0]) * 2, 0, 
            color="red", width=0.05, head_width=0.2, label="PC1")
    ax2.arrow(0, 0, 0, np.sqrt(pca_sklearn.explained_variance_[1]) * 2, 
            color="purple", width=0.05, head_width=0.2, label="PC2")


    ax1.set_title("Kết quả PCA (Tự cài đặt)")
    ax1.set_xlabel("PC 1")
    ax1.set_ylabel("PC 2")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()
    ax1.axis("equal")  

    ax2.set_title("Kết quả PCA (Scikit-learn)")
    ax2.set_xlabel("PC 1")
    ax2.set_ylabel("PC 2")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()
    ax2.axis("equal")

    plt.tight_layout()
    plt.show()
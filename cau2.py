import numpy as np

np.random.seed(42)


# Bước 1: Chuẩn hóa dữ liệu
def standardize_data(X: np.ndarray) -> np.ndarray:
    mean = np.mean(X, axis=0)
    return X - mean

# Bước 2: Tinh toán ma trận hiệp phương sai
def compute_covariance_matrix(X: np.ndarray) -> np.ndarray:
    return np.cov(X, rowvar=False)

# Bước 3: Tính toán các giá trị riêng và vector riêng
def compute_eigenvalues_and_eigenvectors(cov_matrix: np.ndarray):
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    return eigenvalues, eigenvectors

# Bước 4: Chọn các thành phần chính
def select_principal_components(eigenvalues: np.ndarray, eigenvectors: np.ndarray, n_components: int):
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues_copy = eigenvalues[idx].copy()
    eigenvectors_copy = eigenvectors[:, idx].copy()
    return eigenvectors_copy[:, :n_components], eigenvalues_copy[:n_components]

# Bước 5: Biểu diễn dữ liệu trên không gian mới
def project_data(X: np.ndarray, eigenvectors: np.ndarray) -> np.ndarray:
    return np.dot(X, eigenvectors)


if __name__ == "__main__":
    X = np.random.randn(10, 5)
    print("Original Data:\n", X)


    Xhat = standardize_data(X)
    cov_matrix = compute_covariance_matrix(Xhat)


    eigenvalues, eigenvectors = compute_eigenvalues_and_eigenvectors(cov_matrix)
    print("Eigenvalues:", eigenvalues)
    print("Eigenvectors:\n", eigenvectors)


    n_components = 2
    selected_eigenvectors, selected_eigenvalues = select_principal_components(eigenvalues, eigenvectors, n_components)
    # print(f"Selected Eigenvalues (Top {n_components}):", selected_eigenvalues)
    # print(f"Selected Eigenvectors (Top {n_components}):\n", selected_eigenvectors)


    projected_data = project_data(Xhat, selected_eigenvectors)
    print("Projected Data:\n", projected_data)
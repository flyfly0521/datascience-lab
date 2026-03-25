# datascience-lab
## Course Project 1: SVM Classification And Dimension Reduction

**Author:** Zheng Feiyang 

**Code Repository:** [flyfly0521/datascience-lab](https://github.com/flyfly0521/datascience-lab) (lab1 sub-folder)

---

### 1. Experimental Setting

We use the **One-vs-Rest (OvR)** strategy to extend Support Vector Machines (SVM) for multi-class classification on the [Animals with Attributes 2 (AwA2)](https://cvml.ist.ac.at/AwA2) dataset. Specifically, we train 50 independent binary SVM classifiers, each corresponding to one class in the dataset.

For each classifier, samples belonging to the target class are treated as positive (+1), while all other samples are treated as negative (-1). During inference, a test sample is evaluated by all classifiers, and the class associated with the highest decision function value is selected as the final prediction.

We implement SVM using the **SMO algorithm** for efficient training. To accelerate the most computationally intensive part—the kernel function evaluation—we leverage **SIMD vectorization with AVX2 instructions**, significantly improving throughput. Furthermore, under the OvR framework, we assign one thread to train each class-specific SVM, enabling parallel training of all classifiers on a 24-core CPU. This design results in substantial speedup compared to a sequential implementation.

---

### 2. K-fold Cross-validation for the decision of C
**Code:** `k-fold.cpp`

To improve generalization performance, we apply hyperparameter tuning for the regularization parameter $C$ using **10-fold cross-validation** on the training set.

#### Table 1: Hyperparameter settings for SVM
| Parameter | Value |
| :--- | :--- |
| Regularization parameter $C$ | $\{10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}, 1, 10, 10^{2}, 10^{3}, 10^{4}\}$ |
| Number of folds ($K$) | 10 |

#### Table 2: SVM 10-fold CV average accuracy for different $C$ values
| $C$ | Avg Accuracy |
| :--- | :--- |
| 0.0001 | 0.4121 |
| 0.001 | 0.6572 |
| 0.01 | 0.7595 |
| 0.1 | 0.7598 |
| 1 | 0.7598 |
| 10 | 0.7598 |
| 100 | 0.7598 |
| 1000 | 0.7598 |
| 10000 | 0.7598 |

As shown in Table 2, the average accuracy improves significantly as $C$ increases from $10^{-4}$ to $1$, indicating that a stronger regularization penalty helps the SVM fit the training data.

The regularization parameter $C$ in soft-margin SVM controls the trade-off between maximizing the margin and minimizing the training error. In our experiments, we observe that for $C \ge 1$, the average accuracy saturates at 0.7598. This is because the deep **ResNet101 features** are already nearly linearly separable, so the SVM is able to correctly classify most training samples even at moderate $C$. Therefore, **$C = 1$** is a reasonable choice, balancing model capacity, numerical stability, and generalization.

---

### 3. Trained Linear Kernel SVM on test set
**Code:** `svm.cpp`

The linear SVM ($C = 1$) was trained on the AwA2 dataset using a 6/4 train/test split. On the test set, the model achieves an accuracy of **0.6780**, which is consistent with the cross-validation results on the training data. While the ResNet101 features are highly expressive and nearly linearly separable, the linear SVM cannot fully capture subtle non-linear inter-class relationships, which limits the maximum achievable accuracy. Nevertheless, this result demonstrates that a linear SVM serves as a reasonable baseline for multi-class classification with deep features.

---

### 4. Dimension Reduction
**Code:** `svm.cpp`, `vae.py`, `test_vae.py`

#### 4.1 Feature Selection
We applied **Fisher feature selection** to identify the most discriminative dimensions from the original 2048-dimensional feature vectors.

#### Table 3: Test accuracy with different numbers of features (Fisher Selection)
| Number of Features ($d$) | Test Accuracy |
| :--- | :--- |
| 128 | 0.0825 |
| 256 | 0.5658 |
| 512 | **0.7747** |
| 1024 | 0.7629 |

Selecting too few features (e.g., 128) leads to very low accuracy due to information loss. The best performance is achieved with **512 features (0.7747)**. Adding more features beyond 512 slightly decreases accuracy, likely because less informative or noisy features are introduced, which can hurt generalization.

#### 4.2 Feature Projection
We further apply **Principal Component Analysis (PCA)** to reduce the dimensionality of the original feature space.

#### Table 4: Test accuracy with different dimensions using PCA
| Number of Components ($d$) | Test Accuracy |
| :--- | :--- |
| 128 | 0.5659 |
| 256 | 0.6567 |
| 512 | 0.7449 |
| 1024 | **0.7565** |

Unlike Fisher selection, PCA's accuracy consistently improves as the number of retained components increases, achieving 0.7565 at 1024 dimensions. Since PCA is unsupervised and does not explicitly optimize class separability, it requires more components to capture enough informative variance for high-accuracy classification.

#### 4.3 Feature Learning
We evaluate **Variational Autoencoder (VAE)** as a nonlinear dimensionality reduction method.

#### Table 5: Test accuracy with different latent dimensions using VAE
| Latent Dimension ($d$) | Test Accuracy |
| :--- | :--- |
| 128 | 0.4523 |
| 256 | 0.3919 |
| 512 | 0.5509 |
| 1024 | **0.5631** |

The performance of VAE is consistently lower than Fisher selection and PCA. As an unsupervised generative model, VAE focuses on data reconstruction rather than maximizing class separability, which may lead to latent representations that are less suitable for SVM classification tasks.

---

### 5. Conclusion
In this project, we compared three dimensionality reduction approaches—Fisher feature selection, PCA, and VAE.

- **Fisher Feature Selection** achieved the best overall performance, reaching the highest test accuracy of **0.7747** at $d=512$.
- **PCA** showed a stable trend with accuracy improving as dimensions increased.
- **VAE** provided the weakest representation for this specific classification task.

Based on these observations, we select **Fisher feature selection with $d=512$** as the optimal configuration, providing the best trade-off between accuracy and computational efficiency.

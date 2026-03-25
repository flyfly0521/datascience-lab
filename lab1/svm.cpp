#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <sstream>
#include <thread>
#include <mutex>
#include <immintrin.h>
#include <set>
#include <eigen3/Eigen/Dense>
//declaration
using namespace std;
using namespace Eigen;
struct SVMModel {
    vector<vector<double>> X;
    vector<int> y;
    vector<double> alpha;
    double b;
};
bool divide_dataset(string BASE_DIR);
SVMModel train_svm(
    vector<vector<double>>& X,
    vector<int>& y,
    double C,
    int d
);
double predict(
    const vector<vector<double>>& X,
    const vector<int>& y,
    const vector<double>& alpha,
    double b,
    const vector<double>& x,
    int d
);
double kernel(const vector<double>& x1, const vector<double>& x2);
int predict_label(const SVMModel& model, const vector<double>& x, int d);
double k_fold_cv(
    vector<vector<double>>& X,
    vector<int>& y,
    int K,
    double C,
    int d,
    ofstream& result_file
);
bool load_data(const string& feature_file,
    const string& label_file,
    vector<vector<double>>& X,
    vector<int>& y
);
double linear_kernel_avx2(const double* x1, const double* x2, int d);
double rbf_kernel_avx2(const double* x1, const double* x2, int d);
vector<vector<double>> fisher_selection(
    const vector<vector<double>>& X,
    const vector<int>& y,
    int k // 保留前k个特征
) ;
vector<vector<double>> pca_reduce(
    const vector<vector<double>>& X,
    int k // 目标维度
);
struct PCA_Model {
    Eigen::RowVectorXd mean;  // 均值 (1 × d)
    Eigen::MatrixXd Vk;       // 主成分 (d × k)
};

//main
int main() {
    string BASE_DIR = "/mnt/sdb1/feiyang/datascience/lab1/Animals_with_Attributes2/Features/ResNet101/";
    int d = 2048;

    string train_feat_file = BASE_DIR + "train_features.txt";
    //string train_feat_file = BASE_DIR + "train_vae_features.txt";  //vae使用
    string train_label_file = BASE_DIR + "train_labels.txt";
    string test_feat_file  = BASE_DIR + "test_features.txt";
    //string test_feat_file  = BASE_DIR + "test_vae_features.txt";   //vae使用
    string test_label_file = BASE_DIR + "test_labels.txt"; 

    vector<vector<double>> X_train, X_test;
    vector<int> y_train, y_test;

    // ===== 1. 读取训练集 =====
    if (!load_data(train_feat_file, train_label_file, X_train, y_train)) {
        return 1;
    }

    cout << "Loaded training data: " << X_train.size()
         << " samples, " << X_train[0].size() << " features." << endl;

    // ===== 2. 读取测试集 =====
    if (!load_data(test_feat_file, test_label_file, X_test, y_test)) {
        return 1;
    }

    cout << "Loaded test data: " << X_test.size()
         << " samples, " << X_test[0].size() << " features." << endl;
    /* 
    // ===== 3. 正确 PCA（fit + transform）=====
    int target_dim = 512;

    cout << "Running PCA..." << endl;

    //  1. 用训练集 fit
    PCA_Model pca = pca_fit(X_train, target_dim);

    // 2. transform 训练集
    X_train = pca_transform(X_train, pca);

    // 3. transform 测试集（关键！）
    X_test = pca_transform(X_test, pca);

    d = target_dim;

    cout << "After PCA: " << d << " dimensions." << endl;
    */
    // ===== 4. 训练 One-vs-Rest SVM =====
    const int num_classes = 50;
    double C = 1.0;

    vector<SVMModel> models(num_classes);
    vector<thread> threads;

    cout << "Training SVM..." << endl;

    for (int cls = 1; cls <= num_classes; cls++) {
        threads.emplace_back([&, cls]() {
            vector<int> y_bin(y_train.size());
            for (int i = 0; i < y_train.size(); i++)
                y_bin[i] = (y_train[i] == cls) ? 1 : -1;

            models[cls - 1] = train_svm(X_train, y_bin, C, d);
        });
    }

    for (auto& t : threads) t.join();

    cout << "Training finished." << endl;

    // ===== 5. 测试 =====
    int correct = 0;

    for (int i = 0; i < X_test.size(); i++) {
        int best_cls = 1;
        double best_score = -1e9;

        for (int cls = 0; cls < num_classes; cls++) {
            const SVMModel& m = models[cls];

            double score = predict(
                m.X, m.y, m.alpha, m.b,
                X_test[i], d
            );

            if (score > best_score) {
                best_score = score;
                best_cls = cls + 1;
            }
        }

        if (best_cls == y_test[i]) correct++;
    }

    double acc = (double)correct / X_test.size();

    cout << "Test Accuracy: " << acc << endl;

    return 0;
}

SVMModel train_svm(
    vector<vector<double>>& X,
    vector<int>& y,
    double C,
    int d
) {
    int n = X.size();

    vector<double> alpha(n, 0.0);
    double b = 0.0;

    // 误差缓存（核心）
    vector<double> E(n);
    for (int i = 0; i < n; i++) {
        E[i] = -y[i];  // 初始 f(x)=0
    }

    int max_iter = 50;
    bool examine_all = true;
    int iter = 0;

    while (iter < max_iter) {
        int num_changed = 0;

        for (int i = 0; i < n; i++) {

            // support vector 优先扫描
            if (!examine_all) {
                if (!(alpha[i] > 0 && alpha[i] < C)) continue;
            }

            double Ei = E[i];

            // KKT 条件
            if ((y[i]*Ei < -1e-8 && alpha[i] < C) ||
                (y[i]*Ei >  1e-8 && alpha[i] > 0)) {

                // 选择 j（最大 |Ei - Ej|）
                int j = -1;
                double max_diff = 0;
                for (int k = 0; k < n; k++) {
                    if (k == i) continue;
                    double diff = fabs(Ei - E[k]);
                    if (diff > max_diff) {
                        max_diff = diff;
                        j = k;
                    }
                }
                if (j == -1) continue;

                double Ej = E[j];

                double ai_old = alpha[i];
                double aj_old = alpha[j];

                // 计算 L / H
                double L, H;
                if (y[i] != y[j]) {
                    L = max(0.0, aj_old - ai_old);
                    H = min(C, C + aj_old - ai_old);
                } else {
                    L = max(0.0, ai_old + aj_old - C);
                    H = min(C, ai_old + aj_old);
                }
                if (L == H) continue;

                // kernel（只算必要的）
                double Kii = linear_kernel_avx2(X[i].data(), X[i].data(), d);
                double Kjj = linear_kernel_avx2(X[j].data(), X[j].data(), d);
                double Kij = linear_kernel_avx2(X[i].data(), X[j].data(), d);

                double eta = Kii + Kjj - 2 * Kij;
                if (eta <= 1e-10) continue;  // + 1e-12 * (Kii + Kjj)

                // 更新 alpha[j]
                double new_aj = aj_old + y[j] * (Ei - Ej) / eta;
                new_aj = min(H, max(L, new_aj));

                double diff = fabs(new_aj - aj_old);
                if (diff < 1e-10) //+ 1e-8 * (fabs(new_aj) + fabs(aj_old))
                    continue;
                alpha[j] = new_aj;

                // 更新 alpha[i]
                alpha[i] += y[i]*y[j]*(aj_old - alpha[j]);
                alpha[i] = min(C, max(0.0, alpha[i]));

                // 更新 b
                double b_old = b;

                double b1 = b - Ei
                    - y[i]*(alpha[i]-ai_old)*Kii
                    - y[j]*(alpha[j]-aj_old)*Kij;

                double b2 = b - Ej
                    - y[i]*(alpha[i]-ai_old)*Kij
                    - y[j]*(alpha[j]-aj_old)*Kjj;

                if (alpha[i] > 0 && alpha[i] < C) b = b1;
                else if (alpha[j] > 0 && alpha[j] < C) b = b2;
                else b = (b1 + b2) / 2;

                // 核心优化：增量更新 E（避免 predict）
                for (int k = 0; k < n; k++) {
                    double Kik = linear_kernel_avx2(X[i].data(), X[k].data(), d);
                    double Kjk = linear_kernel_avx2(X[j].data(), X[k].data(), d);

                    E[k] += y[i]*(alpha[i]-ai_old)*Kik
                          + y[j]*(alpha[j]-aj_old)*Kjk
                          + (b - b_old);
                }

                num_changed++;
            }
        }
        
        if (iter % 5 == 0){
            cout << "C=" << C
             << " iter=" << iter
             << " changed=" << num_changed
             << " examine_all=" << examine_all << endl;
        }
        if (examine_all)
            examine_all = false;
        else if (num_changed == 0)
            examine_all = true;

        if (!examine_all && num_changed == 0){
            cout<<"train termiantes at iter = "<<iter<<endl;
            break;
        }
            

        iter++;
    }

    return {X, y, alpha, b};
}

double predict(
    const vector<vector<double>>& X,
    const vector<int>& y,
    const vector<double>& alpha,
    double b,
    const vector<double>& x,
    int d
) {
    double sum = 0;
    for (int i = 0; i < X.size(); i++) {
        if (alpha[i] > 0) {
            sum += alpha[i] * y[i] * linear_kernel_avx2(X[i].data(), x.data(), d);
        }
    }
    return sum + b;
}

double kernel(const vector<double>& x1, const vector<double>& x2) {
    // RBF kernel
    double sum = 0;
    for (int i = 0; i < x1.size(); i++) {
        double d = x1[i] - x2[i];
        sum += d * d;
    }
    return exp(-0.05 * sum);  // gamma = 0.05
}

int predict_label(const SVMModel& model, const vector<double>& x, int d) {
    double sum = 0;
    for (int i = 0; i < model.X.size(); i++) {
        if (model.alpha[i] > 0) {
            sum += model.alpha[i] * model.y[i] *
                linear_kernel_avx2(model.X[i].data(), x.data(),d);
        }
    }
    sum += model.b;
    return sum >= 0 ? 1 : -1;
}

double k_fold_cv(
    vector<vector<double>>& X,
    vector<int>& y,
    int K,
    double C,
    int d,
    ofstream& result_file   // 新增参数
) {
    int n = X.size();
    vector<int> indices(n);
    for (int i = 0; i < n; i++) indices[i] = i;

    int fold_size = n / K;
    double total_acc = 0;
    const int num_classes = 50;

    for (int k = 0; k < K; k++) {
        cout << "current fold: " << k << endl;

        vector<vector<double>> X_train, X_test;
        vector<int> y_train, y_test;

        int start = k * fold_size;
        int end = (k == K-1) ? n : start + fold_size;

        for (int i = 0; i < n; i++) {
            if (i >= start && i < end) {
                X_test.push_back(X[indices[i]]);
                y_test.push_back(y[indices[i]]);
            } else {
                X_train.push_back(X[indices[i]]);
                y_train.push_back(y[indices[i]]);
            }
        }

        // One-vs-Rest: 并发训练 50 个 SVM
        vector<SVMModel> models(num_classes);
        vector<thread> threads;

        for (int cls = 1; cls <= num_classes; cls++) {
            threads.push_back(thread([&, cls]() {
                vector<int> y_bin(y_train.size());
                for (int i = 0; i < y_train.size(); i++)
                    y_bin[i] = (y_train[i] == cls) ? 1 : -1;

                models[cls-1] = train_svm(X_train, y_bin, C, d);
            }));
        }

        for (auto& t : threads) t.join();

        int correct = 0;
        for (int i = 0; i < X_test.size(); i++) {
            int best_cls = 1;
            double best_score = -1e9;

            for (int cls = 0; cls < num_classes; cls++) {
                const SVMModel& m = models[cls];
                double score = predict(m.X, m.y, m.alpha, m.b, X_test[i], d);
                if (score > best_score) {
                    best_score = score;
                    best_cls = cls + 1;
                }
            }

            if (best_cls == y_test[i]) correct++;
        }

        double acc = (double)correct / X_test.size();
        total_acc += acc;

        // 写入每个 fold 的准确率
        cout << "C = " << C << ", Fold " << (k+1) << " accuracy: " << acc << endl;
        result_file  << "C = " << C << ", Fold " << (k+1) << " accuracy: " << acc << endl;
    }

    double avg_acc = total_acc / K;
    result_file << "Average accuracy for C=" << C << " : " << avg_acc << endl;
    result_file << "----------------------------------" << endl;
    return avg_acc;
}

bool load_data(const string& feature_file,
    const string& label_file,
    vector<vector<double>>& X,
    vector<int>& y) {
    ifstream fin_feat(feature_file);
    ifstream fin_label(label_file);
    if (!fin_feat || !fin_label) {
    cerr << "Error opening files!" << endl;
    return false;
    }

    string line;
    while (getline(fin_feat, line)) {
    stringstream ss(line);
    vector<double> vec;
    double val;
    while (ss >> val) vec.push_back(val);
    if (!vec.empty()) X.push_back(vec);
    }

    while (getline(fin_label, line)) {
    y.push_back(stoi(line));
    }

    fin_feat.close();
    fin_label.close();

    if (X.size() != y.size()) {
    cerr << "Data size mismatch!" << endl;
    return false;
    }

    return true;
}

// AVX2 RBF kernel for double precision
double rbf_kernel_avx2(const double* x1, const double* x2, int d) {
    __m256d sum_vec = _mm256_setzero_pd(); // 初始化向量累加器
    int i = 0;

    // 向量化循环，每次处理4个double
    for (; i + 3 < d; i += 4) {
        __m256d a = _mm256_loadu_pd(x1 + i);
        __m256d b = _mm256_loadu_pd(x2 + i);
        __m256d diff = _mm256_sub_pd(a, b);
        sum_vec = _mm256_add_pd(sum_vec, _mm256_mul_pd(diff, diff));
    }

    // 横向累加 sum_vec
    double tmp[4];
    _mm256_storeu_pd(tmp, sum_vec);
    double sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];

    // 处理尾部元素
    for (; i < d; ++i) {
        double diff = x1[i] - x2[i];
        sum += diff * diff;
    }

    return exp(-0.05 * sum); // gamma = 0.05
}

double linear_kernel_avx2(const double* x1, const double* x2, int d) {
    __m256d sum_vec = _mm256_setzero_pd(); // 初始化向量累加器
    int i = 0;

    // AVX2 向量化循环，每次处理4个 double
    for (; i + 3 < d; i += 4) {
        __m256d a = _mm256_loadu_pd(x1 + i);       // 加载 4 个 double
        __m256d b = _mm256_loadu_pd(x2 + i);
        sum_vec = _mm256_add_pd(sum_vec, _mm256_mul_pd(a, b)); // 累加乘积
    }

    // 横向累加寄存器 sum_vec
    double tmp[4];
    _mm256_storeu_pd(tmp, sum_vec);
    double sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];

    // 处理尾部元素
    for (; i < d; ++i) {
        sum += x1[i] * x2[i];
    }

    return sum;
}

vector<vector<double>> fisher_selection(
    const vector<vector<double>>& X,
    const vector<int>& y,
    int k // 保留前k个特征
) {
    int n = X.size();
    int d = X[0].size();

    vector<double> score(d, 0.0);

    // 找类别
    set<int> classes(y.begin(), y.end());

    for (int j = 0; j < d; j++) {
        double global_mean = 0;
        for (int i = 0; i < n; i++) {
            global_mean += X[i][j];
        }
        global_mean /= n;

        double num = 0, den = 0;

        for (int c : classes) {
            double mean_c = 0;
            int count = 0;

            for (int i = 0; i < n; i++) {
                if (y[i] == c) {
                    mean_c += X[i][j];
                    count++;
                }
            }

            if (count == 0) continue;
            mean_c /= count;

            num += count * (mean_c - global_mean) * (mean_c - global_mean);

            for (int i = 0; i < n; i++) {
                if (y[i] == c) {
                    double diff = X[i][j] - mean_c;
                    den += diff * diff;
                }
            }
        }

        score[j] = (den == 0) ? 0 : num / den;
    }

    // 排序选前k
    vector<int> idx(d);
    iota(idx.begin(), idx.end(), 0);

    sort(idx.begin(), idx.end(), [&](int a, int b) {
        return score[a] > score[b];
    });

    idx.resize(k);

    // 构造新X
    vector<vector<double>> X_new(n, vector<double>(k));
    for (int i = 0; i < n; i++) {
        for (int t = 0; t < k; t++) {
            X_new[i][t] = X[i][idx[t]];
        }
    }

    return X_new;
}

PCA_Model pca_fit(
    const vector<vector<double>>& X,
    int k
) {
    int n = X.size();
    int d = X[0].size();

    // ===== 转 Eigen =====
    Eigen::MatrixXd mat(n, d);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < d; j++)
            mat(i, j) = X[i][j];

    // ===== 中心化 =====
    Eigen::RowVectorXd mean = mat.colwise().mean();
    Eigen::MatrixXd centered = mat.rowwise() - mean;

    // ===== SVD =====
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(centered, Eigen::ComputeThinV);
    Eigen::MatrixXd V = svd.matrixV();  // d × d

    // ===== 取前 k 个主成分 =====
    Eigen::MatrixXd Vk = V.leftCols(k); // d × k

    return {mean, Vk};
}

vector<vector<double>> pca_transform(
    const vector<vector<double>>& X,
    const PCA_Model& model
) {
    int n = X.size();
    int d = X[0].size();
    int k = model.Vk.cols();

    // ===== 转 Eigen =====
    Eigen::MatrixXd mat(n, d);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < d; j++)
            mat(i, j) = X[i][j];

    // ===== 用训练集的 mean =====
    Eigen::MatrixXd centered = mat.rowwise() - model.mean;

    // ===== 投影 =====
    Eigen::MatrixXd reduced = centered * model.Vk;

    // ===== 转回 vector =====
    vector<vector<double>> X_new(n, vector<double>(k));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < k; j++)
            X_new[i][j] = reduced(i, j);

    return X_new;
}
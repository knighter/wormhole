/*!
 * \file kmeans.cc
 * \brief kmeans using rabit allreduce
 */
#include <algorithm>
#include <vector>
#include <cmath>
#include <rabit.h>
#include <dmlc/io.h>
#include <dmlc/data.h>
#include <dmlc/logging.h>

using namespace rabit;
using namespace dmlc;
/*!\brief computes a random number modulo the value */
inline int Random(int value) {
  return static_cast<int>(static_cast<double>(rand()) / RAND_MAX * value);
}

// simple dense matrix, mshadow or Eigen matrix was better
// use this to make a standalone example
struct Matrix {
  inline void Init(size_t nrow, size_t ncol, float v = 0.0f) {
    this->nrow = nrow;
    this->ncol = ncol;
    data.resize(nrow * ncol);
    std::fill(data.begin(), data.end(), v);
  }
  inline float *operator[](size_t i) {
    return &data[0] + i * ncol;
  }
  inline const float *operator[](size_t i) const {
    return &data[0] + i * ncol;
  }
  inline void Print(dmlc::Stream *fo) {
    dmlc::ostream os(fo);
    for (size_t i = 0; i < data.size(); ++i) {
      os << data[i];
      if ((i+1) % ncol == 0) {
        os << '\n';
      } else {
        os << ' ';
      }
    }
  }
  // number of data
  size_t nrow, ncol;
  std::vector<float> data;
};

// kmeans model
class Model : public dmlc::Serializable {
 public:
  // matrix of centroids
  Matrix centroids;
  // load from stream
  virtual void Load(dmlc::Stream *fi) {
    fi->Read(&centroids.nrow, sizeof(centroids.nrow));
    fi->Read(&centroids.ncol, sizeof(centroids.ncol));
    fi->Read(&centroids.data);
  }
  /*! \brief save the model to the stream */
  virtual void Save(dmlc::Stream *fo) const {
    fo->Write(&centroids.nrow, sizeof(centroids.nrow));
    fo->Write(&centroids.ncol, sizeof(centroids.ncol));
    fo->Write(centroids.data);
  }
  virtual void InitModel(unsigned num_cluster, unsigned feat_dim) {
    centroids.Init(num_cluster, feat_dim);
  }
  // normalize L2 norm
  inline void Normalize(void) {
    for (size_t i = 0; i < centroids.nrow; ++i) {
      float *row = centroids[i];
      double wsum = 0.0;
      for (size_t j = 0; j < centroids.ncol; ++j) {
        wsum += row[j] * row[j];
      }
      wsum = sqrt(wsum);
      if (wsum < 1e-6) return;
      float winv = 1.0 / wsum;
      for (size_t j = 0; j < centroids.ncol; ++j) {
        row[j] *= winv;
      }
    }
  }
};
// initialize the cluster centroids
inline void InitCentroids(dmlc::RowBlockIter<unsigned> *data,
                          Matrix *centroids) {
  data->BeforeFirst();
  CHECK(data->Next()) << "dataset is empty";
  // RowBlockIter的value为RowBlock
  const RowBlock<unsigned> &block = data->Value();
  int num_cluster = centroids->nrow;
  // 循环i in num_cluster次（聚类簇个数）
  for (int i = 0; i < num_cluster; ++i) {
    //  block.size为数据的个数。
    int index = Random(block.size); // 随机一个数据点
    Row<unsigned> v = block[index]; // v是这个数据点的value向量。
    for (unsigned j = 0; j < v.length; ++j) { // j是value中的一个下标。
      // https://github.com/dmlc/dmlc-core/blob/master/src/data/row_block.h
      // https://github.com/dmlc/dmlc-core/blob/master/include/dmlc/data.h Row的定义
      // v.index[j]是什么？v的第j个index，返回一个IndexType
      // 补充：v.index[j]返回的是特征所对应的index。因为libsvm的格式，是稀疏向量。
      // 将随机的数据点v的值赋给第i个聚类中心。一个聚类中心其实就是一个数据点。
      (*centroids)[i][v.index[j]] = v.get_value(j); 
    }
  }
  for (int i = 0; i < num_cluster; ++i) {
    int proc = Random(rabit::GetWorldSize());
    // 注意这里的broadcast，将随机某个进程proc的第i个聚类中心broadcast到其他节点。
    rabit::Broadcast((*centroids)[i], centroids->ncol * sizeof(float), proc);
  }
}
// calculate cosine distance
inline double Cos(const float *row,
                  const Row<unsigned> &v) {
  double rdot = 0.0, rnorm = 0.0; 
  for (unsigned i = 0; i < v.length; ++i) {
    const dmlc::real_t fv = v.get_value(i);
    rdot += row[v.index[i]] * fv;
    rnorm += fv * fv;
  }
  return rdot  / sqrt(rnorm);
}
// get cluster of a certain vector
inline size_t GetCluster(const Matrix &centroids,
                         const Row<unsigned> &v) {
  size_t imin = 0;
  double dmin = Cos(centroids[0], v);
  for (size_t k = 1; k < centroids.nrow; ++k) {
    double dist = Cos(centroids[k], v);
    if (dist > dmin) {
      dmin = dist; imin = k;
    }
  }
  return imin;
}
             
int main(int argc, char *argv[]) {
  if (argc < 5) {
    // intialize rabit engine
    rabit::Init(argc, argv);
    if (rabit::GetRank() == 0) {
      rabit::TrackerPrintf("Usage: <data_path> num_cluster max_iter <out_model>\n");
    }
    rabit::Finalize();
    return 0;
  }
  srand(0);    
  // set the parameters
  int num_cluster = atoi(argv[2]);
  int max_iter = atoi(argv[3]);
  // intialize rabit engine
  rabit::Init(argc, argv);
  
  //读数据：使用RowBlockIter，参数为数据文件地址，当前节点，总节点数，格式。
  RowBlockIter<index_t> *data
      = RowBlockIter<index_t>::Create
      (argv[1],
       rabit::GetRank(),
       rabit::GetWorldSize(),
       "libsvm");
  // load model，模型主要为一个matrix。
  Model model;  
  // 获取当前迭代次数
  int iter = rabit::LoadCheckPoint(&model);
  if (iter == 0) {
    // 若为0则初始化，
    size_t fdim = data->NumCol(); // 列数，即特征数
    rabit::Allreduce<op::Max>(&fdim, 1);
    model.InitModel(num_cluster, fdim); // 初始化matrix 
    InitCentroids(data, &model.centroids); // 随机初始化聚类中心
    model.Normalize();
  }
  const unsigned num_feat = static_cast<unsigned>(model.centroids.ncol); // 列数为特征数
  
  // matrix to store the result
  // 存储了每个数据点
  Matrix temp;
  for (int r = iter; r < max_iter; ++r) {
    temp.Init(num_cluster, num_feat + 1, 0.0f);
    auto lazy_get_centroid = [&]()
    {
      // lambda function used to calculate the data if necessary
      // this function may not be called when the result can be directly recovered
      // 不大懂这个语法。。。
      data->BeforeFirst();
      // 每个节点都取全部的数据？？  RowBlockIter的 BeforeFirst()，Next()，Value()。
      while (data->Next()) {
        const auto &batch = data->Value(); // 多个data，很可能是按batch_size切分。
        for (size_t i = 0; i < batch.size; ++i) {
          auto v = batch[i];
          size_t k = GetCluster(model.centroids, v); // 找到某个data的所属的聚类簇
          // temp[k] += v
          for (size_t j = 0; j < v.length; ++j) {
            temp[k][v.index[j]] += v.get_value(j); // 让temp的第k个聚类簇的数据加上v。
          }
          // use last column to record counts
          temp[k][num_feat] += 1.0f;
        }
      }
    };
    // 全局更新结果矩阵
    rabit::Allreduce<op::Sum>(&temp.data[0], temp.data.size(), lazy_get_centroid);
    // set number
    for (int k = 0; k < num_cluster; ++k) {
      float cnt = temp[k][num_feat];
      if (cnt != 0.0f) {        
        for (unsigned i = 0; i < num_feat; ++i) {
          model.centroids[k][i] = temp[k][i] / cnt; // 利用SUM/cnt取AVG，更新第k个聚类中心。
        }
      } else {
        rabit::TrackerPrintf("Error: found zero size cluster, maybe too less number of datapoints?\n");
        exit(-1);
      }
    }
    model.Normalize();
    rabit::LazyCheckPoint(&model);
    if (rabit::GetRank() == 0) {
      rabit::TrackerPrintf("Finish %d-th iteration\n", r);
    }
  }
  delete data;

  // output the model file to somewhere
  if (rabit::GetRank() == 0) {
    auto *fo = Stream::Create(argv[4], "w");
    model.centroids.Print(fo);
    delete fo;
    rabit::TrackerPrintf("All iteration finished, centroids saved to %s\n", argv[4]);
  }
  rabit::Finalize();
  return 0;
}

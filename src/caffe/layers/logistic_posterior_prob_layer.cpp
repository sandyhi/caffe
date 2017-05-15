#include <cmath>
#include <vector>

#include "caffe/layers/logistic_posterior_prob_layer.hpp"

namespace caffe {

template <typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return 1. / (1. + exp(-x));
}

template <typename Dtype>
void LogisticPosteriorProbLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  if (bottom.size() != 2) { return; }
  const Dtype* bottom_data_first = bottom[0]->cpu_data();
  const Dtype* bottom_data_second = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count_first = bottom[0]->count();
  const int count_second = bottom[1]->count();
  if (count_first != count_second) { return; }
  for (int i = 0; i < count_first; ++i) {
    Dtype temp_val = bottom_data_first[i] - bottom_data_second[i] 
    top_data[i] = sigmoid(temp_val);
  }
}

template <typename Dtype>
void LogisticPosteriorProbLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (bottom.size() == 1) {return ;}
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  for (int i = 0; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
      Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
      const int count = bottom[i]->count();
      for (int j = 0; j < count; ++j) {
        const Dtype sigmoid_x = top_data[j];
        bottom_diff[j] = top_diff[j] * sigmoid_x * (1. - sigmoid_x);
      }
    }
  }  
}

#ifdef CPU_ONLY
STUB_GPU(LogisticPosteriorProbLayer);
#endif

INSTANTIATE_CLASS(LogisticPosteriorProbLayer);
REGISTER_LAYER_CLASS(LogisticPosteriorProb);

}  // namespace caffe

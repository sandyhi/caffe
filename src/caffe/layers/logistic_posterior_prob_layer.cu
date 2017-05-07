#include <cmath>
#include <vector>

#include "caffe/layers/logistic_posterior_prob_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SigmoidForward(const int n, const Dtype* in_first, 
    const Dtype* in_second, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype temp_val = in_first[index] - in_second[index]
    out[index] = 1. / (1. + exp(-temp_val));
  }
}

template <typename Dtype>
void LogisticPosteriorProbLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  if (bottom.size() != 2) { return; }
  const Dtype* bottom_data_first = bottom[0]->gpu_data();
  const Dtype* bottom_data_second = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  SigmoidForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data_first, bottom_data_second, top_data);
  CUDA_POST_KERNEL_CHECK;
  // << " count: " << count << " bottom_data: "
  //     << (unsigned long)bottom_data
  //     << " top_data: " << (unsigned long)top_data
  //     << " blocks: " << CAFFE_GET_BLOCKS(count)
  //     << " threads: " << CAFFE_CUDA_NUM_THREADS;
}

template <typename Dtype>
__global__ void SigmoidBackward(const int n, const Dtype* in_diff,
    const Dtype* out_data, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    const Dtype sigmoid_x = out_data[index];
    out_diff[index] = in_diff[index] * sigmoid_x * (1 - sigmoid_x);
  }
}

template <typename Dtype>
void LogisticPosteriorProbLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (bottom.size() != 2) { return; }
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  for (int i = 0; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      const int count = bottom[i]->count();
      // NOLINT_NEXT_LINE(whitespace/operators)
      SigmoidBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, top_diff, top_data, bottom_diff);
      CUDA_POST_KERNEL_CHECK;
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(LogisticPosteriorProbLayer);


}  // namespace caffe

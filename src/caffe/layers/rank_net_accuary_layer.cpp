#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/rank_net_accuracy_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void RankNetAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  has_ignore_label_ =
    this->layer_param_.accuracy_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.accuracy_param().ignore_label();
  }
}

template <typename Dtype>
void RankNetAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->count(), bottom[1]->count())
      << "pred prob number shoud be equal to true label number.";
  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void RankNetAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype accuracy = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  //const int dim = bottom[0]->count() / outer_num_;
  int count = 0;
  int batch_num = bottom[0]->count();
  for (int i = 0; i < batch_num; ++i) {
    float pred_prob = static_cast<float>(bottom_data[i]);
    float ture_label = static_cast<float>(bottom_label[i]);
    DCHECK_GE(pred_prob, 0);
    if (((pred_prob - 0.5) * (ture_label - 0.5)) > 0 || 
        ((pred_prob - 0.5) + (ture_label - 0.5)) == 0) {
      ++accuracy;
    }  
    ++count;
  }

  // LOG(INFO) << "Accuracy: " << accuracy;
  top[0]->mutable_cpu_data()[0] = accuracy / count;
  // Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(RankNetAccuracyLayer);
REGISTER_LAYER_CLASS(RankNetAccuracy);

}  // namespace caffe

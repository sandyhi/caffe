//
// This script converts the image pair to the leveldb format 

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "stdint.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/format.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#ifdef USE_LEVELDB
#include "leveldb/db.h"


DEFINE_bool(shuffle, false,
    "Randomly shuffle the order of images and their labels");

bool read_image(const std::string& image_file_first, const std::string& image_file_second, 
        char* pixels) {
  std::streampos size_first;
  std::streampos size_second;

  std::fstream file_first(image_file_first.c_str(), std::ios::in|std::ios::binary|std::ios::ate);
  if (file_first.is_open()) {
    size_first = file_first.tellg();
    file_first.seekg(0, std::ios::beg);
    file_first.read(pixels, size_first);
    file_first.close();
  } else {
    return false;
  }

  std::fstream file_second(image_file_second.c_str(), std::ios::in|std::ios::binary|std::ios::ate);
  if (file_second.is_open()) { 
    size_second = file_second.tellg();
    file_second.seekg(0, std::ios::beg);
    file_second.read(pixels + size_first, size_second);
    file_second.close();
  } else {
    return false;
  }
  
  return true;
}

void convert_dataset(const std::string& image_filename, const char* db_filename) {
  // Open files
  std::ifstream infile(image_filename.c_str());
  CHECK(infile) << "Unable to open file " << image_filename;
  std::vector<std::pair<std::vector<std::string>, float> > lines;
  
  std::vector<std::string> image_pair_vec; 
  std::string filename_first, filename_second;
  float label;
  while (infile >> filename_first >> filename_second >> label) {
    if (!image_pair_vec.empty()) {
      image_pair_vec.clear();
    }
    image_pair_vec.push_back(filename_first);
    image_pair_vec.push_back(filename_second);
    lines.push_back(std::make_pair(image_pair_vec, label));
  }
  if (FLAGS_shuffle) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    caffe::shuffle(lines.begin(), lines.end());
  }
  LOG(INFO) << "A total of " << lines.size() << " image-pairs.";

  // get image size
  std::string tmp_img_name = image_pair_vec[0]; 
  cv::Mat cv_img_origin = cv::imread(tmp_img_name);
  int rows = cv_img_origin.rows;
  int cols = cv_img_origin.cols;
  int channels = cv_img_origin.channels();
  CHECK_EQ(2 * channels, 6) << "Illegal image channel !";

  // Open leveldb
  leveldb::DB* db;
  leveldb::Options options;
  options.create_if_missing = true;
  options.error_if_exists = true;
  leveldb::Status status = leveldb::DB::Open(
      options, db_filename, &db);
  CHECK(status.ok()) << "Failed to open leveldb " << db_filename
      << ". Is it already existing?";

  char* pixels = new char[6 * rows * cols];
  std::string value;

  caffe::Datum datum;
  datum.set_channels(6);  // one channel for each image in the pair
  datum.set_height(rows);
  datum.set_width(cols);
  int num_items = lines.size();
  LOG(INFO) << "A total of " << num_items << " items.";
  LOG(INFO) << "Rows: " << rows << " Cols: " << cols;
  for (int line_id = 0; line_id < num_items; ++line_id) {
    std::string image_file_first = lines[line_id].first[0];
    std::string image_file_second = lines[line_id].first[1];
    label = lines[line_id].second;
    read_image(image_file_first, image_file_second, 
        pixels);
    datum.set_data(pixels, 6*rows*cols);
    datum.set_label(label);
    datum.SerializeToString(&value);
    std::string key_str = caffe::format_int(line_id, 8);
    db->Put(leveldb::WriteOptions(), key_str, value);
  }

  delete db;
  delete [] pixels;
}

int main(int argc, char** argv) {
  #ifndef GFLAGS_GFLAGS_H_
    namespace gflags = google;
  #endif
  if (argc != 3) {
    printf("Convert a set of images to the leveldb/lmdb\n"
        "format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_image_pair_data [FLAGS] input_image_file output_db_file\n");
    gflags::ParseCommandLineFlags(&argc, &argv, true);
  } else {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_image_pair_data");
    google::InitGoogleLogging(argv[0]);
    convert_dataset(argv[1], argv[2]);
  }
  return 0;
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires LevelDB; compile with USE_LEVELDB.";
}
#endif  // USE_LEVELDB

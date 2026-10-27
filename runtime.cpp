#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct Layer { std::uint32_t rows, cols; std::vector<std::int8_t> weights; std::vector<float> scales, bias; };
std::uint32_t u32(std::istream& in) {
  unsigned char bytes[4]; if (!in.read(reinterpret_cast<char*>(bytes), 4)) throw std::runtime_error("truncated artifact");
  return std::uint32_t(bytes[0]) | (std::uint32_t(bytes[1])<<8) | (std::uint32_t(bytes[2])<<16) | (std::uint32_t(bytes[3])<<24);
}
float f32(std::istream& in) {
  static_assert(sizeof(float)==4, "requires 32-bit float");
  const auto bits=u32(in); float value; std::memcpy(&value,&bits,4);
  if (!std::isfinite(value)) throw std::runtime_error("nonfinite artifact value"); return value;
}
void magic(std::istream& in,const std::string& expected) {
  char bytes[4]; if (!in.read(bytes,4) || std::string(bytes,4)!=expected) throw std::runtime_error("invalid artifact header");
}
std::vector<Layer> load_model(const char* path) {
  std::ifstream in(path,std::ios::binary); magic(in,"ELQ1"); const auto count=u32(in);
  if (count==0 || count>16) throw std::runtime_error("invalid layer count");
  std::vector<Layer> layers;
  for (std::uint32_t i=0;i<count;++i) {
    Layer layer; layer.rows=u32(in); layer.cols=u32(in);
    if (!layer.rows || !layer.cols || layer.rows>4096 || layer.cols>4096 || (!layers.empty() && layer.cols!=layers.back().rows)) throw std::runtime_error("invalid layer dimensions");
    layer.weights.resize(std::size_t(layer.rows)*layer.cols);
    if (!in.read(reinterpret_cast<char*>(layer.weights.data()),static_cast<std::streamsize>(layer.weights.size()))) throw std::runtime_error("truncated weights");
    for (std::uint32_t j=0;j<layer.rows;++j) { const auto scale=f32(in); if (scale<=0) throw std::runtime_error("scale must be positive"); layer.scales.push_back(scale); }
    for (std::uint32_t j=0;j<layer.rows;++j) layer.bias.push_back(f32(in));
    layers.push_back(std::move(layer));
  }
  if (in.peek()!=std::char_traits<char>::eof()) throw std::runtime_error("trailing model bytes");
  return layers;
}
std::vector<float> infer(const std::vector<Layer>& layers,std::vector<float> input) {
  for (std::size_t index=0;index<layers.size();++index) {
    const auto& layer=layers[index]; if (input.size()!=layer.cols) throw std::runtime_error("input dimension mismatch");
    std::vector<float> output(layer.rows);
    for (std::uint32_t row=0;row<layer.rows;++row) {
      float sum=0;
      for (std::uint32_t col=0;col<layer.cols;++col) sum += input[col]*static_cast<float>(layer.weights[std::size_t(row)*layer.cols+col]);
      output[row]=sum*layer.scales[row]+layer.bias[row];
      if (index+1<layers.size()) output[row]=std::max(0.0f,output[row]);
      if (!std::isfinite(output[row])) throw std::runtime_error("nonfinite inference output");
    }
    input=std::move(output);
  }
  return input;
}
int main(int argc,char** argv) {
  try {
    if (argc!=3) throw std::runtime_error("usage: edge_runtime model.elq samples.els");
    const auto layers=load_model(argv[1]); std::ifstream input(argv[2],std::ios::binary); magic(input,"ELS1");
    const auto rows=u32(input),cols=u32(input);
    if (!rows || rows>10000 || cols!=layers.front().cols) throw std::runtime_error("invalid sample dimensions");
    std::vector<std::vector<float>> samples(rows,std::vector<float>(cols));
    for (auto& sample:samples) for (auto& value:sample) value=f32(input);
    if (input.peek()!=std::char_traits<char>::eof()) throw std::runtime_error("trailing sample bytes");
    std::vector<std::vector<float>> outputs;
    for (const auto& sample:samples) outputs.push_back(infer(layers,sample));
    volatile float sink=0; std::vector<double> times;
    for (int repeat=-3;repeat<9;++repeat) {
      const auto start=std::chrono::steady_clock::now();
      for (const auto& sample:samples) sink=infer(layers,sample)[0];
      const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/rows;
      if (repeat>=0) times.push_back(ms);
    }
    (void)sink; std::sort(times.begin(),times.end());
    std::cout<<std::setprecision(9)<<"{\"batch_amortized_ms_per_image\":"<<times[times.size()/2]<<",\"predictions\":[";
    for (std::size_t i=0;i<outputs.size();++i) std::cout<<(i?",":"")<<std::distance(outputs[i].begin(),std::max_element(outputs[i].begin(),outputs[i].end()));
    std::cout<<"],\"logits\":[";
    for (std::size_t i=0;i<outputs.size();++i) { std::cout<<(i?",":"")<<"["; for(std::size_t j=0;j<outputs[i].size();++j) std::cout<<(j?",":"")<<outputs[i][j]; std::cout<<"]"; }
    std::cout<<"]}\n"; return 0;
  } catch(const std::exception& error) { std::cerr<<"edge_runtime: "<<error.what()<<"\n"; return 1; }
}

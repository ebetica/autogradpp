#pragma once

#include <fstream>

#include "cereal/archives/binary.hpp"
#include "cereal/types/polymorphic.hpp"

#include "cereal/types/string.hpp"
#include "cereal/types/unordered_map.hpp"
#include "cereal/types/vector.hpp"

namespace autograd {

// Some convenience functions for saving and loading
template <typename T>
void save(std::ostream& stream, T const& obj) {
  cereal::BinaryOutputArchive archive(stream);
  archive(*obj);
}
template <typename T>
void load(std::istream& stream, T& obj) {
  cereal::BinaryInputArchive archive(stream);
  archive(*obj);
}
template <typename T>
void save(std::ostream& stream, T const* obj) {
  cereal::BinaryOutputArchive archive(stream);
  archive(*obj);
}
template <typename T>
void load(std::istream& stream, T* obj) {
  cereal::BinaryInputArchive archive(stream);
  archive(*obj);
}
template <typename T>
void save(std::string const& path, T const& obj) {
  std::ofstream os(path, std::ios::binary);
  autograd::save(os, obj);
}
template <typename T>
void load(std::string const& path, T& obj) {
  std::ifstream is(path, std::ios::binary);
  autograd::load(is, obj);
}

} // namespace autograd

// This is super ugly and I don't know how to simplify it
CEREAL_REGISTER_TYPE(autograd::SGD);
CEREAL_REGISTER_POLYMORPHIC_RELATION(autograd::OptimizerImpl, autograd::SGD);
CEREAL_REGISTER_TYPE(autograd::Adagrad);
CEREAL_REGISTER_POLYMORPHIC_RELATION(
    autograd::OptimizerImpl,
    autograd::Adagrad);
CEREAL_REGISTER_TYPE(autograd::RMSprop);
CEREAL_REGISTER_POLYMORPHIC_RELATION(
    autograd::OptimizerImpl,
    autograd::RMSprop);
CEREAL_REGISTER_TYPE(autograd::Adam);
CEREAL_REGISTER_POLYMORPHIC_RELATION(autograd::OptimizerImpl, autograd::Adam);

namespace cereal {

namespace agimpl {

template <class Archive>
void saveBinary(Archive& archive, void const* data, std::size_t size) {
  // In general, there's no direct `saveBinary`-like method on archives
  std::vector<char> v(
      reinterpret_cast<char const*>(data),
      reinterpret_cast<char const*>(data) + size);
  archive(v);
}
template <>
inline void
saveBinary(BinaryOutputArchive& archive, void const* data, std::size_t size) {
  // Writes to output stream without extra copy
  archive.saveBinary(data, size);
}

template <class Archive>
void loadBinary(Archive& archive, void* data, std::size_t size) {
  // In general, there's no direct `loadBinary`-like method on archives
  std::vector<char> v(size);
  archive(v);
  std::memcpy(data, v.data(), size);
}
template <>
inline void
loadBinary(BinaryInputArchive& archive, void* data, std::size_t size) {
  // Read from input stream without extra copy
  archive.loadBinary(data, size);
}

} // namespace agimpl

// Gradients will not be saved for variables
template <class Archive>
void save(Archive& archive, at::Tensor const& tensor) {
  if (!tensor.defined()) {
    auto type = at::ScalarType::Undefined;
    archive(CEREAL_NVP(type));
    return;
  } else {
    auto type = tensor.type().scalarType();
    archive(CEREAL_NVP(type));
  }
  auto sizes = std::vector<int64_t>();
  auto buf = std::vector<uint8_t>();
  for (auto s : tensor.sizes()) {
    sizes.push_back(s);
  }
  auto contig = tensor.toBackend(at::kCPU).contiguous();
  uint64_t size = tensor.numel() * tensor.type().elementSizeInBytes();
  at::Backend backend = tensor.type().backend();

  archive(CEREAL_NVP(backend), CEREAL_NVP(sizes), CEREAL_NVP(size));
  agimpl::saveBinary(
      archive,
      contig.data_ptr(),
      size);
}

/**
 * We follow these rules for loading:
 * 1. If tensor is defined, and the same ScalarType as the saved tensor,
 *    then we simply copy the data into the tensor, with resizing.
 * 2. Otherwise, overwrite the provided tensor with the right type and backend
 **/
template <class Archive>
void load(Archive& archive, at::Tensor& tensor) {
  at::ScalarType type;
  archive(CEREAL_NVP(type));
  if (type == at::ScalarType::Undefined) {
    tensor = at::Tensor();
    return;
  }

  at::Backend backend;
  auto sizes = std::vector<int64_t>();
  auto buf = std::vector<uint8_t>();
  uint64_t size;
  archive(CEREAL_NVP(backend), CEREAL_NVP(sizes), CEREAL_NVP(size));

  if (!tensor.defined() || tensor.type().scalarType() != type) {
    tensor = at::getType(backend, type).tensor();
  }
  tensor.resize_(sizes);

  // Backward compatibility with older code that used to save "size" as the
  // total size of the underlying storage
  size = std::min(
      size, tensor.numel() * tensor.type().elementSizeInBytes());

  if (tensor.type().is_cuda()) {
    // should actually use cudamemcpy probably
    auto cputensor = at::CPU(tensor.type().scalarType()).tensor(sizes);
    agimpl::loadBinary(archive, cputensor.data_ptr(), size);
    tensor.copy_(cputensor);
  } else {
    agimpl::loadBinary(archive, tensor.data_ptr(), size);
  }
}

template <class Archive>
void load(Archive& archive, tag::Variable& var) {
  load(archive, var.data());
}

} // namespace cereal

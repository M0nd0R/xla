/* Copyright 2026 The OpenXLA Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/tools/split_proto/split_proto_cli_lib.h"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"
#include "riegeli/base/maker.h"
#include "riegeli/base/types.h"
#include "riegeli/bytes/reader.h"
#include "riegeli/bytes/string_reader.h"
#include "riegeli/bytes/string_writer.h"
#include "riegeli/bytes/writer.h"
#include "riegeli/messages/parse_message.h"
#include "riegeli/messages/serialize_message.h"
#include "riegeli/records/record_reader.h"
#include "xla/backends/gpu/codegen/kernels/custom_kernel.pb.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/service/cpu/executable.pb.h"
#include "xla/service/gpu/gpu_executable.pb.h"
#include "xla/stream_executor/abi/executable_abi_version.pb.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/tools/split_proto/split_proto_cli.pb.h"
#include "xla/tsl/util/fixed_option_set_flag.h"
#include "xla/util/split_proto/split_executable_and_options_writer.h"
#include "xla/util/split_proto/split_gpu_executable_writer.h"
#include "xla/util/split_proto/split_proto.pb.h"
#include "xla/util/split_proto/split_proto_reader.h"
#include "xla/xla_data.pb.h"

namespace xla::split_proto_cli {

static const FixedOptionSetFlagParser<ProtoFormat>& GetProtoFormatParser() {
  static const auto& parser = GetFixedOptionSetFlagParser<ProtoFormat>(
      {{"text", ProtoFormat::kText, "Text format (also textproto, pbtxt)"},
       {"textproto", ProtoFormat::kText, "Alias for text"},
       {"pbtxt", ProtoFormat::kText, "Alias for text"},
       {"binary", ProtoFormat::kBinary, "Binary format (also pb)"},
       {"pb", ProtoFormat::kBinary, "Alias for binary"},
       {"auto", ProtoFormat::kAuto, "Auto detect format"}},
      {/*allow_aliases=*/true});
  return parser;
}

bool AbslParseFlag(absl::string_view text, ProtoFormat* format,
                   std::string* error) {
  return GetProtoFormatParser().Parse(text, format, error);
}

std::string AbslUnparseFlag(ProtoFormat format) {
  return GetProtoFormatParser().Unparse(format);
}

namespace {

absl::Status ParseProto(riegeli::Reader& reader, ProtoFormat format,
                        google::protobuf::Message& message) {
  switch (format) {
    case ProtoFormat::kBinary: {
      RETURN_IF_ERROR(riegeli::ParseMessage(reader, message));
      break;
    }
    case ProtoFormat::kText: {
      bool parse_success = false;
      {
        riegeli::ReaderInputStream zero_copy_stream(&reader);
        parse_success = google::protobuf::TextFormat::Parse(&zero_copy_stream, &message);
      }
      if (!parse_success) {
        return absl::InvalidArgumentError("Failed to parse text proto");
      }
      break;
    }
    case ProtoFormat::kAuto: {
      riegeli::Position pos = reader.pos();
      bool parse_success = false;
      {
        riegeli::ReaderInputStream zero_copy_stream(&reader);
        parse_success = google::protobuf::TextFormat::Parse(&zero_copy_stream, &message);
      }
      if (parse_success) {
        return absl::OkStatus();
      }

      // Failed to parse as text, try binary.
      message.Clear();
      if (!reader.Seek(pos)) {
        return reader.status();
      }
      RETURN_IF_ERROR(riegeli::ParseMessage(reader, message));
      break;
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status Pack(std::unique_ptr<riegeli::Reader> reader,
                  std::unique_ptr<riegeli::Writer> writer,
                  const PackOptions& options) {
  LOG(INFO) << "Packing standard protobuf (type: " << options.proto_type
            << ") to split proto format";
  std::unique_ptr<google::protobuf::Message> message;
  if (options.proto_type == "xla.gpu.GpuExecutableProto") {
    message = std::make_unique<xla::gpu::GpuExecutableProto>();
  } else if (options.proto_type == "xla.ExecutableAndOptionsProto") {
    message = std::make_unique<xla::ExecutableAndOptionsProto>();
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported proto type: ", options.proto_type));
  }

  RETURN_IF_ERROR(ParseProto(*reader, options.input_format, *message));

  if (options.proto_type == "xla.gpu.GpuExecutableProto") {
    auto* gpu_proto =
        absl::down_cast<xla::gpu::GpuExecutableProto*>(message.get());
    return WriteSplitGpuExecutable(std::move(*gpu_proto), std::move(writer));
  }

  if (options.proto_type == "xla.ExecutableAndOptionsProto") {
    auto* options_proto =
        absl::down_cast<xla::ExecutableAndOptionsProto*>(message.get());
    return WriteSplitExecutableAndOptions(*options_proto, std::move(writer));
  }

  return absl::InvalidArgumentError("Unreachable");
}

namespace {
absl::StatusOr<SplitProtoManifest> ReadManifest(riegeli::Reader& reader) {
  SplitProtoManifest manifest;
  riegeli::Position initial_pos = reader.pos();
  {
    riegeli::RecordReader record_reader(&reader);
    if (!record_reader.ReadRecord(manifest)) {
      return record_reader.status().ok()
                 ? absl::InvalidArgumentError(
                       "Failed to read manifest; input "
                       "is likely not a split proto file")
                 : record_reader.status();
    }
  }
  // Reset the reader to the initial position, so that the caller can read the
  // proto from the beginning.
  if (!reader.Seek(initial_pos)) {
    return reader.status();
  }
  return manifest;
}

absl::Status SerializeProto(const google::protobuf::Message& message, ProtoFormat format,
                            riegeli::Writer& writer) {
  if (format == ProtoFormat::kText) {
    bool print_success = false;
    {
      riegeli::WriterOutputStream zero_copy_stream(&writer);
      print_success = google::protobuf::TextFormat::Print(message, &zero_copy_stream);
    }
    if (!writer.ok()) {
      return writer.status();
    }
    if (!print_success) {
      return absl::InternalError("Failed to serialize text proto");
    }
  } else {
    RETURN_IF_ERROR(riegeli::SerializeMessage(message, writer));
  }
  return absl::OkStatus();
}
}  // namespace

absl::Status Unpack(std::unique_ptr<riegeli::Reader> reader,
                    std::unique_ptr<riegeli::Writer> writer,
                    const UnpackOptions& options) {
  ASSIGN_OR_RETURN(SplitProtoManifest manifest, ReadManifest(*reader));
  LOG(INFO) << "Unpacking split proto format (type: "
            << manifest.result_proto_type() << ") to standard protobuf ("
            << AbslUnparseFlag(options.output_format) << " format)";

  std::unique_ptr<google::protobuf::Message> message;
  if (manifest.result_proto_type() == "xla.gpu.GpuExecutableProto") {
    message = std::make_unique<xla::gpu::GpuExecutableProto>();
  } else if (manifest.result_proto_type() == "xla.ExecutableAndOptionsProto") {
    message = std::make_unique<xla::ExecutableAndOptionsProto>();
  } else {
    return absl::InvalidArgumentError(absl::StrCat(
        "Unsupported proto type in manifest: ", manifest.result_proto_type()));
  }

  RETURN_IF_ERROR(ReadSplitProto(std::move(reader), *message));

  RETURN_IF_ERROR(SerializeProto(*message, options.output_format, *writer));

  if (!writer->Close()) {
    return writer->status();
  }

  return absl::OkStatus();
}

absl::Status PackAot(std::unique_ptr<riegeli::Reader> reader,
                     std::unique_ptr<riegeli::Writer> writer,
                     const PackOptions& options) {
  LOG(INFO) << "Packing DeserializedSplitExecutableAndOptions to split "
               "ExecutableAndOptionsProto";
  DeserializedSplitExecutableAndOptions wrapper;
  RETURN_IF_ERROR(ParseProto(*reader, options.input_format, wrapper));

  std::unique_ptr<riegeli::Writer> serialized_executable_writer =
      riegeli::Maker<riegeli::StringWriter>(
          wrapper.mutable_executable_and_options()
              ->mutable_serialized_executable());
  RETURN_IF_ERROR(
      WriteSplitGpuExecutable(std::move(*wrapper.mutable_gpu_executable()),
                              std::move(serialized_executable_writer)));

  RETURN_IF_ERROR(WriteSplitExecutableAndOptions(
      *wrapper.mutable_executable_and_options(), std::move(writer)));

  return absl::OkStatus();
}

absl::Status UnpackAot(std::unique_ptr<riegeli::Reader> reader,
                       std::unique_ptr<riegeli::Writer> writer,
                       const UnpackOptions& options) {
  LOG(INFO) << "Unpacking split ExecutableAndOptionsProto to "
               "DeserializedSplitExecutableAndOptions ("
            << AbslUnparseFlag(options.output_format) << " format)";
  ExecutableAndOptionsProto executable_and_options_proto;
  RETURN_IF_ERROR(
      ReadSplitProto(std::move(reader), executable_and_options_proto));

  xla::gpu::GpuExecutableProto gpu_proto;
  RETURN_IF_ERROR(ReadSplitProto(
      riegeli::Maker<riegeli::StringReader>(
          executable_and_options_proto.mutable_serialized_executable()),
      gpu_proto));

  executable_and_options_proto.clear_serialized_executable();

  DeserializedSplitExecutableAndOptions wrapper;
  *wrapper.mutable_executable_and_options() =
      std::move(executable_and_options_proto);
  *wrapper.mutable_gpu_executable() = std::move(gpu_proto);

  RETURN_IF_ERROR(SerializeProto(wrapper, options.output_format, *writer));

  if (!writer->Close()) {
    return writer->status();
  }

  return absl::OkStatus();
}

namespace {

using ::xla::gpu::ThunkProto;
using ImplCase = ::xla::gpu::ThunkProto::ImplCase;

// Common thunks that are always present don't tell us anything meaningful
// about the model, so we skip them to reduce noise.
bool ShouldSkipThunk(ImplCase impl_case) {
  switch (impl_case) {
    case ThunkProto::kCopyThunk:
    case ThunkProto::kMemzeroThunk:
    case ThunkProto::kDeviceToDeviceCopyThunk:
    case ThunkProto::kHostToDeviceCopyThunk:
    case ThunkProto::kDeviceToHostCopyThunk:
    case ThunkProto::kMemset32BitValueThunk:
    case ThunkProto::kAsyncDoneThunk:
    case ThunkProto::kAsyncStartThunk:
      return true;
    default:
      return false;
  }
}

bool IsContainerThunk(ImplCase impl_case) {
  switch (impl_case) {
    case ThunkProto::kSequentialThunk:
    case ThunkProto::kWhileThunk:
    case ThunkProto::kConditionalThunk:
    case ThunkProto::kAsyncStartThunk:
    case ThunkProto::kCollectiveGroupThunk:
    case ThunkProto::kDynamicSliceThunk:
    case ThunkProto::kDynamicSliceFusionThunk:
      return true;
    default:
      return false;
  }
}

std::string StripTrailingNumbersAndUnderscores(absl::string_view name) {
  std::string result(name);
  while (!result.empty() &&
         (std::isdigit(static_cast<unsigned char>(result.back())) ||
          result.back() == '_' || result.back() == '.')) {
    result.pop_back();
  }
  return result;
}

std::string GetThunkName(const ThunkProto& thunk) {
  std::string name;
  switch (thunk.impl_case()) {
    case ThunkProto::kCustomCallThunk:
      name = thunk.custom_call_thunk().target_name();
      break;
    case ThunkProto::kKernelThunk:
      name = thunk.kernel_thunk().kernel_name();
      break;
    case ThunkProto::kCustomKernelThunk:
      name = thunk.custom_kernel_thunk().custom_kernel().name();
      break;
    default:
      return "";
  }
  return StripTrailingNumbersAndUnderscores(name);
}

void CollectThunkInfo(const ThunkProto& thunk,
                      absl::flat_hash_map<std::string, int>& thunk_counts);

void ProcessNestedThunks(const ThunkProto& thunk,
                         absl::flat_hash_map<std::string, int>& thunk_counts) {
  switch (thunk.impl_case()) {
    case ThunkProto::kSequentialThunk:
      for (const ThunkProto& inner_thunk : thunk.sequential_thunk().thunks()) {
        CollectThunkInfo(inner_thunk, thunk_counts);
      }
      break;
    case ThunkProto::kWhileThunk:
      if (thunk.while_thunk().has_condition_thunk_sequence()) {
        for (const ThunkProto& inner_thunk :
             thunk.while_thunk().condition_thunk_sequence().thunks()) {
          CollectThunkInfo(inner_thunk, thunk_counts);
        }
      }
      if (thunk.while_thunk().has_body_thunk_sequence()) {
        for (const ThunkProto& inner_thunk :
             thunk.while_thunk().body_thunk_sequence().thunks()) {
          CollectThunkInfo(inner_thunk, thunk_counts);
        }
      }
      break;
    case ThunkProto::kConditionalThunk:
      for (const xla::gpu::ThunkSequenceProto& inner_thunk :
           thunk.conditional_thunk().branch_thunks()) {
        for (const ThunkProto& inner_inner_thunk : inner_thunk.thunks()) {
          CollectThunkInfo(inner_inner_thunk, thunk_counts);
        }
      }
      break;
    case ThunkProto::kAsyncStartThunk:
      for (const ThunkProto& inner_thunk :
           thunk.async_start_thunk().thunks().thunks()) {
        CollectThunkInfo(inner_thunk, thunk_counts);
      }
      break;
    case ThunkProto::kCollectiveGroupThunk:
      for (const ThunkProto& inner_thunk :
           thunk.collective_group_thunk().thunks()) {
        CollectThunkInfo(inner_thunk, thunk_counts);
      }
      break;
    case ThunkProto::kDynamicSliceThunk:
      for (const ThunkProto& inner_thunk :
           thunk.dynamic_slice_thunk().embedded_thunk().thunks()) {
        CollectThunkInfo(inner_thunk, thunk_counts);
      }
      break;
    case ThunkProto::kDynamicSliceFusionThunk:
      for (const ThunkProto& inner_thunk :
           thunk.dynamic_slice_fusion_thunk().embedded_thunks().thunks()) {
        CollectThunkInfo(inner_thunk, thunk_counts);
      }
      break;
    default:
      break;
  }
}

std::string GetThunkInfoString(const ThunkProto& thunk) {
  const google::protobuf::FieldDescriptor* field_desc =
      thunk.GetReflection()->GetOneofFieldDescriptor(
          thunk, thunk.GetDescriptor()->FindOneofByName("impl"));
  std::string thunk_type =
      field_desc ? std::string(field_desc->name()) : "unknown_thunk";
  std::string thunk_name = GetThunkName(thunk);

  std::string info = thunk_type;
  if (!thunk_name.empty()) {
    absl::StrAppend(&info, " (", thunk_name, ")");
  }
  return info;
}

void CollectThunkInfo(const ThunkProto& thunk,
                      absl::flat_hash_map<std::string, int>& thunk_counts) {
  if (ShouldSkipThunk(thunk.impl_case())) {
    return;
  }

  std::string key = GetThunkInfoString(thunk);
  if (!IsContainerThunk(thunk.impl_case())) {
    thunk_counts[key]++;
  }

  ProcessNestedThunks(thunk, thunk_counts);
}
}  // namespace

absl::Status Info(std::unique_ptr<riegeli::Reader> reader) {
  xla::ExecutableAndOptionsProto executable_options;
  RETURN_IF_ERROR(ReadSplitProto(std::move(reader), executable_options));
  xla::gpu::GpuExecutableProto gpu_exec;
  RETURN_IF_ERROR(
      ReadSplitProto(riegeli::Maker<riegeli::StringReader>(
                         executable_options.serialized_executable()),
                     gpu_exec));
  std::cout << "=== Executable Information ===\n";
  std::cout << "Module Name: " << gpu_exec.module_name() << "\n";

  std::cout << "Target Compute Capability:\n"
            << gpu_exec.gpu_compute_capability().DebugString() << "\n";
  std::cout << "CPU Target Machine Options:\n"
            << (gpu_exec.has_cpu_target_machine_options()
                    ? gpu_exec.cpu_target_machine_options().DebugString()
                    : "not set")
            << "\n";
  std::cout << "Executable ABI version:\n "
            << gpu_exec.executable_abi_version().DebugString() << "\n";
  std::cout << "Target Config:\n"
            << (executable_options.has_compile_options() &&
                        executable_options.compile_options().has_target_config()
                    ? executable_options.compile_options()
                          .target_config()
                          .DebugString()
                    : "not set")
            << "\n";

  absl::flat_hash_map<std::string, int> thunk_counts;
  for (const ThunkProto& thunk : gpu_exec.thunks()) {
    CollectThunkInfo(thunk, thunk_counts);
  }

  std::vector<std::pair<std::string, int>> sorted_counts(thunk_counts.begin(),
                                                         thunk_counts.end());
  std::sort(sorted_counts.begin(), sorted_counts.end(),
            [](const auto& a, const auto& b) {
              return a.second > b.second;  // Descending order
            });

  std::cout << "\n=== Thunk Summary ===\n";
  for (const auto& [name, count] : sorted_counts) {
    std::cout << count << "x " << name << "\n";
  }

  return absl::OkStatus();
}

}  // namespace xla::split_proto_cli

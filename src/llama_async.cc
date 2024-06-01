// Copyright 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <dirent.h>
#include <getopt.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <condition_variable>
#include <fstream>
#include <iostream>
#include <iterator>
#include <mutex>
#include <queue>
#include <string>

#include "grpc_client.h"
#include "http_client.h"
#include "json_utils.h"

namespace tc = triton::client;

#define FAIL_IF_ERR(X, MSG)                                        \
  {                                                                \
    tc::Error err = (X);                                           \
    if (!err.IsOk()) {                                             \
      std::cerr << "error: " << (MSG) << ": " << err << std::endl; \
      exit(1);                                                     \
    }                                                              \
  }

#define INFO(msg) \
  {               \
    std::cout << "[INFO] " << "[" << __FILE__ << ":" << __LINE__ << "] " << msg << std::endl;\
  }


int
main(int argc, char** argv)
{
  bool verbose = true;
  bool async = false;
  bool streaming = false;
  int batch_size = 1;
  std::string model_name = "ensemble";
  std::string model_version = "";
  std::string url("localhost:8001");
  tc::Headers http_headers;

  // Create the inference client for the server. From it
  // extract and validate that the model meets the requirements for
  // image classification.
  std::unique_ptr<tc::InferenceServerGrpcClient> client;

  tc::Error err;
  FAIL_IF_ERR(
      tc::InferenceServerGrpcClient::Create(&client, url, verbose),
      "unable to create client for inference");
  
  inference::ModelMetadataResponse model_metadata;
  
  FAIL_IF_ERR(
      client->ModelMetadata(
          &model_metadata, model_name, model_version, http_headers),
      "failed to get model metadata");

  inference::ModelConfigResponse model_config;
  
  FAIL_IF_ERR(
      client->ModelConfig(
          &model_config, model_name, model_version, http_headers),
      "failed to get model config");
  INFO("model_config.config().input().size()=" << model_config.config().input().size());

  std::vector<int64_t> shape{1, 1};

  std::string input_string = "请介绍一下苹果公司";
  std::vector<std::string> input_text;
  input_text.push_back(input_string);
  INFO("input_text.size()=" << input_text.size());

  std::vector<uint32_t> max_tokens_int{512};
  std::vector<std::string> bad_words_text{""};
  std::vector<std::string> stop_words_text{""};


  tc::InferInput* text_input;
  FAIL_IF_ERR(
      tc::InferInput::Create(
          &text_input, "text_input", shape, "BYTES"),
      "unable to get input text_input");

  tc::InferInput* max_tokens;
  FAIL_IF_ERR(
      tc::InferInput::Create(
          &max_tokens, "max_tokens", shape, "INT32"),
      "unable to get input max_tokens");

  tc::InferInput* bad_words;
  FAIL_IF_ERR(
      tc::InferInput::Create(
          &bad_words, "bad_words", shape, "BYTES"),
      "unable to get input bad_words");

  tc::InferInput* stop_words;
  FAIL_IF_ERR(
      tc::InferInput::Create(
          &stop_words, "stop_words", shape, "BYTES"),
      "unable to get input stop_words");

  std::shared_ptr<tc::InferInput> text_input_ptr;
  text_input_ptr.reset(text_input);
  std::shared_ptr<tc::InferInput> max_tokens_ptr;
  max_tokens_ptr.reset(max_tokens);
  std::shared_ptr<tc::InferInput> bad_words_ptr;
  bad_words_ptr.reset(bad_words);
  std::shared_ptr<tc::InferInput> stop_words_ptr;
  stop_words_ptr.reset(stop_words);

  FAIL_IF_ERR(
      text_input_ptr->AppendFromString(input_text),
      "unable to set data for text_input");
  FAIL_IF_ERR(
      max_tokens_ptr->AppendRaw(
        reinterpret_cast<uint8_t*>(&max_tokens_int[0]), max_tokens_int.size() * sizeof(uint32_t)
      ),
      "unable to set data for max_tokens");
  FAIL_IF_ERR(
      bad_words_ptr->AppendFromString(bad_words_text),
      "unable to set data for bad_words");
  FAIL_IF_ERR(
      stop_words_ptr->AppendFromString(stop_words_text),
      "unable to set data for stop_words");

  tc::InferRequestedOutput* text_output;
  FAIL_IF_ERR(
      tc::InferRequestedOutput::Create(&text_output, "text_output"),
      "unable to get output text_output");


  std::shared_ptr<tc::InferRequestedOutput> text_output_ptr;
  text_output_ptr.reset(text_output);

  std::vector<tc::InferInput*> inputs = {text_input_ptr.get(), max_tokens_ptr.get(), bad_words_ptr.get(), stop_words_ptr.get()};
  std::vector<const tc::InferRequestedOutput*> outputs = {text_output_ptr.get()};

  tc::InferOptions options(model_name);
  options.model_version_ = model_version;

  std::mutex mtx;
  std::condition_variable cv;
  std::unique_ptr<tc::InferResult> results_ptr;
  auto callback_func = [&](tc::InferResult* results) {
    {
      std::lock_guard<std::mutex> lk(mtx);
      results_ptr.reset(results);
    }
    cv.notify_all();
  };

  // Send async request.
  FAIL_IF_ERR(
      client->AsyncInfer(
        callback_func, options, inputs, outputs, http_headers),
        "failed to send asynchronous infer request");

  // Wait for the results
  INFO("waiting for results");
  std::unique_lock<std::mutex> lk(mtx);
  cv.wait(lk);

  std::string output_name = "text_output";
  
  auto result = results_ptr.get();
  if (!result->RequestStatus().IsOk()) {
    std::cerr << "inference  failed with error: " << result->RequestStatus()
              << std::endl;
    exit(1);
  }

  // Get and validate the shape and datatype
  std::vector<int64_t> shape1;
  FAIL_IF_ERR(
      result->Shape(output_name, &shape1),
      "unable to get shape1 for text_output");
  INFO("shape1.size()=" << shape1.size());

  std::string datatype;
  FAIL_IF_ERR(
      result->Datatype(output_name, &datatype),
      "unable to get datatype for text_output");
  INFO("datatype=" << datatype);
  
  // Validate datatype
  if (datatype.compare("BYTES") != 0) {
    std::cerr << "received incorrect datatype for " << output_name << ": "
              << datatype << std::endl;
    exit(1);
  }

  std::vector<std::string> output_text;
  FAIL_IF_ERR(
      result->StringData(output_name, &output_text),
      "unable to get data for text_output");

  INFO("output_text.size()=" << output_text.size());
  for (size_t i = 0; i < output_text.size(); ++i) {
    INFO("output_text[" << i << "]=" << output_text[i]);
  }


  // Get pointers to the result returned...
  // float* outputc_data;
  // size_t outputc_byte_size;
  //   result->RawData(
  //       output_name, (const uint8_t**)&outputc_data, &outputc_byte_size);
  
  // std::cout << "outputc_byte_size=" << outputc_byte_size << std::endl;
  // std::cout << "outputc_data[0]=" << outputc_data[0] << std::endl;

  INFO(result->DebugString());
  return 0;
}



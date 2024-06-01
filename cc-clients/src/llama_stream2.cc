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

#define FAIL_IF_ERR(X, MSG)                                                    \
  {                                                                            \
    tc::Error err = (X);                                                       \
    if (!err.IsOk()) {                                                         \
      std::cerr << "[ERROR] "                                                  \
                << "[" << __FILE__ << ":" << __LINE__ << "] " << (MSG) << ": " \
                << err << std::endl;                                           \
      exit(1);                                                                 \
    }                                                                          \
  }

#define INFO(msg)                                                  \
  {                                                                \
    std::cout << "[INFO] "                                         \
              << "[" << __FILE__ << ":" << __LINE__ << "] " << msg \
              << std::endl;                                        \
  }

using CallbackFn = std::function<void(tc::InferResult*)>;


class LlamaTrtClient {
 public:
  LlamaTrtClient(const std::string& url, bool verbose = false)
      : url(url), verbose(verbose), chat_count(0), stop_threads(false)
  {
    FAIL_IF_ERR(
        tc::InferenceServerGrpcClient::Create(&client, url, verbose),
        "unable to create client for inference");

    pre_model_name = "preprocessing";
    llama_model_name = "tensorrt_llm";
    post_model_name = "postprocessing";
    pre_model_version = "";
    llama_model_version = "";
    post_model_version = "";
    init_special_tokens();

    callback_fn = [&](tc::InferResult* result) {
      if (!result->RequestStatus().IsOk()) {
        std::cerr << "inference  failed with error: " << result->RequestStatus()
                  << std::endl;
        exit(1);
      }
      postprocess(result);
    };

    // Start Stream
    FAIL_IF_ERR(
        client->StartStream(
            callback_fn, true /* enable_stats*/, 0 /*stream_timeout*/,
            http_headers),
        "failed to start stream");

    print_input_thread = std::thread(&LlamaTrtClient::print_intput, this);
    print_output_thread = std::thread(&LlamaTrtClient::print_output, this);
    print_input_thread.detach();
    print_output_thread.detach();
  }

  ~LlamaTrtClient()
  {
    client->StopStream();
    stop_threads = true;
    cv_in.notify_one();
    cv_out.notify_one();
  }


  int init_special_tokens();

  int set_system_prompt(const std::string& msg)
  {
    system_message = msg;
    return 0;
  }

  int concat_message(
      const std::string& system_message,
      const std::vector<std::string>& user_messages,
      const std::vector<std::string>& assistant_messages,
      std::unique_ptr<std::string>* message);

  int clear_message();

  int create_input(
      const std::string& name, const std::vector<int64_t>& shape,
      const std::string& dtype, const std::vector<std::string>& data,
      std::shared_ptr<tc::InferInput>* input);

  int create_input(
      const std::string& name, const std::vector<int64_t>& shape,
      const std::string& dtype, const std::vector<int32_t>& data,
      std::shared_ptr<tc::InferInput>* input);

  int create_input(
      const std::string& name, const std::vector<int64_t>& shape,
      const std::string& dtype, const std::vector<float>& data,
      std::shared_ptr<tc::InferInput>* input);

  int create_output(
      const std::string& name,
      std::shared_ptr<tc::InferRequestedOutput>* output);

  int get_output_data_and_shape(
      const tc::InferResult* results_ptr, const std::string& name,
      std::vector<int64_t>* shape, std::unique_ptr<std::vector<int32_t>>* data);

  int get_output_data_and_shape(
      const tc::InferResult* results_ptr, const std::string& name,
      std::vector<int64_t>* shape,
      std::unique_ptr<std::vector<std::string>>* data);

  int preprocess(std::unique_ptr<tc::InferResult>* results_ptr);

  int llama_infer(const std::unique_ptr<tc::InferResult>& pre_results_ptr);

  int postprocess(const tc::InferResult* pre_results);

  bool is_done(const std::vector<int32_t>& tokens);

  void done_callback();

  void print_output();

  void print_intput();

  int run_one_inference(const std::string& user_messages);

  int run_chat();


 private:
  std::unique_ptr<tc::InferenceServerGrpcClient> client;
  tc::Headers http_headers;
  std::string pre_model_name;
  std::string llama_model_name;
  std::string post_model_name;
  std::string pre_model_version;
  std::string llama_model_version;
  std::string post_model_version;
  std::string url;
  bool verbose;
  std::string system_message;
  std::vector<std::string> user_messages;
  std::vector<std::string> assistant_messages;
  std::string assistant_msg;

  std::string bos;
  std::string system_header;
  std::string user_header;
  std::string assistant_header;
  std::string eot;
  std::string eos;

  std::function<void(tc::InferResult*)> callback_fn;

  std::queue<std::string> assistant_msg_queue;
  std::queue<std::string> user_msg_queue;
  std::atomic<int> chat_count;


  std::mutex mtx;
  std::mutex mtx_in;
  std::mutex mtx_out;
  std::condition_variable cv_out;
  std::condition_variable cv_in;
  std::condition_variable cv_done;

  std::thread print_input_thread;
  std::thread print_output_thread;
  std::atomic<bool> stop_threads;
};


int
LlamaTrtClient::init_special_tokens()
{
  bos = "<|begin_of_text|>";
  system_header = "<|start_header_id|>system<|end_header_id|>\n\n";
  user_header = "<|start_header_id|>user<|end_header_id|>\n\n";
  assistant_header = "<|start_header_id|>assistant<|end_header_id|>\n\n";
  eot = "<|eot_id|>";
  eos = "<|end_of_text|>";
  return 0;
}

int
LlamaTrtClient::concat_message(
    const std::string& system_message,
    const std::vector<std::string>& user_messages,
    const std::vector<std::string>& assistant_messages,
    std::unique_ptr<std::string>* message)
{
  std::string message_str = bos;

  message_str += system_header + system_message + eot;

  size_t msg_size_max =
      std::max(user_messages.size(), assistant_messages.size());
  for (size_t i = 0; i < msg_size_max; ++i) {
    if (i < user_messages.size()) {
      message_str += user_header + user_messages[i] + eot;
    }
    if (i < assistant_messages.size()) {
      message_str += assistant_header + assistant_messages[i] + eot;
    }
  }
  message_str += assistant_header;

  message->reset(new std::string(message_str));
  return 0;
}


int
LlamaTrtClient::clear_message()
{
  user_messages.clear();
  assistant_messages.clear();
  assistant_msg.clear();
  return 0;
}


int
LlamaTrtClient::create_input(
    const std::string& name, const std::vector<int64_t>& shape,
    const std::string& dtype, const std::vector<std::string>& data,
    std::shared_ptr<tc::InferInput>* input)
{
  tc::InferInput* input_ptr;
  FAIL_IF_ERR(
      tc::InferInput::Create(&input_ptr, name, shape, dtype),
      "unable to get input " + name);

  FAIL_IF_ERR(
      input_ptr->AppendFromString(data), "unable to set data for " + name);
  input->reset(input_ptr);
  return 0;
}


int
LlamaTrtClient::create_input(
    const std::string& name, const std::vector<int64_t>& shape,
    const std::string& dtype, const std::vector<int32_t>& data,
    std::shared_ptr<tc::InferInput>* input)
{
  tc::InferInput* input_ptr;
  FAIL_IF_ERR(
      tc::InferInput::Create(&input_ptr, name, shape, dtype),
      "unable to get input " + name);

  FAIL_IF_ERR(
      input_ptr->AppendRaw(
          reinterpret_cast<const uint8_t*>(&data[0]),
          data.size() * sizeof(int32_t)),
      "unable to set data for " + name);
  input->reset(input_ptr);
  return 0;
}


int
LlamaTrtClient::create_input(
    const std::string& name, const std::vector<int64_t>& shape,
    const std::string& dtype, const std::vector<float>& data,
    std::shared_ptr<tc::InferInput>* input)
{
  tc::InferInput* input_ptr;
  FAIL_IF_ERR(
      tc::InferInput::Create(&input_ptr, name, shape, dtype),
      "unable to get input " + name);

  FAIL_IF_ERR(
      input_ptr->AppendRaw(
          reinterpret_cast<const uint8_t*>(&data[0]),
          data.size() * sizeof(int32_t)),
      "unable to set data for " + name);
  input->reset(input_ptr);
  return 0;
}


int
LlamaTrtClient::create_output(
    const std::string& name, std::shared_ptr<tc::InferRequestedOutput>* output)
{
  tc::InferRequestedOutput* output_ptr;
  FAIL_IF_ERR(
      tc::InferRequestedOutput::Create(&output_ptr, name),
      "unable to get output " + name);
  output->reset(output_ptr);
  return 0;
}


int
LlamaTrtClient::get_output_data_and_shape(
    const tc::InferResult* results_ptr, const std::string& name,
    std::vector<int64_t>* shape, std::unique_ptr<std::vector<int32_t>>* data)
{
  if (!results_ptr->RequestStatus().IsOk()) {
    std::cerr << "inference  failed with error: "
              << results_ptr->RequestStatus() << std::endl;
    exit(1);
  }

  FAIL_IF_ERR(
      results_ptr->Shape(name, shape),
      "unable to get output shape for " + name);

  int32_t* tmp_data;
  size_t byte_size;
  FAIL_IF_ERR(
      results_ptr->RawData(name, (const uint8_t**)&tmp_data, &byte_size),
      "unable to get data for " + name);
  std::vector<int32_t> tmp_data_vector;
  tmp_data_vector.resize(byte_size / sizeof(int32_t));
  std::memcpy(tmp_data_vector.data(), tmp_data, byte_size);
  data->reset(new std::vector<int32_t>(tmp_data_vector));

  return 0;
}


int
LlamaTrtClient::get_output_data_and_shape(
    const tc::InferResult* results_ptr, const std::string& name,
    std::vector<int64_t>* shape,
    std::unique_ptr<std::vector<std::string>>* data)
{
  if (!results_ptr->RequestStatus().IsOk()) {
    std::cerr << "inference  failed with error: "
              << results_ptr->RequestStatus() << std::endl;
    exit(1);
  }

  FAIL_IF_ERR(
      results_ptr->Shape(name, shape),
      "unable to get output shape for " + name);

  std::vector<std::string> tmp_data;
  FAIL_IF_ERR(
      results_ptr->StringData(name, &tmp_data),
      "unable to get data for " + name);
  data->reset(new std::vector<std::string>(tmp_data));

  return 0;
}


int
LlamaTrtClient::preprocess(std::unique_ptr<tc::InferResult>* results_ptr)
{
  // 输入字符串
  assistant_msg.clear();
  std::unique_ptr<std::string> message;
  concat_message(system_message, user_messages, assistant_messages, &message);

  std::shared_ptr<tc::InferInput> query;
  create_input("QUERY", {1, 1}, "BYTES", {*message}, &query);

  // 输入最大token数
  std::vector<int32_t> max_tokens_int{1024};
  std::shared_ptr<tc::InferInput> request_output_len_input;
  create_input(
      "REQUEST_OUTPUT_LEN", {1, max_tokens_int.size()}, "INT32", max_tokens_int,
      &request_output_len_input);

  // 输入bad words
  std::vector<std::string> bad_words_text{""};
  std::shared_ptr<tc::InferInput> bad_words;
  create_input("BAD_WORDS_DICT", {1, 1}, "BYTES", bad_words_text, &bad_words);

  // 输入stop words
  std::vector<std::string> stop_words_text{eos, eot};
  std::shared_ptr<tc::InferInput> stop_words;
  create_input(
      "STOP_WORDS_DICT", {1, stop_words_text.size()}, "BYTES", stop_words_text,
      &stop_words);

  // 输出input id
  std::shared_ptr<tc::InferRequestedOutput> input_id;
  create_output("INPUT_ID", &input_id);

  // 输出request output len
  std::shared_ptr<tc::InferRequestedOutput> request_output_len;
  create_output("REQUEST_OUTPUT_LEN", &request_output_len);

  // 输出stop words ids
  std::shared_ptr<tc::InferRequestedOutput> stop_words_ids;
  create_output("STOP_WORDS_IDS", &stop_words_ids);

  std::vector<tc::InferInput*> inputs = {
      query.get(), request_output_len_input.get(), bad_words.get(),
      stop_words.get()};

  std::vector<const tc::InferRequestedOutput*> outputs = {
      input_id.get(), request_output_len.get(), stop_words_ids.get()};

  tc::InferOptions options(pre_model_name);
  options.model_version_ = pre_model_version;

  // Send request.
  tc::InferResult* pre_results;
  FAIL_IF_ERR(
      client->Infer(&pre_results, options, inputs, outputs, http_headers),
      "failed to send synchronous infer request");

  results_ptr->reset(pre_results);
  return 0;
}

int
LlamaTrtClient::llama_infer(
    const std::unique_ptr<tc::InferResult>& pre_results_ptr)
{
  auto pre_result = pre_results_ptr.get();
  if (!pre_result->RequestStatus().IsOk()) {
    std::cerr << "preprocessing failed with error: "
              << pre_result->RequestStatus() << std::endl;
    exit(1);
  }

  // 输入streaming
  std::vector<int32_t> streaming_bool{1};
  std::shared_ptr<tc::InferInput> streaming;
  create_input(
      "streaming", {1, streaming_bool.size()}, "BOOL", streaming_bool,
      &streaming);

  // 输入input ids
  std::vector<int64_t> input_ids_shape;
  std::unique_ptr<std::vector<int32_t>> input_ids_data;
  get_output_data_and_shape(
      pre_result, "INPUT_ID", &input_ids_shape, &input_ids_data);

  std::shared_ptr<tc::InferInput> input_ids;
  create_input(
      "input_ids", input_ids_shape, "INT32", *input_ids_data, &input_ids);

  // 输入input lengths
  std::vector<int32_t> input_ids_len{input_ids_data->size()};
  std::shared_ptr<tc::InferInput> input_lengths;
  create_input(
      "input_lengths", {1, input_ids_len.size()}, "INT32", input_ids_len,
      &input_lengths);

  // 输入request output len
  std::vector<int64_t> request_output_len_shape;
  std::unique_ptr<std::vector<int32_t>> request_output_len_data;
  get_output_data_and_shape(
      pre_result, "REQUEST_OUTPUT_LEN", &request_output_len_shape,
      &request_output_len_data);

  std::shared_ptr<tc::InferInput> request_output_len;
  create_input(
      "request_output_len", request_output_len_shape, "INT32",
      *request_output_len_data, &request_output_len);

  // 输入end id
  std::vector<int32_t> end_id_vector{128001};
  std::shared_ptr<tc::InferInput> end_id;
  create_input(
      "end_id", {1, end_id_vector.size()}, "INT32", end_id_vector, &end_id);

  // 输入stop words list
  std::vector<int64_t> stop_words_shape;
  std::unique_ptr<std::vector<int32_t>> stop_words_data;
  get_output_data_and_shape(
      pre_result, "STOP_WORDS_IDS", &stop_words_shape, &stop_words_data);

  std::shared_ptr<tc::InferInput> stop_words_list;
  create_input(
      "stop_words_list", stop_words_shape, "INT32", *stop_words_data,
      &stop_words_list);

  // 输出output ids
  std::shared_ptr<tc::InferRequestedOutput> output_ids;
  create_output("output_ids", &output_ids);

  // 输出sequence length
  std::shared_ptr<tc::InferRequestedOutput> sequence_length;
  create_output("sequence_length", &sequence_length);

  std::vector<tc::InferInput*> inputs = {
      streaming.get(),       input_ids.get(),     end_id.get(),
      stop_words_list.get(), input_lengths.get(), request_output_len.get()};

  std::vector<const tc::InferRequestedOutput*> outputs = {
      output_ids.get(), sequence_length.get()};

  tc::InferOptions options(llama_model_name);
  options.model_version_ = llama_model_version;

  // Send async request
  FAIL_IF_ERR(
      client->AsyncStreamInfer(options, inputs, outputs),
      "failed to send async infer request");

  return 0;
}

int
LlamaTrtClient::postprocess(const tc::InferResult* pre_result)
{
  if (!pre_result->RequestStatus().IsOk()) {
    std::cerr << "inference  failed with error: " << pre_result->RequestStatus()
              << std::endl;
    exit(1);
  }

  // 输入tokens_batch
  std::vector<int64_t> tokens_batch_shape;
  std::unique_ptr<std::vector<int32_t>> tokens_batch_data;
  get_output_data_and_shape(
      pre_result, "output_ids", &tokens_batch_shape, &tokens_batch_data);

  if (is_done(*tokens_batch_data)) {
    done_callback();
    return 0;
  }

  std::shared_ptr<tc::InferInput> tokens_batch;
  create_input(
      "TOKENS_BATCH", tokens_batch_shape, "INT32", *tokens_batch_data,
      &tokens_batch);

  // 输入sequence length
  std::vector<int64_t> sequence_length_shape;
  std::unique_ptr<std::vector<int32_t>> sequence_length_data;
  get_output_data_and_shape(
      pre_result, "sequence_length", &sequence_length_shape,
      &sequence_length_data);

  std::shared_ptr<tc::InferInput> sequence_length;
  create_input(
      "SEQUENCE_LENGTH", sequence_length_shape, "INT32", *sequence_length_data,
      &sequence_length);

  // 输入cum_log_probs
  std::vector<float> cum_log_probs_data{0.0};
  std::shared_ptr<tc::InferInput> cum_log_probs;
  create_input(
      "CUM_LOG_PROBS", {1, cum_log_probs_data.size()}, "FP32",
      cum_log_probs_data, &cum_log_probs);

  // 输入output_log_probs
  std::vector<float> output_log_probs_data{0.0};
  std::shared_ptr<tc::InferInput> output_log_probs;
  create_input(
      "OUTPUT_LOG_PROBS", {1, 1, output_log_probs_data.size()}, "FP32",
      output_log_probs_data, &output_log_probs);

  // 输入context_logits
  std::vector<float> context_logits_data{0.0};
  std::shared_ptr<tc::InferInput> context_logits;
  create_input(
      "CONTEXT_LOGITS", {1, 1, context_logits_data.size()}, "FP32",
      context_logits_data, &context_logits);

  // 输入generation_logits
  std::vector<float> generation_logits_data{0.0};
  std::shared_ptr<tc::InferInput> generation_logits;
  create_input(
      "GENERATION_LOGITS", {1, 1, 1, generation_logits_data.size()}, "FP32",
      generation_logits_data, &generation_logits);

  // 输出output
  std::shared_ptr<tc::InferRequestedOutput> output;
  create_output("OUTPUT", &output);

  std::vector<tc::InferInput*> inputs = {
      tokens_batch.get(),     sequence_length.get(), cum_log_probs.get(),
      output_log_probs.get(), context_logits.get(),  generation_logits.get()};

  std::vector<const tc::InferRequestedOutput*> outputs = {output.get()};

  tc::InferOptions options(post_model_name);
  options.model_version_ = post_model_version;

  // Send request.
  tc::InferResult* post_results;
  FAIL_IF_ERR(
      client->Infer(&post_results, options, inputs, outputs, http_headers),
      "failed to send synchronous infer request");

  std::vector<std::string> result_data;
  FAIL_IF_ERR(
      post_results->StringData("OUTPUT", &result_data),
      "unable to get data for OUTPUT");
  {
    for (size_t i = 0; i < result_data.size(); ++i) {
      // std::cout << result_data[i] << std::flush;
      std::lock_guard<std::mutex> lk(mtx_out);
      assistant_msg += result_data[i];
      assistant_msg_queue.push(result_data[i]);
      cv_out.notify_one();
    }
  }
  return 0;
}


void
LlamaTrtClient::done_callback()
{
  std::lock_guard<std::mutex> lk(mtx);
  assistant_messages.push_back(assistant_msg);
  assistant_msg.clear();
  cv_done.notify_all();
}


bool
LlamaTrtClient::is_done(const std::vector<int32_t>& tokens)
{
  return std::find(tokens.begin(), tokens.end(), 128001) != tokens.end() or
         std::find(tokens.begin(), tokens.end(), 128009) != tokens.end();
}


int
LlamaTrtClient::run_one_inference(const std::string& msg)
{
  if (msg.empty()) {
    return 0;
  }
  user_msg_queue.push(msg);
  cv_in.notify_one();

  this->user_messages.push_back(msg);
  std::unique_ptr<tc::InferResult> pre_results_ptr;
  preprocess(&pre_results_ptr);

  std::unique_ptr<tc::InferResult> results_ptr;
  llama_infer(pre_results_ptr);

  std::unique_lock<std::mutex> lk(mtx);
  cv_done.wait(lk);
  return 0;
}


int
LlamaTrtClient::run_chat()
{
  std::cout << "Welcome to Llama Chatbot!" << std::endl;
  std::string user_msg;
  while (true) {
    std::cout << std::endl << "Input[" << chat_count << "]: " << std::flush;
    std::getline(std::cin, user_msg);
    if (user_msg.empty()) {
      continue;
    } else if (user_msg == "exit") {
      std::cout << "Goodbye!" << std::endl;
      break;
    } else if (user_msg == "clear") {
      std::cout << "Chat history cleared!" << std::endl;
      clear_message();
      continue;
    }
    run_one_inference(user_msg);
    chat_count++;
  }
}


void
LlamaTrtClient::print_output()
{
  while (true) {
    std::unique_lock<std::mutex> lk(mtx_out);
    cv_out.wait(
        lk, [this] { return !assistant_msg_queue.empty() or stop_threads; });
    if (stop_threads) {
      break;
    }
    std::string msg = assistant_msg_queue.front();
    assistant_msg_queue.pop();
    std::cout << msg << std::flush;
  }
}

void
LlamaTrtClient::print_intput()
{
  while (true) {
    std::unique_lock<std::mutex> lk(mtx_in);
    cv_in.wait(lk, [this] { return !user_msg_queue.empty() or stop_threads; });
    if (stop_threads) {
      break;
    }
    std::string user_msg = user_msg_queue.front();
    user_msg_queue.pop();
    std::cout << std::endl;
    std::cout << "Input[" << chat_count << "]: " << user_msg << std::endl
              << std::flush;
    std::cout << "Output[" << chat_count << "]: " << std::flush;
  }
}

int
main(int argc, char** argv)
{
  bool verbose = false;
  std::string model_version = "";
  std::string url("localhost:8001");

  std::string system_message =
      "You are a helpful AI assistant for travel tips and recommendations";
  std::string user_message = "What is France's capital?";

  // Parse commandline...
  int opt;
  while ((opt = getopt(argc, argv, "vu:i:s:")) != -1) {
    switch (opt) {
      case 'v':
        verbose = true;
        break;
      case 'u':
        url = optarg;
        break;
      case 'i': {
        user_message = optarg;
        break;
      }
      case 's': {
        system_message = optarg;
        break;
      }
      case '?':
        break;
    }
  }

  LlamaTrtClient client(url, verbose);
  client.set_system_prompt(system_message);
  // client.run_one_inference(user_message);
  client.run_chat();

  std::cout << std::endl;
  return 0;
}

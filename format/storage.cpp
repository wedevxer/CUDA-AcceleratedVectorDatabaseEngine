#include "storage.h"
#include <arrow/io/file.h>
#include <arrow/ipc/writer.h>
#include <arrow/ipc/reader.h>
#include <arrow/table.h>
#include <arrow/array.h>
#include <arrow/buffer.h>
#include <arrow/type.h>
#include <arrow/builder.h>
#include <arrow/memory_pool.h>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <json/json.h>
#include <chrono>
#include <random>

namespace vdb {
namespace storage {

// IndexManifest JSON serialization
Json::Value IndexManifest::to_json() const {
    Json::Value root;
    
    // Basic index information
    root["index_name"] = index_name;
    root["epoch"] = epoch;
    root["dimension"] = dimension;
    root["nlist"] = nlist;
    root["metric"] = metric;
    
    // PQ parameters
    Json::Value pq_json;
    pq_json["m"] = pq_params.m;
    pq_json["nbits"] = pq_params.nbits;
    root["pq_params"] = pq_json;
    
    // Shard information
    Json::Value shards_json(Json::arrayValue);
    for (const auto& shard : shards) {
        Json::Value shard_json;
        shard_json["list_id"] = shard.list_id;
        shard_json["path"] = shard.path;
        shard_json["num_vectors"] = static_cast<Json::UInt64>(shard.num_vectors);
        shard_json["file_size"] = static_cast<Json::UInt64>(shard.file_size);
        shards_json.append(shard_json);
    }
    root["shards"] = shards_json;
    
    // Timestamp
    auto time_since_epoch = created_at.time_since_epoch();
    auto timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(time_since_epoch).count();
    root["created_at"] = static_cast<Json::UInt64>(timestamp_ns);
    
    return root;
}

// IndexManifest JSON deserialization
IndexManifest IndexManifest::from_json(const Json::Value& json) {
    IndexManifest manifest;
    
    // Basic index information
    manifest.index_name = json["index_name"].asString();
    manifest.epoch = json["epoch"].asString();
    manifest.dimension = json["dimension"].asUInt();
    manifest.nlist = json["nlist"].asUInt();
    manifest.metric = json["metric"].asString();
    
    // PQ parameters
    if (json.isMember("pq_params")) {
        const auto& pq_json = json["pq_params"];
        manifest.pq_params.m = pq_json["m"].asUInt();
        manifest.pq_params.nbits = pq_json["nbits"].asUInt();
    }
    
    // Shard information
    if (json.isMember("shards")) {
        const auto& shards_json = json["shards"];
        for (const auto& shard_json : shards_json) {
            ShardInfo shard;
            shard.list_id = shard_json["list_id"].asUInt();
            shard.path = shard_json["path"].asString();
            shard.num_vectors = shard_json["num_vectors"].asUInt64();
            shard.file_size = shard_json["file_size"].asUInt64();
            manifest.shards.push_back(shard);
        }
    }
    
    // Timestamp
    if (json.isMember("created_at")) {
        auto timestamp_ns = json["created_at"].asUInt64();
        manifest.created_at = std::chrono::system_clock::time_point(
            std::chrono::nanoseconds(timestamp_ns));
    }
    
    return manifest;
}

// Save manifest to file
void IndexManifest::save(const std::string& path) const {
    Json::Value json_manifest = to_json();
    
    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open manifest file for writing: " + path);
    }
    
    Json::StreamWriterBuilder builder;
    builder["indentation"] = "  ";  // Pretty print with 2-space indent
    std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
    
    writer->write(json_manifest, &file);
    file.close();
}

// Load manifest from file
IndexManifest IndexManifest::load(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open manifest file for reading: " + path);
    }
    
    Json::Value json_manifest;
    Json::CharReaderBuilder builder;
    std::string errors;
    
    if (!Json::parseFromStream(builder, file, &json_manifest, &errors)) {
        throw std::runtime_error("Failed to parse manifest JSON: " + errors);
    }
    
    return from_json(json_manifest);
}

// ArrowStorage constructor
ArrowStorage::ArrowStorage() 
    : pool_(arrow::default_memory_pool()) {
}

ArrowStorage::~ArrowStorage() = default;

// Read vectors from Arrow file
arrow::Result<std::shared_ptr<arrow::Table>> ArrowStorage::read_vectors(
    const std::string& path, int64_t offset, int64_t length) {
    
    // Open file for reading
    ARROW_ASSIGN_OR_RAISE(auto input_file, arrow::io::ReadableFile::Open(path));
    
    // Create IPC reader
    ARROW_ASSIGN_OR_RAISE(auto ipc_reader, 
                         arrow::ipc::RecordBatchFileReader::Open(input_file));
    
    // Read all record batches
    std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
    int num_batches = ipc_reader->num_record_batches();
    
    for (int i = 0; i < num_batches; ++i) {
        ARROW_ASSIGN_OR_RAISE(auto batch, ipc_reader->ReadRecordBatch(i));
        
        // Apply offset and length filtering if specified
        if (offset > 0 || length > 0) {
            int64_t start = offset;
            int64_t end = (length > 0) ? std::min(offset + length, batch->num_rows()) : batch->num_rows();
            
            if (start < batch->num_rows() && start < end) {
                ARROW_ASSIGN_OR_RAISE(auto sliced_batch, batch->Slice(start, end - start));
                batches.push_back(sliced_batch);
            }
        } else {
            batches.push_back(batch);
        }
    }
    
    // Combine batches into a table
    if (batches.empty()) {
        return arrow::Status::Invalid("No data found in range");
    }
    
    ARROW_ASSIGN_OR_RAISE(auto table, arrow::Table::FromRecordBatches(batches));
    return table;
}

// Write vectors to Arrow file
arrow::Status ArrowStorage::write_vectors(
    const std::string& path, const float* vectors, const uint64_t* ids,
    size_t n_vectors, uint32_t dimension) {
    
    // Create schema for vector data
    auto schema = create_vector_schema(dimension);
    
    // Create builders for each column
    arrow::UInt64Builder id_builder;
    arrow::ListBuilder vector_builder(pool_, std::make_shared<arrow::FloatBuilder>(pool_));
    auto float_builder = static_cast<arrow::FloatBuilder*>(vector_builder.value_builder());
    
    // Build arrays
    ARROW_RETURN_NOT_OK(id_builder.Reserve(n_vectors));
    ARROW_RETURN_NOT_OK(vector_builder.Reserve(n_vectors));
    ARROW_RETURN_NOT_OK(float_builder->Reserve(n_vectors * dimension));
    
    for (size_t i = 0; i < n_vectors; ++i) {
        // Add vector ID
        ARROW_RETURN_NOT_OK(id_builder.Append(ids[i]));
        
        // Add vector values
        ARROW_RETURN_NOT_OK(vector_builder.Append());
        for (uint32_t d = 0; d < dimension; ++d) {
            ARROW_RETURN_NOT_OK(float_builder->Append(vectors[i * dimension + d]));
        }
    }
    
    // Finish building arrays
    ARROW_ASSIGN_OR_RAISE(auto id_array, id_builder.Finish());
    ARROW_ASSIGN_OR_RAISE(auto vector_array, vector_builder.Finish());
    
    // Create record batch
    auto batch = arrow::RecordBatch::Make(schema, n_vectors, {id_array, vector_array});
    
    // Write to file
    ARROW_ASSIGN_OR_RAISE(auto output_file, arrow::io::FileOutputStream::Open(path));
    ARROW_ASSIGN_OR_RAISE(auto writer, arrow::ipc::MakeFileWriter(output_file, schema));
    
    ARROW_RETURN_NOT_OK(writer->WriteRecordBatch(*batch));
    ARROW_RETURN_NOT_OK(writer->Close());
    
    return arrow::Status::OK();
}

// Read centroids from Arrow file
arrow::Result<std::shared_ptr<arrow::Table>> ArrowStorage::read_centroids(
    const std::string& path) {
    return read_vectors(path);  // Same format as vectors
}

// Write centroids to Arrow file
arrow::Status ArrowStorage::write_centroids(
    const std::string& path, const float* centroids, 
    uint32_t nlist, uint32_t dimension) {
    
    // Create sequential IDs for centroids
    std::vector<uint64_t> ids(nlist);
    for (uint32_t i = 0; i < nlist; ++i) {
        ids[i] = i;
    }
    
    return write_vectors(path, centroids, ids.data(), nlist, dimension);
}

// Read PQ codebooks from Arrow file
arrow::Result<std::shared_ptr<arrow::Table>> ArrowStorage::read_pq_codebooks(
    const std::string& path) {
    return read_vectors(path);  // Same format, different interpretation
}

// Write PQ codebooks to Arrow file
arrow::Status ArrowStorage::write_pq_codebooks(
    const std::string& path, const float* codebooks,
    uint32_t m, uint32_t ks, uint32_t dsub) {
    
    // Flatten codebooks: m subquantizers, ks centroids each, dsub dimensions
    size_t total_centroids = m * ks;
    
    // Create IDs for each centroid (subquantizer_id << 16 | centroid_id)
    std::vector<uint64_t> ids(total_centroids);
    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t j = 0; j < ks; ++j) {
            ids[i * ks + j] = (static_cast<uint64_t>(i) << 16) | j;
        }
    }
    
    return write_vectors(path, codebooks, ids.data(), total_centroids, dsub);
}

// Memory-map file
arrow::Result<std::shared_ptr<arrow::Buffer>> ArrowStorage::mmap_file(
    const std::string& path) {
    
    ARROW_ASSIGN_OR_RAISE(auto input_file, arrow::io::ReadableFile::Open(path));
    ARROW_ASSIGN_OR_RAISE(auto size, input_file->GetSize());
    
    // Memory map the entire file
    ARROW_ASSIGN_OR_RAISE(auto buffer, input_file->ReadAt(0, size));
    
    return buffer;
}

// Create vector schema (ID + vector values)
std::shared_ptr<arrow::Schema> ArrowStorage::create_vector_schema(uint32_t dimension) {
    auto id_field = arrow::field("id", arrow::uint64());
    auto vector_field = arrow::field("vector", arrow::list(arrow::float32()));
    
    return arrow::schema({id_field, vector_field});
}

// Create centroid schema (same as vector schema)
std::shared_ptr<arrow::Schema> ArrowStorage::create_centroid_schema(uint32_t dimension) {
    return create_vector_schema(dimension);
}

// Create PQ schema (same as vector schema, different interpretation)
std::shared_ptr<arrow::Schema> ArrowStorage::create_pq_schema(uint32_t dsub) {
    return create_vector_schema(dsub);
}

// EpochManager constructor
EpochManager::EpochManager(const std::string& base_path) 
    : base_path_(base_path) {
    
    // Create base directory if it doesn't exist
    std::filesystem::create_directories(base_path_);
    
    // Load existing epoch state
    load_epoch_state();
}

EpochManager::~EpochManager() = default;

// Create new epoch
std::string EpochManager::create_epoch(const IndexManifest& manifest) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    
    std::string epoch_id = generate_epoch_id();
    std::string epoch_path = base_path_ + "/" + epoch_id;
    
    // Create epoch directory
    std::filesystem::create_directories(epoch_path);
    
    // Save manifest
    IndexManifest epoch_manifest = manifest;
    epoch_manifest.epoch = epoch_id;
    epoch_manifest.created_at = std::chrono::system_clock::now();
    epoch_manifest.save(epoch_path + "/manifest.json");
    
    // Create epoch entry
    auto epoch = std::make_shared<Epoch>();
    epoch->id = epoch_id;
    epoch->manifest = epoch_manifest;
    epoch->base_path = epoch_path;
    epoch->active = false;
    epoch->created_at = epoch_manifest.created_at;
    
    epochs_[epoch_id] = epoch;
    
    // Persist state
    persist_epoch_state();
    
    std::cout << "Created epoch: " << epoch_id << std::endl;
    return epoch_id;
}

// Activate epoch
void EpochManager::activate_epoch(const std::string& epoch_id) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    
    auto it = epochs_.find(epoch_id);
    if (it == epochs_.end()) {
        throw std::runtime_error("Epoch not found: " + epoch_id);
    }
    
    // Deactivate current epoch
    if (!active_epoch_id_.empty()) {
        auto current_it = epochs_.find(active_epoch_id_);
        if (current_it != epochs_.end()) {
            current_it->second->active = false;
        }
    }
    
    // Activate new epoch
    it->second->active = true;
    active_epoch_id_ = epoch_id;
    
    // Persist state
    persist_epoch_state();
    
    std::cout << "Activated epoch: " << epoch_id << std::endl;
}

// Deactivate epoch
void EpochManager::deactivate_epoch(const std::string& epoch_id) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    
    auto it = epochs_.find(epoch_id);
    if (it != epochs_.end()) {
        it->second->active = false;
    }
    
    if (active_epoch_id_ == epoch_id) {
        active_epoch_id_.clear();
    }
    
    persist_epoch_state();
    
    std::cout << "Deactivated epoch: " << epoch_id << std::endl;
}

// Get active epoch
std::shared_ptr<EpochManager::Epoch> EpochManager::get_active_epoch() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    if (active_epoch_id_.empty()) {
        return nullptr;
    }
    
    auto it = epochs_.find(active_epoch_id_);
    return (it != epochs_.end()) ? it->second : nullptr;
}

// Get specific epoch
std::shared_ptr<EpochManager::Epoch> EpochManager::get_epoch(const std::string& epoch_id) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    auto it = epochs_.find(epoch_id);
    return (it != epochs_.end()) ? it->second : nullptr;
}

// List all epochs
std::vector<std::string> EpochManager::list_epochs() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    std::vector<std::string> epoch_list;
    epoch_list.reserve(epochs_.size());
    
    for (const auto& pair : epochs_) {
        epoch_list.push_back(pair.first);
    }
    
    return epoch_list;
}

// Cleanup old epochs (keep most recent N)
void EpochManager::cleanup_old_epochs(int keep_n) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    
    if (epochs_.size() <= keep_n) {
        return;  // Nothing to clean up
    }
    
    // Sort epochs by creation time (newest first)
    std::vector<std::pair<std::chrono::system_clock::time_point, std::string>> sorted_epochs;
    for (const auto& pair : epochs_) {
        sorted_epochs.emplace_back(pair.second->created_at, pair.first);
    }
    
    std::sort(sorted_epochs.begin(), sorted_epochs.end(), std::greater<>());
    
    // Remove old epochs (beyond keep_n)
    for (size_t i = keep_n; i < sorted_epochs.size(); ++i) {
        const std::string& epoch_id = sorted_epochs[i].second;
        auto it = epochs_.find(epoch_id);
        
        if (it != epochs_.end() && !it->second->active) {
            // Remove epoch directory
            std::filesystem::remove_all(it->second->base_path);
            
            // Remove from memory
            epochs_.erase(it);
            
            std::cout << "Cleaned up old epoch: " << epoch_id << std::endl;
        }
    }
    
    persist_epoch_state();
}

// Generate unique epoch ID
std::string EpochManager::generate_epoch_id() const {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    
    // Add random component to avoid collisions
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1000, 9999);
    
    std::ostringstream oss;
    oss << "epoch_" << time_t << "_" << dis(gen);
    
    return oss.str();
}

// Persist epoch state to disk
void EpochManager::persist_epoch_state() {
    Json::Value state;
    state["active_epoch"] = active_epoch_id_;
    
    Json::Value epochs_json(Json::arrayValue);
    for (const auto& pair : epochs_) {
        Json::Value epoch_json;
        epoch_json["id"] = pair.first;
        epoch_json["active"] = pair.second->active;
        epoch_json["base_path"] = pair.second->base_path;
        
        auto timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>
            (pair.second->created_at.time_since_epoch()).count();
        epoch_json["created_at"] = static_cast<Json::UInt64>(timestamp_ns);
        
        epochs_json.append(epoch_json);
    }
    state["epochs"] = epochs_json;
    
    // Write to state file
    std::string state_file = base_path_ + "/epochs.json";
    std::ofstream file(state_file);
    if (file.is_open()) {
        Json::StreamWriterBuilder builder;
        std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
        writer->write(state, &file);
    }
}

// Load epoch state from disk
void EpochManager::load_epoch_state() {
    std::string state_file = base_path_ + "/epochs.json";
    
    if (!std::filesystem::exists(state_file)) {
        return;  // No existing state
    }
    
    std::ifstream file(state_file);
    if (!file.is_open()) {
        return;
    }
    
    Json::Value state;
    Json::CharReaderBuilder builder;
    std::string errors;
    
    if (!Json::parseFromStream(builder, file, &state, &errors)) {
        std::cerr << "Failed to parse epoch state: " << errors << std::endl;
        return;
    }
    
    // Load active epoch
    if (state.isMember("active_epoch")) {
        active_epoch_id_ = state["active_epoch"].asString();
    }
    
    // Load epochs
    if (state.isMember("epochs")) {
        const auto& epochs_json = state["epochs"];
        for (const auto& epoch_json : epochs_json) {
            std::string epoch_id = epoch_json["id"].asString();
            std::string epoch_path = epoch_json["base_path"].asString();
            
            // Verify epoch directory exists
            if (!std::filesystem::exists(epoch_path)) {
                continue;
            }
            
            // Load manifest
            std::string manifest_path = epoch_path + "/manifest.json";
            if (!std::filesystem::exists(manifest_path)) {
                continue;
            }
            
            try {
                auto manifest = IndexManifest::load(manifest_path);
                
                auto epoch = std::make_shared<Epoch>();
                epoch->id = epoch_id;
                epoch->manifest = manifest;
                epoch->base_path = epoch_path;
                epoch->active = epoch_json["active"].asBool();
                
                if (epoch_json.isMember("created_at")) {
                    auto timestamp_ns = epoch_json["created_at"].asUInt64();
                    epoch->created_at = std::chrono::system_clock::time_point(
                        std::chrono::nanoseconds(timestamp_ns));
                }
                
                epochs_[epoch_id] = epoch;
                
            } catch (const std::exception& e) {
                std::cerr << "Failed to load epoch " << epoch_id << ": " << e.what() << std::endl;
            }
        }
    }
    
    std::cout << "Loaded " << epochs_.size() << " epochs" << std::endl;
}

} // namespace storage
} // namespace vdb
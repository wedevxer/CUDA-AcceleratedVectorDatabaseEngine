# Format library - Storage layer with Arrow integration
# Handles vector storage, epoch management, and NVMe optimization

# Find required packages
find_package(Arrow REQUIRED)        # Apache Arrow for columnar storage
find_package(RocksDB REQUIRED)      # RocksDB for metadata
find_package(PkgConfig REQUIRED)    # For finding system packages

# Find jsoncpp for manifest serialization
pkg_check_modules(JSONCPP REQUIRED jsoncpp)

# Define format library
add_library(vdb_format SHARED
    storage.cpp                     # Arrow storage implementation
    epoch_manager.cpp              # Epoch-based index management
    shard_manager.cpp              # NVMe shard management
)

# Set C++ standard
set_target_properties(vdb_format PROPERTIES
    CXX_STANDARD 20                # C++20 features
    CXX_STANDARD_REQUIRED ON      # Enforce standard
)

# Link required libraries
target_link_libraries(vdb_format
    arrow                          # Apache Arrow
    RocksDB::rocksdb              # RocksDB for metadata
    ${JSONCPP_LIBRARIES}          # JSON parsing
)

# Include directories
target_include_directories(vdb_format PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}   # Format headers
    ${Arrow_INCLUDE_DIRS}         # Arrow headers
    ${RocksDB_INCLUDE_DIRS}       # RocksDB headers
    ${JSONCPP_INCLUDE_DIRS}       # JSON headers
)

# Compiler flags from pkg-config
target_compile_options(vdb_format PRIVATE
    ${JSONCPP_CFLAGS_OTHER}       # Additional compiler flags
)
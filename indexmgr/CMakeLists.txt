# Index Manager library - Index building and maintenance
# Handles offline index construction and epoch management

# Find required packages
find_package(Threads REQUIRED)      # Threading for parallel processing

# Define index manager library
add_library(vdb_indexmgr SHARED
    index_builder.cpp               # Index construction logic
    kmeans.cpp                      # K-means clustering for centroids
    pq_trainer.cpp                  # Product quantization training
    compactor.cpp                   # Index compaction and optimization
)

# Set C++ standard
set_target_properties(vdb_indexmgr PROPERTIES
    CXX_STANDARD 20                # C++20 features
    CXX_STANDARD_REQUIRED ON      # Enforce standard
)

# Link required libraries
target_link_libraries(vdb_indexmgr
    vdb_engine                     # Engine for CUDA operations
    vdb_format                     # Format for storage operations
    Threads::Threads               # Threading support
)

# Include directories
target_include_directories(vdb_indexmgr PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}    # Index manager headers
)

# Define index builder executable
add_executable(vdb_index_builder
    main.cpp                       # Index builder main function
)

# Set executable name
set_target_properties(vdb_index_builder PROPERTIES
    OUTPUT_NAME vdb_build_index    # Executable name
)

# Link index builder executable
target_link_libraries(vdb_index_builder
    vdb_indexmgr                   # Index manager library
    cxxopts                        # Command-line parsing
)
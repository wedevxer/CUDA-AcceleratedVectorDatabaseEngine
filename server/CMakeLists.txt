# Server library and executable - gRPC service implementation
# Provides query and admin services with metrics and monitoring

# Use parent scope dependencies
# find_package calls are in root CMakeLists.txt

# Generate protobuf and gRPC code from .proto files
set(PROTO_FILES ../proto/vdb.proto)
set(PROTO_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../proto)
set(GENERATED_PROTOBUF_PATH ${CMAKE_CURRENT_BINARY_DIR})

# Create output directory
file(MAKE_DIRECTORY ${GENERATED_PROTOBUF_PATH})

# Generated source files
set(PROTO_SRCS ${GENERATED_PROTOBUF_PATH}/vdb.pb.cc)
set(PROTO_HDRS ${GENERATED_PROTOBUF_PATH}/vdb.pb.h)
set(GRPC_SRCS ${GENERATED_PROTOBUF_PATH}/vdb.grpc.pb.cc)
set(GRPC_HDRS ${GENERATED_PROTOBUF_PATH}/vdb.grpc.pb.h)

# Generate protobuf sources
add_custom_command(
    OUTPUT ${PROTO_SRCS} ${PROTO_HDRS}
    COMMAND ${Protobuf_PROTOC_EXECUTABLE}
    ARGS --cpp_out=${GENERATED_PROTOBUF_PATH}
         --proto_path=${PROTO_PATH}
         ${PROTO_FILES}
    DEPENDS ${PROTO_FILES}
    COMMENT "Generating Protocol Buffer sources"
    VERBATIM
)

# Generate gRPC sources
add_custom_command(
    OUTPUT ${GRPC_SRCS} ${GRPC_HDRS}
    COMMAND ${Protobuf_PROTOC_EXECUTABLE}
    ARGS --grpc_out=${GENERATED_PROTOBUF_PATH}
         --plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>
         --proto_path=${PROTO_PATH}
         ${PROTO_FILES}
    DEPENDS ${PROTO_FILES} gRPC::grpc_cpp_plugin
    COMMENT "Generating gRPC sources"
    VERBATIM
)

# Define server library
add_library(vdb_server SHARED
    query_service.cpp               # Query service implementation
    admin_service.cpp               # Admin service implementation
    metrics.cpp                     # Prometheus metrics
    ${PROTO_SRCS}                  # Generated protobuf code
    ${GRPC_SRCS}                   # Generated gRPC code
)

# Set C++ standard
set_target_properties(vdb_server PROPERTIES
    CXX_STANDARD 20                # C++20 features
    CXX_STANDARD_REQUIRED ON      # Enforce standard
)

# Link required libraries
target_link_libraries(vdb_server
    gRPC::grpc++                   # gRPC C++ library
    protobuf::libprotobuf          # Protocol Buffers
    vdb_engine                     # Engine library
    vdb_format                     # Format library
    Threads::Threads               # Threading
)

# Include directories
target_include_directories(vdb_server PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}    # Server headers
    ${CMAKE_CURRENT_BINARY_DIR}    # Generated headers
)

# Define server executable
add_executable(vdb_server_bin
    main.cpp                       # Server main function
)

# Set executable name
set_target_properties(vdb_server_bin PROPERTIES
    OUTPUT_NAME vdb_server         # Executable name
)

# Link server executable
target_link_libraries(vdb_server_bin
    vdb_server                     # Server library
)
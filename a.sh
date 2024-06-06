cmake --build build                                  # 2
export GST_PLUGIN_PATH=$(pwd)
rm libgstcudafilter.so                         #
ln -s ./build/libgstcudafilter-cu.so libgstcudafilter.so 
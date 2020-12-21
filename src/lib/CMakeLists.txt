add_library(liso
            liso_preprocess_thread.cpp)
            
target_link_libraries(liso ${catkin_LIBRARIES} ${PCL_LIBRARIES})
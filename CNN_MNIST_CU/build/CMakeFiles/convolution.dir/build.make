# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/nvidia/CNN_MNIST_CU

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/nvidia/CNN_MNIST_CU/build

# Include any dependencies generated for this target.
include CMakeFiles/convolution.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/convolution.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/convolution.dir/flags.make

CMakeFiles/convolution.dir/scr/Convolution.cu.o: CMakeFiles/convolution.dir/flags.make
CMakeFiles/convolution.dir/scr/Convolution.cu.o: ../scr/Convolution.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nvidia/CNN_MNIST_CU/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/convolution.dir/scr/Convolution.cu.o"
	/usr/local/cuda/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/nvidia/CNN_MNIST_CU/scr/Convolution.cu -o CMakeFiles/convolution.dir/scr/Convolution.cu.o

CMakeFiles/convolution.dir/scr/Convolution.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/convolution.dir/scr/Convolution.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/convolution.dir/scr/Convolution.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/convolution.dir/scr/Convolution.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target convolution
convolution_OBJECTS = \
"CMakeFiles/convolution.dir/scr/Convolution.cu.o"

# External object files for target convolution
convolution_EXTERNAL_OBJECTS =

convolution: CMakeFiles/convolution.dir/scr/Convolution.cu.o
convolution: CMakeFiles/convolution.dir/build.make
convolution: CMakeFiles/convolution.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/nvidia/CNN_MNIST_CU/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable convolution"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/convolution.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/convolution.dir/build: convolution

.PHONY : CMakeFiles/convolution.dir/build

CMakeFiles/convolution.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/convolution.dir/cmake_clean.cmake
.PHONY : CMakeFiles/convolution.dir/clean

CMakeFiles/convolution.dir/depend:
	cd /home/nvidia/CNN_MNIST_CU/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/nvidia/CNN_MNIST_CU /home/nvidia/CNN_MNIST_CU /home/nvidia/CNN_MNIST_CU/build /home/nvidia/CNN_MNIST_CU/build /home/nvidia/CNN_MNIST_CU/build/CMakeFiles/convolution.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/convolution.dir/depend


# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/cdaini/Desktop/Sycomores/MyVGG/CNN_MNIST_CPP

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/cdaini/Desktop/Sycomores/MyVGG/CNN_MNIST_CPP

# Include any dependencies generated for this target.
include CMakeFiles/test_maxpoollayer.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/test_maxpoollayer.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/test_maxpoollayer.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test_maxpoollayer.dir/flags.make

CMakeFiles/test_maxpoollayer.dir/test_maxpoollayer.cpp.o: CMakeFiles/test_maxpoollayer.dir/flags.make
CMakeFiles/test_maxpoollayer.dir/test_maxpoollayer.cpp.o: test_maxpoollayer.cpp
CMakeFiles/test_maxpoollayer.dir/test_maxpoollayer.cpp.o: CMakeFiles/test_maxpoollayer.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/cdaini/Desktop/Sycomores/MyVGG/CNN_MNIST_CPP/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/test_maxpoollayer.dir/test_maxpoollayer.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/test_maxpoollayer.dir/test_maxpoollayer.cpp.o -MF CMakeFiles/test_maxpoollayer.dir/test_maxpoollayer.cpp.o.d -o CMakeFiles/test_maxpoollayer.dir/test_maxpoollayer.cpp.o -c /home/cdaini/Desktop/Sycomores/MyVGG/CNN_MNIST_CPP/test_maxpoollayer.cpp

CMakeFiles/test_maxpoollayer.dir/test_maxpoollayer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_maxpoollayer.dir/test_maxpoollayer.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/cdaini/Desktop/Sycomores/MyVGG/CNN_MNIST_CPP/test_maxpoollayer.cpp > CMakeFiles/test_maxpoollayer.dir/test_maxpoollayer.cpp.i

CMakeFiles/test_maxpoollayer.dir/test_maxpoollayer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_maxpoollayer.dir/test_maxpoollayer.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/cdaini/Desktop/Sycomores/MyVGG/CNN_MNIST_CPP/test_maxpoollayer.cpp -o CMakeFiles/test_maxpoollayer.dir/test_maxpoollayer.cpp.s

# Object files for target test_maxpoollayer
test_maxpoollayer_OBJECTS = \
"CMakeFiles/test_maxpoollayer.dir/test_maxpoollayer.cpp.o"

# External object files for target test_maxpoollayer
test_maxpoollayer_EXTERNAL_OBJECTS =

test_maxpoollayer: CMakeFiles/test_maxpoollayer.dir/test_maxpoollayer.cpp.o
test_maxpoollayer: CMakeFiles/test_maxpoollayer.dir/build.make
test_maxpoollayer: CMakeFiles/test_maxpoollayer.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/cdaini/Desktop/Sycomores/MyVGG/CNN_MNIST_CPP/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test_maxpoollayer"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_maxpoollayer.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test_maxpoollayer.dir/build: test_maxpoollayer
.PHONY : CMakeFiles/test_maxpoollayer.dir/build

CMakeFiles/test_maxpoollayer.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test_maxpoollayer.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test_maxpoollayer.dir/clean

CMakeFiles/test_maxpoollayer.dir/depend:
	cd /home/cdaini/Desktop/Sycomores/MyVGG/CNN_MNIST_CPP && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/cdaini/Desktop/Sycomores/MyVGG/CNN_MNIST_CPP /home/cdaini/Desktop/Sycomores/MyVGG/CNN_MNIST_CPP /home/cdaini/Desktop/Sycomores/MyVGG/CNN_MNIST_CPP /home/cdaini/Desktop/Sycomores/MyVGG/CNN_MNIST_CPP /home/cdaini/Desktop/Sycomores/MyVGG/CNN_MNIST_CPP/CMakeFiles/test_maxpoollayer.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/test_maxpoollayer.dir/depend


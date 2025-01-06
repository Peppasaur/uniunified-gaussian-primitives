# uniunified-gaussian-primitives
a partial implementation of "unified gaussian primitives for scene representation and rendering"

usage:
    g++  -O2  -o o2render *.cpp ../Shading/SurfaceParameters.cpp ../Shading/Disney.cpp ../Shading/Fresnel.cpp ../Shading/Ggx.cpp ../MathLib/*.cpp ../SystemLib/MemoryAllocation.cpp  -I../ -     I/home/qinhaoran/libs/include -L/home/qinhaoran/libs/lib -lfcl -lccd -lcnpy -lz --std=c++11 -pthread


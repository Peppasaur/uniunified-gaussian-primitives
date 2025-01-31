# uniunified-gaussian-primitives
A partial implementation of "unified gaussian primitives for scene representation and rendering"

usage
```bash
cd Gaussian
g++ -O2 -o o2render *.cpp ../Shading/SurfaceParameters.cpp ../Shading/Disney.cpp ../Shading/Fresnel.cpp ../Shading/Ggx.cpp ../MathLib/*.cpp ../SystemLib/MemoryAllocation.cpp -I../ -I/home/qinhaoran/libs/include -L/home/qinhaoran/libs/lib -lfcl -lccd -lcnpy -lz --std=c++11 -pthread
./o2render'''

![image](https://github.com/user-attachments/assets/c3a24c8d-bab6-4650-b003-83461b39d1b3)
![image](https://github.com/user-attachments/assets/6e6182c8-9cb7-4b34-9d73-fbe3b4cba207)
![image](https://github.com/user-attachments/assets/993ddd03-7c5c-42d3-a05a-a826e88ccd57)

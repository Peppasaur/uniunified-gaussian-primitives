A partial implementation of "unified gaussian primitives for scene representation and rendering"

usage
```bash
cd Gaussian
g++ -O2 -o o2render *.cpp ../Shading/SurfaceParameters.cpp ../Shading/Disney.cpp ../Shading/Fresnel.cpp ../Shading/Ggx.cpp ../MathLib/*.cpp ../SystemLib/MemoryAllocation.cpp -I../ -I/home/qinhaoran/libs/include -L/home/qinhaoran/libs/lib -lfcl -lccd -lcnpy -lz --std=c++11 -pthread
./o2render
```

<p align="center">
  <img src="https://github.com/user-attachments/assets/edb5b2b9-2117-46d9-a257-5409e5803735" width="200">
  <img src="https://github.com/user-attachments/assets/24eb121f-de21-4bdf-93fd-3bcb628a812f" width="200">
  <img src="https://github.com/user-attachments/assets/0f12d1b9-ac65-4014-b629-513618cfcb41" width="200">
</p>

![image](https://github.com/user-attachments/assets/0f12d1b9-ac65-4014-b629-513618cfcb41)

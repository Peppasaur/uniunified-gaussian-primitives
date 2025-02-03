A partial implementation of "unified gaussian primitives for scene representation and rendering"

The paper "Gaussian General Scene Representation" proposes a differentiable rendering framework that assigns PBR parameters to each 3D Gaussian primitive and performs ray sampling on the primitives, applying BSDF to the sampled primitives. At the same time, the normal distribution of each Gaussian primitive is not fixed, but follows a VNDF distribution based on SGGX. As a result, this method can represent objects with different characteristics.

usage
```bash
cd Gaussian
g++ -O2 -o o2render *.cpp ../Shading/SurfaceParameters.cpp ../Shading/Disney.cpp ../Shading/Fresnel.cpp ../Shading/Ggx.cpp ../MathLib/*.cpp ../SystemLib/MemoryAllocation.cpp -I../ -I/home/qinhaoran/libs/include -L/home/qinhaoran/libs/lib -lfcl -lccd -lcnpy -lz --std=c++11 -pthread
./o2render
```

This renderer supports different types of material
<p align="center">
  <img src="https://github.com/user-attachments/assets/edb5b2b9-2117-46d9-a257-5409e5803735" width="200">
  <img src="https://github.com/user-attachments/assets/24eb121f-de21-4bdf-93fd-3bcb628a812f" width="200">
  <img src="https://github.com/user-attachments/assets/0f12d1b9-ac65-4014-b629-513618cfcb41" width="200">
</p>

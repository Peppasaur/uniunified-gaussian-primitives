![image](https://github.com/user-attachments/assets/71d6b6f4-1881-4ff7-86c4-495fab29090f)# uniunified-gaussian-primitives
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
  <img src="https://github.com/user-attachments/assets/4b2cb40e-5d34-46c6-a9c0-b9acd92873e4" width="200">
</p>

![image](https://github.com/user-attachments/assets/edb5b2b9-2117-46d9-a257-5409e5803735)
![image](https://github.com/user-attachments/assets/24eb121f-de21-4bdf-93fd-3bcb628a812f)
![image](https://github.com/user-attachments/assets/4b2cb40e-5d34-46c6-a9c0-b9acd92873e4)

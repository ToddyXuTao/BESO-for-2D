# BESO for 2D
A basic topological optimization program using soft-kill BESO technique.
![image](img scr="https://github.com/ToddyXuTao/BESO-for-2D/blob/master/Examples/Introduction.png")
![image](img scr=https://github.com/ToddyXuTao/BESO-for-2D/blob/master/Examples/Long_Cantilever_Hires_History.png)

## Introduction
The program was initially written in Jupyter-lab and emerged to a single .py file.
All the necessary parameters can be modified in 1172-1250 lines.

### Predefined Load Conditions
- Half beam
- Beam
- Cantilever
- Michell structure with two simple supports
- Michell structure with one simple support and one roller

### Predefine Finite Element solvers
- Numpy solver (Slow, unefficient but formidable)
- Cvxopt solver (Recommended)

## References
- Huang, X. and Xie, M., 2010. Evolutionary topology optimization of continuum structures: methods and applications. John Wiley & Sons.
- https://github.com/williamhunter/topy
- https://github.com/worbit/topopt2D
- https://github.com/1iyiwei/topopt

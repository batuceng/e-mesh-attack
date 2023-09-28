import numpy as np
import matplotlib

import matplotlib.pyplot as plt

A = np.array([10,0, 0])
B = np.array([5,5, 0])
C = np.array([2,2,6])

triangle = np.array([A,B,C,A])

print(triangle)

xs, ys, zs = zip(*triangle)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.plot(xs,ys, zs) 


ax.text(A[0], A[1], A[2], "A")
ax.text(B[0], B[1], B[2], "B")
ax.text(C[0], C[1], C[2], "C")


#P = np.array([-1,3,8])
P = np.array([0,0,10])

ax.scatter(P[0], P[1],  P[2], color='r')

#plt.savefig("x_3d.png")
#exit()


res = P.copy()

AB = B-A
AC = C-A


AP = P-A
dA1 = np.dot(AB,AP)
dA2 = np.dot(AC,AP)

if( dA1<=0 and dA2 <=0):
    res = A

BP = P - B
dB1 = np.dot(AB, BP)
dB2 = np.dot(AC, BP)

if( dB1 >= 0 and dB2 <=0 ):
    res = B

CP = P - C
dC1 = np.dot(AB,CP)
dC2 = np.dot(AC,CP)
if( dC2 >= 0 and dC1 <= dC2 ):
    res = C

EdgeAB = dA1*dB2 - dB1*dA2
if( EdgeAB <= 0 and dA1 >= 0 and dB1 <=0):
   AP = P - A
   AB = B - A
   
   res = (np.dot(AP,AB)/np.dot(AB,AB) * AB) + A
   
   print("Project to AB")


EdgeBC = dB1*dC2 - dC1*dB2; 
if( EdgeBC <= 0 and (dB2-dB1)>=0 and (dC1-dC2)>=0):

   BP = P - B
   BC = C - B
   
   res = (np.dot(BP,BC)/np.dot(BC,BC) * BC) + B

   print("Project to BC")

EdgeAC = dC1*dA2 - dA1*dC2
if( EdgeAC <= 0 and dA2>=0 and dC2<=0 ):
    
   AP = P - A
   AC = C - A
   
   res = (np.dot(AP,AC)/np.dot(AC,AC) * AC) + A
   
   print("Project to AC")

   #lies in edge region AC bary coords (1-u, 0 ,u )
   #project point onto edge AC and return result



ax.scatter(res[0], res[1], res[2], color='g')


print(res)

#plt.annotate("A", (A[0], A[1]))
#plt.annotate("B", (B[0], B[1]))
#plt.annotate("C", (C[0], C[1]))

plt.savefig("x_3d.png")

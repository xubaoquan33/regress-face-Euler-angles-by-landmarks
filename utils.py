import math
import numpy as np

'''
a1 = cos(z)*cos(y)*a + (-sin(z)*cos(x) + cos(z)*sin(y)*sin(x))*b + (sin(z)*sin(x) + cos(z)*sin(y)*cos(x))*c
b1 = sin(z)*cos(y)*a + (cos(z)*cos(x) + sin(z)*sin(y)*sin(x))*b + (-cos(z)*sin(x) + sin(z)*sin(y)*cos(x))*c
c1 = (-sin(y))*a + cos(y)*sin(x)*b + cos(y)*cos(x)*c 

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
roll x

loss = (a1-a2)^2 + (b1-b2)^2 + (c1-c2)^2
dloss_dx = 2*(a1-a2)*da1/dx + 2*(b1-b2)*db1/dx + 2*(c1-c2)*dt1/dx

da1/dx = (sin(z)*sin(x) + cos(z)*sin(y)*cos(x))*b + (sin(z)*cos(x) - cos(z)*sin(y)*sin(x))*c
db1/dx = (-cos(z)*sin(x) + sin(z)*sin(y)*cos(x))*b + (-cos(z)*cos(x) - sin(z)*sin(y)*sin(x))*c
dc1/dx = cos(y)*cos(x)*b - cos(y)*sin(x)*c 
x = x-dloss_dx

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
pitch y

loss = (a1-a2)^2 + (b1-b2)^2 + (c1-c2)^2
dloss_dy = 2*(a1-a2)*da1/dy + 2*(b1-b2)*db1/dy + 2*(c1-c2)*dt1/dy

da1/dy = -cos(z)*sin(y)*a + ( cos(z)*cos(y)*sin(x))*b + ( cos(z)*cos(y)*cos(x))*c
db1/dy = -sin(z)*sin(y)*a + (sin(z)*cos(y)*sin(x))*b + (sin(z)*cos(y)*cos(x))*c
dc1/dy = (-cos(y))*a - sin(y)*sin(x)*b - sin(y)*cos(x)*c 
y = y-dloss_dy

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
yaw z

loss = (a1-a2)^2 + (b1-b2)^2 + (c1-c2)^2
dloss_dx = 2*(a1-a2)*da1/dz + 2*(b1-b2)*db1/dz + 2*(c1-c2)*dt1/dz

da1/dz = -sin(z)*cos(y)*a + (-cos(z)*cos(x) - sin(z)*sin(y)*sin(x))*b + (cos(z)*sin(x) - sin(z)*sin(y)*cos(x))*c
db1/dz = cos(z)*cos(y)*a + (-sin(z)*cos(x) + cos(z)*sin(y)*sin(x))*b + (sin(z)*sin(x) + cos(z)*sin(y)*cos(x))*c
dc1/dz = 0
z=z-dloss_dz

'''
def eulerAnglesToRotationMatrix(theta) :
     
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])     
    
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
                 
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
                                     
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R

def angle2matrix(angles):
    z=angles[2]
    y=angles[1]
    x=angles[0]
    R=np.array([[math.cos(z)*math.cos(y), -math.sin(z)*math.cos(x)+math.cos(z)*math.sin(y)*math.sin(x), 
                 math.sin(z)*math.sin(x)+math.cos(z)*math.sin(y)*math.cos(x)],
                [math.sin(z)*math.cos(y), math.cos(z)*math.cos(x)+math.sin(z)*math.sin(y)*math.sin(x),  
                 -math.cos(z)*math.sin(x)+math.sin(z)*math.sin(y)*math.cos(x)],
                [-math.sin(y),       math.cos(y)*math.sin(x),           math.cos(y)*math.cos(x)  ]])
    return R

def roll(angle):
    R_x = np.array([[1,         0,                  0                   ],
                [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                ])
    return R_x


         
def pitch(angle):
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
    return R_y

def yaw(angle):
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                [math.sin(theta[2]),    math.cos(theta[2]),     0],
                [0,                     0,                      1]
                ])
    return R_z


def mse_loss(input,target):
    result=[]
    for i in range(input.shape[0]):
        dif=input[i]-target[i]
        temp=np.dot(dif,dif)
        result.append(temp)
    return sum(result)

def rotate(input,angles):
    R=eulerAnglesToRotationMatrix(theta)
    output=np.zeros(input.shape)
    for i in range(input.shape[0]):
        output[i]=np.dot(R,input[i])
    return output

def roll_descent(input,output,target,angles):
    total=[]
    for i in range(input.shape[0]):
        a=input[i][0]
        b=input[i][1]
        c=input[i][2]
        a1=output[i][0]
        a1=output[i][1]
        b1=output[i][2]
        a2=target[i][0]
        b2=target[i][1]
        c2=target[i][2]
        
        da1_dx = (math.sin(z)* math. math.sin(x) +  math.cos(z)* math.sin(y)* math.cos(x))*b + 
                    ( math.sin(z)* math.cos(x) -  math.cos(z)* math.sin(y)* math.sin(x))*c
        db1_dx = (-math.cos(z)* math.sin(x) +  math.sin(z)* math.sin(y)* math.cos(x))*b + 
                    (- math.cos(z)* math.cos(x) -  math.sin(z)* math.sin(y)* math.sin(x))*c
        dc1_dx =  math.cos(y)* math.cos(x)*b -  math.cos(y)* math.sin(x)*c 
        
        dloss_dx = 2*(a1-a2)*da1_dx + 2*(b1-b2)*db1_dx + 2*(c1-c2)*dt1_dx
        total.append(dloss_dx)
    dx=sum(total)/input.shape[0]
    return dx

def pitch_descent(input,output,target,angles):
    total=[]
    for i in range(input.shape[0]):
        a=input[i][0]
        b=input[i][1]
        c=input[i][2]
        a1=output[i][0]
        a1=output[i][1]
        b1=output[i][2]
        a2=target[i][0]
        b2=target[i][1]
        c2=target[i][2]
        
        da1_dy = -math.cos(z)*math.sin(y)*a + (math.cos(z)*math.cos(y)*math.sin(x))*b + 
                    (math.cos(z)*math.cos(y)*math.cos(x))*c
        db1_dy = -math.sin(z)*math.sin(y)*a + (math.sin(z)*math.cos(y)*math.sin(x))*b + 
                    (math.sin(z)*math.cos(y)*math.cos(x))*c
        dc1_dy = (-math.cos(y))*a - math.sin(y)*math.sin(x)*b - math.sin(y)*math.cos(x)*c 
        
        dloss_dy = 2*(a1-a2)*da1_dy + 2*(b1-b2)*db1_dy + 2*(c1-c2)*dt1_dy
        total.append(dloss_dy)
    dy=sum(total)/input.shape[0]
    return dy

def yaw_descent(input,output,target,angles):
    total=[]
    for i in range(input.shape[0]):
        a=input[i][0]
        b=input[i][1]
        c=input[i][2]
        a1=output[i][0]
        a1=output[i][1]
        b1=output[i][2]
        a2=target[i][0]
        b2=target[i][1]
        c2=target[i][2]
        
        da1_dz = -math.sin(z)*math.cos(y)*a + (-math.cos(z)*cos(x) - math.sin(z)*math.sin(y)*math.sin(x))*b + 
                    (math.cos(z)*math.sin(x) - math.sin(z)*math.sin(y)*math.cos(x))*c
        db1_dz = math.cos(z)*cos(y)*a + (-math.sin(z)*cos(x) + math.cos(z)*math.sin(y)*sin(x))*b + 
                    (math.sin(z)*math.sin(x) + math.cos(z)*math.sin(y)*math.cos(x))*c
        dc1_dz = 0
        
        dloss_dz = 2*(a1-a2)*da1_dz + 2*(b1-b2)*db1_dz + 2*(c1-c2)*dt1_dz
        total.append(dloss_dz)
    dz=sum(total)/input.shape[0]
    return dz

def regress_angle(input,target):
    angles=[0,0,0]
    i=100
    min_error=10000
    best_angle=[0,0,0]
    while i>0 :
        output=rotate(input,angles)
        error=mse_loss(input,target)
        if error<min_error:
            min_error=error
            best_angle=angles
            i=100
        else:
            i-=1
        dx=roll_descent(input,output,target,angles)
        dy=pitch_descent(input,output,target,angles)
        dz=yaw_descent(input,output,target,angles)
        angles[0]-=dx
        angles[1]-=dy
        angles[2]-=dz
    return best_angles
        

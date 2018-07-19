#coding=utf-8

import math
import numpy as np
import scipy.io as sio
from skimage import io
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
                                     
    R = np.dot(R_x, np.dot( R_y, R_z))
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

def pitch(angle):
    R_x = np.array([[1,         0,                  0                   ],
                [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                ])
    return R_x


         
def yaw(angle):
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
    return R_y

def roll(angle):
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
    return sum(result)/input.shape[0]

def rotate(input,angles):
    R=eulerAnglesToRotationMatrix(angles)
    output=np.zeros(input.shape)
    for i in range(input.shape[0]):
        output[i]=np.dot(R,input[i])
    return output

def pitch_descent(input,output,target,angles):
    total=[]
    for i in range(input.shape[0]):
        a=input[i][0]
        b=input[i][1]
        c=input[i][2]
        a1=output[i][0]
        b1=output[i][1]
        c1=output[i][2]
        a2=target[i][0]
        b2=target[i][1]
        c2=target[i][2]
        x=angles[0]
        y=angles[1]
        z=angles[2]
        da1_dx = (math.sin(z)* math.sin(x) +  math.cos(z)* math.sin(y)* math.cos(x))*b +( math.sin(z)* math.cos(x) -  math.cos(z)* math.sin(y)* math.sin(x))*c
        db1_dx = (-math.cos(z)* math.sin(x) +  math.sin(z)* math.sin(y)* math.cos(x))*b  +(- math.cos(z)* math.cos(x) -  math.sin(z)* math.sin(y)* math.sin(x))*c
        dc1_dx =  math.cos(y)* math.cos(x)*b -  math.cos(y)* math.sin(x)*c 
        
        dloss_dx = 2*(a1-a2)*da1_dx + 2*(b1-b2)*db1_dx + 2*(c1-c2)*dc1_dx
        total.append(dloss_dx)
    dx=sum(total)/input.shape[0]
    return dx

def yaw_descent(input,output,target,angles):
    total=[]
    for i in range(input.shape[0]):
        a=input[i][0]
        b=input[i][1]
        c=input[i][2]
        a1=output[i][0]
        b1=output[i][1]
        c1=output[i][2]
        a2=target[i][0]
        b2=target[i][1]
        c2=target[i][2]
        x=angles[0]
        y=angles[1]
        z=angles[2]
        da1_dy = -math.cos(z)*math.sin(y)*a + (math.cos(z)*math.cos(y)*math.sin(x))*b + (math.cos(z)*math.cos(y)*math.cos(x))*c
        db1_dy = -math.sin(z)*math.sin(y)*a + (math.sin(z)*math.cos(y)*math.sin(x))*b + (math.sin(z)*math.cos(y)*math.cos(x))*c
        dc1_dy = (-math.cos(y))*a - math.sin(y)*math.sin(x)*b - math.sin(y)*math.cos(x)*c 
        
        dloss_dy = 2*(a1-a2)*da1_dy + 2*(b1-b2)*db1_dy + 2*(c1-c2)*dc1_dy
        total.append(dloss_dy)
    dy=sum(total)/input.shape[0]
    return dy

def roll_descent(input,output,target,angles):
    total=[]
    for i in range(input.shape[0]):
        a=input[i][0]
        b=input[i][1]
        c=input[i][2]
        a1=output[i][0]
        b1=output[i][1]
        c1=output[i][2]
        a2=target[i][0]
        b2=target[i][1]
        c2=target[i][2]
        x=angles[0]
        y=angles[1]
        z=angles[2]
        da1_dz = -math.sin(z)*math.cos(y)*a + (-math.cos(z)*math.cos(x) - math.sin(z)*math.sin(y)*math.sin(x))*b  + (math.cos(z)*math.sin(x) - math.sin(z)*math.sin(y)*math.cos(x))*c
        db1_dz = math.cos(z)*math.cos(y)*a + (-math.sin(z)*math.cos(x) + math.cos(z)*math.sin(y)*math.sin(x))*b  + (math.sin(z)*math.sin(x) + math.cos(z)*math.sin(y)*math.cos(x))*c
        dc1_dz = 0
        
        dloss_dz = 2*(a1-a2)*da1_dz + 2*(b1-b2)*db1_dz + 2*(c1-c2)*dc1_dz
        total.append(dloss_dz)
    dz=sum(total)/input.shape[0]
    return dz

def regress_angles(input,target):
    angles=[0,0,0]
    i=100
    min_error=10000
    best_angles=[0,0,0]
    lr=1
    n=0
    while i>0 :
        output=rotate(input,angles)
        error=mse_loss(input,target)
        if error<min_error:
            min_error=error
            best_angles=angles
            i=100
        else:
            i-=1
        dx=pitch_descent(input,output,target,angles)
        #print dx
        dy=yaw_descent(input,output,target,angles)
        #print dy
        dz=roll_descent(input,output,target,angles)
        #print dz
        angles[0]-=lr*dx
        angles[1]-=lr*dy
        angles[2]-=lr*dz
        '''
        for i in range(3):
            if angles[i]>2*np.pi:
                angles[i]-=2*np.pi
            if angles[i]<-2*np.pi:
                angles[i]+=2*np.pi
        '''
        n+=1

    return best_angles
        
def normalize(landmarks):

    center=landmarks[30]
    landmarks[:,0]-=center[0]
    landmarks[:,1]-=center[1]
    landmarks[:,2]-=center[2]
    dif=landmarks[30]-landmarks[27]
    l=np.sqrt(np.dot(dif,dif))
    landmarks=landmarks/l

    return landmarks

def get_full_model_points(filename='/train/execute/face-alignment/facemodel68.txt'):
    """Get all 68 3D model points from file"""
    raw_value = []
    with open(filename) as file:
        for line in file:
            raw_value.append(line)
    model_points = np.array(raw_value, dtype=np.float32)
    model_points = np.reshape(model_points, (3, -1)).T
    # model_points *= 4
    model_points[:, -1] *= -1
    return model_points

def get_full_model_points_plus(filename='/train/execute/face-alignment/landmark68.txt'):
    raw_value = []

    with open(filename) as file:
        for line in file:
            temp=map(float,line.split('/n')[0].split(','))
            raw_value.append(temp)
    landmarks=np.array(raw_value)
    return landmarks

def error(m,n):
    '''
    k1=abs(m[0])-abs(n[0])
    k2=abs(m[1])-abs(n[1])
    k3=abs(m[2])-abs(n[2])
    '''
    k1=m[0]-n[0]
    k2=m[1]-n[1]
    k3=m[2]-n[2]    
    return (k1*k1+k2*k2+k3*k3)/3
    #return k1*k1
    
#间隔10，测试误差37，循环次数18*18*18
def search_angles(input,target):
    min_error=10000
    best_angles=[0,0,0]
    for i in range(-90,100,10):
        for j in range(-90,100,10):
            for k in range(-90,100,10):
                angles=[i*np.pi/180,j*np.pi/180,k*np.pi/180]
                output=rotate(input,angles)
                error=mse_loss(output,target)
                if error<min_error:
                    min_error=error
                    best_angles=angles
    return best_angles

#二分查找，误差较大，在最后的角度回归中，容易匹配到错误的方向，导致偏离正确结果
#出现一个问题，当我们以及提供了正确的角度，但是有偏差的结果loss比正确答案低，导致无法回归正确的角度
#二分停止参数设为10：AFLW error 43

def binary_search(input,target):
    min_error=-1
    best_angles=[0,0,0]
    x=[-45*np.pi/180,45*np.pi/180]
    y=[-45*np.pi/180,45*np.pi/180]
    z=[-45*np.pi/180,45*np.pi/180]
    d=110*np.pi/180
    while d>np.pi/180:
        for i in x:
            for j in y:
                for k in z:
                    angles=[i,j,k]
                    output=rotate(input,angles)
                    error=mse_loss(output,target)
                    if min_error==-1:
                        min_error=error
                        best_angles=angles
                    else:
                        if error<min_error:
                            min_error=error
                            best_angles=angles
    
        min_error=-1
        d/=2
        x=[best_angles[0]-d/2,best_angles[0]+d/2]
        y=[best_angles[1]-d/2,best_angles[1]+d/2]
        z=[best_angles[2]-d/2,best_angles[2]+d/2]
    '''
    #print best_angles[0]*180/np.pi,best_angles[1]*180/np.pi,best_angles[2]*180/np.pi
    l=1.5*np.pi/180
    n=int(d*2//l)
    #print n
    start_x=best_angles[0]-d
    start_y=best_angles[1]-d
    start_z=best_angles[2]-d
    min_error=-1
    for i in range(n+1):
        for j in range(n+1):
            for k in range(n+1):
                angles=[start_x+i*l,start_y+j*l,start_z+k*l]
                #print angles[0]*180/np.pi,angles[1]*180/np.pi,angles[2]*180/np.pi
                output=rotate(input,angles)
                error=mse_loss(output,target)
                if min_error==-1:
                    min_error=error
                    best_angles=angles
                else:
                    if error<min_error:
                        min_error=error
                        best_angles=angles
                #print best_angles[0]*180/np.pi,best_angles[1]*180/np.pi,best_angles[2]*180/np.pi
    '''
    return best_angles

                    
        
def run():
    model_points_68 = get_full_model_points()
    model_points_68_1 = get_full_model_points_plus()
    model_points_68=normalize(model_points_68)
    model_points_68_1=normalize(model_points_68_1)
    file = open('/train/results/AFLW2000/filepath_list.txt','r')
    file_list= file.read().splitlines()
    length=len(file_list)
    total=[]
    t=1
    print 'here'
    pitch_s1=[]
    yaw_s1=[]
    roll_s1=[]
    pitch_s2=[]
    yaw_s2=[]
    roll_s2=[]
    yaw_error = .0
    pitch_error = .0
    roll_error = .0
    for path in file_list:
        input_path=path + '.jpg'
        mat_path=path+'.mat'
        input = io.imread(input_path)
        #print input.shape
        mat = sio.loadmat(mat_path)
        # [pitch yaw roll tdx tdy tdz scale_factor]
        pre_pose_params = mat['Pose_Para'][0]

        # Get [pitch, yaw, roll, tdx, tdy]
        pose_params = pre_pose_params[:3]*180/np.pi

        preds_m=mat['pt3d_68']
        preds=np.zeros([68,3])
        for i in range(68):
            preds[i]=np.array([preds_m[0,i],preds_m[1,i],preds_m[2,i]])
        preds=normalize(preds)
        #print model_points_68
        #print preds
        angles=np.array(binary_search(model_points_68_1,preds))
        angles = angles*180/np.pi
        a=[angles[0],-angles[1],angles[2]]
        '''
        suberror=error(a,pose_params)
        if suberror>60:
            print a,pose_params
            pitch_s1.append(abs(a[0]))
            yaw_s1.append(abs(a[1]))
            roll_s1.append(abs(a[2]))
        if suberror<20:
            print a,pose_params
            pitch_s2.append(abs(a[0]))
            yaw_s2.append(abs(a[1]))
            roll_s2.append(abs(a[2]))
        total.append(suberror)
        '''
        pitch_error += abs(pose_params[0] - a[0])
        yaw_error += abs(pose_params[1] - a[1] )
        roll_error += abs(pose_params[2] - a[2])
        
    print('Test error in degrees of the dgface on the ' + str(length) +
        ' test images. Yaw: %.4f, Pitch: %.4f, Roll: %.4f' % (yaw_error / length,
        pitch_error / length, roll_error / length))


    '''
    result=sum(total)/length
    result1=sum(pitch_s1)/len(pitch_s1)
    result2=sum(yaw_s1)/len(yaw_s1)
    result3=sum(roll_s1)/len(roll_s1)
    result4=sum(pitch_s2)/len(pitch_s2)
    result5=sum(yaw_s2)/len(yaw_s2)
    result6=sum(roll_s2)/len(roll_s2)
    print result
    print result1,result2,result3
    print result4,result5,result6
    '''

    
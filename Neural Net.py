####                    Understanding neural nets in Python               ####

### A simple gate
def forwardMultiplyGate(x,y):
	return x*y


### Strategy 1 : Random search to find the best 'x' and 'y' to increase the output by tweaking a little...
### This takes large no of iterations to get the best x and y values..(Expensive)
from random import uniform           ### uniform generates floating point numbers
x,y=-2,3
tweak_amt=0.01
best_out=-9
best_x,best_y=x,y
for i in range(100):
	x_try=x+tweak_amt*(uniform(0,1)*2-1)
	y_try=y+tweak_amt*(uniform(0,1)*2-1)
	out=forwardMultiplyGate(x_try,y_try)
	if out>best_out:
		best_out=out
		best_x,best_y=x_try,y_try

print "Random Search approach : Best x,y,out",best_x,best_y,best_out


###  Strategy 2 : Numeric gradient
### This approach computes gradient for each input of the network..(A little expensive but better than previous approach)
out=forwardMultiplyGate(x,y)
h=0.0001

xph=x+h
out2=forwardMultiplyGate(xph,y)
x_derivative=(out2-out)/h

yph=y+h
out3=forwardMultiplyGate(x,yph)
y_derivative=(out3-out)/h


### Change the inputs based on the derivative..
step=0.01
out=forwardMultiplyGate(x,y)
x=x+step*x_derivative
y=y+step*y_derivative
out=forwardMultiplyGate(x,y)
print "Numeric Gradient approach : Best x,y,output :",x,y,out 


### Strategy 3 : ANALYTIC GRADIENT
### This approach computes the gradient without tweaking any input by using calculus...(Efficient of all..)
### For function f(x,y)=x*y,df/dx=y and df/dy=x
x,y=-2,3
x_derivative=y
y_derivative=x
x=x+step*x_derivative
y=y+step*y_derivative
out=forwardMultiplyGate(x,y)
print "Analytic Gradient approach : Best x,y,out :",x,y,out

### Multiple gates...

def forwardAddGate(x,y):
	return x+y
def forwardCircuit(x,y,z):
	a=forwardAddGate(x,y)
	b=forwardMultiplyGate(a,z)
	return b

x,y,z=-2,5,-4
q=forwardAddGate(x,y)
out=forwardCircuit(x,y,z)
print "### For Multiple Gates ###"
print "Input :",x,y,z,"And Output is ",out

### Derivative is calculated using chain rule: df/dx=df/dq*dq/dx,df/dy=df/dq*dq/dy,df/dz=x+y

derivative_f_wrt_q=z
derivative_f_wrt_z=q

derivative_q_wrt_x=1
derivative_q_wrt_y=1

derivative_f_wrt_x=derivative_f_wrt_q*derivative_q_wrt_x
derivative_f_wrt_y=derivative_f_wrt_q*derivative_q_wrt_y

# step=0.01
### Changing the inputs based on the derivative..
x=x+step*derivative_f_wrt_x
y=y+step*derivative_f_wrt_y
z=z+step*derivative_f_wrt_z
out=forwardCircuit(x,y,z)
print "Gradients ",derivative_f_wrt_x,derivative_f_wrt_y,derivative_f_wrt_z
print "Best Input(x,y,z) :",x,y,z,"Output : ",out

### Cross-checking with Numeric gradient...
x,y,z=-2,5,-4
h=0.0001
x_grad=(forwardCircuit(x+h,y,z)-forwardCircuit(x,y,z))/h
y_grad=(forwardCircuit(x,y+h,z)-forwardCircuit(x,y,z))/h
z_grad=(forwardCircuit(x,y,z+h)-forwardCircuit(x,y,z))/h

print "Gradient checking using Numeric gradient Approach : ",x_grad,y_grad,z_grad

### A Single Neuron implementation...without using classes..

import math
def sigmoid(x):
	return 1/(1+math.exp(-x))

a,b,c,x,y=1,2,-3,-1,3
ax=forwardMultiplyGate(a,x)
by=forwardMultiplyGate(b,y)
axpby=forwardAddGate(ax,by)
axpbypc=forwardAddGate(axpby,c)
sig=sigmoid(axpbypc)
print "Single Neuron - Sigmoid"
print "Output ",sig

### Gradient calculation...

sig_grad_wrt_axpbypc=sig*(1-sig)
sig_grad_wrt_axpby=1*sig_grad_wrt_axpbypc
sig_grad_wrt_c=1*sig_grad_wrt_axpbypc
sig_grad_wrt_ax=1*sig_grad_wrt_axpby
sig_grad_wrt_by=1*sig_grad_wrt_axpby
sig_grad_wrt_a=x*sig_grad_wrt_ax
sig_grad_wrt_x=a*sig_grad_wrt_ax
sig_grad_wrt_b=y*sig_grad_wrt_by
sig_grad_wrt_y=b*sig_grad_wrt_by

print "Gradients wrt a,b,c,x,y :",sig_grad_wrt_a,sig_grad_wrt_b,sig_grad_wrt_c,sig_grad_wrt_x,sig_grad_wrt_y


### Using Python classes

class Unit:
	
	def __init__(self,val,grad):
		self.val=val
		self.grad=grad

class multiply:

	def forward(self,u0,u1):
		self.u0=u0
		self.u1=u1
		self.utop=Unit(u0.val*u1.val,0.0)
		return self.utop
	def backward(self):
		self.u0.grad+=self.u1.val*self.utop.grad
		self.u1.grad+=self.u0.val*self.utop.grad

class add:
	
	def forward(self,u0,u1):
		self.u0=u0
		self.u1=u1
		self.utop=Unit(u0.val+u1.val,0.0)
		return self.utop
	def backward(self):
		self.u0.grad+=1*self.utop.grad
		self.u1.grad+=1*self.utop.grad

class sig:
	def sig(self,u0):
		return 1/(1+math.exp(-u0.val))
	def forward(self,u0):
		self.u0=u0
		temp=self.sig(u0)
		self.utop=Unit(temp,0.0)
		return self.utop
	def backward(self):
		s=sigmoid(self.u0.val)
		self.u0.grad+=(s*(1-s))*self.utop.grad

a=Unit(1.0,0.0)
b=Unit(2.0,0.0)
c=Unit(-3.0,0.0)
x=Unit(-1.0,0.0)
y=Unit(3.0,0.0)

mul1=multiply()
mul2=multiply()
add1=add()
add2=add()
sg=sig()

ax=mul1.forward(a,x)
by=mul2.forward(b,y)
axpby=add1.forward(ax,by)
axpbypc=add2.forward(axpby,c)
ss=sg.forward(axpbypc)

print ss.val,ss.grad

### Backpropagation... ss contains output of sigmoid gate..Its gradient should be 1
ss.grad=1
sg.backward()
add2.backward()
add1.backward()
mul2.backward()
mul1.backward()

print a.grad,b.grad,c.grad,x.grad,y.grad

print "Old values :",a.val,b.val,c.val,x.val,y.val
### Changing inputs according to the gradients...
step=0.01
a.val+=step*a.grad
b.val+=step*b.grad
c.val+=step*c.grad
x.val+=step*x.grad
y.val+=step*y.grad


print "New values :",a.val,b.val,c.val,x.val,y.val
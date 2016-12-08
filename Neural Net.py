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
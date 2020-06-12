#!/anaconda3/envs/fenicsproject/bin/python

# this was grabbed as a work in progress - use it here, but nowhere else.

from fenics import *

'''
Collected FEniCS Expressions for isotropic homogeneous constant coefficient linear
elastostatic 2D/3D problems. A time parameter is allowed to enable quasistatic
solutions. The available classes are:

- lin_lin_2D  for a linear in space and linear in time 2D solution
- non_lin_2D  for a nonlinear in space and linear in time 2D solution
- lin_lin_3D  for a linear in space and linear in time 3D solution
- non_lin_3D  for a nonlinear in space and linear in time 3D solution

Each contains classes for 

- u,    the exact displacement
- epsx, the exact strain
- sigx, the exact stress
- f,    the corresponding body forces
- g,    an empty placeholder for the traction (unused for exact solutions)

Need implementation examples here for vectors and tensors

Remarks:
 - coeffs is passed as a dictionary, e.g. coeffs = {"a":1,"b":2,"c":3}


'''

class lin_lin_2D:
    '''
    FEniCS Expressions for Linear in space and time 2D solution data 
    '''
    
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    class u(UserExpression):
        '''
        u - linear in space, linear in time, 2D
        '''
        #def __init__(self, degree, a, b, t, **kwargs):
            #super().__init__(**kwargs)
            #self.degree = degree
            #self.a      = a
            #self.b      = b
            #self.t      = t
        
        def __init__(self, coeffs, degree, t, **kwargs):
            super().__init__(**kwargs)
            self.degree = degree
            self.a      = coeffs["a"]
            self.b      = coeffs["b"]
            self.t      = t
        
        def eval(self, values, x):
            values[0] = (x[0]+x[1])*(self.a*self.t+1.0)
            values[1] = (x[0]-x[1])*(self.b*self.t-1.0)
    
        def value_shape(self):
            return (2,)
    
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    class epsx(UserExpression):
        '''
        strain, for u - linear in space, linear in time, 2D
        '''
        def __init__(self, coeffs, degree, t, **kwargs):
            super().__init__(**kwargs)
            self.degree = degree
            self.a      = coeffs["a"]
            self.b      = coeffs["b"]
            self.t      = t
        
        def eval(self, values, x):
            values[0] = self.a*self.t+1.0
            values[1] = self.t*(self.a+self.b)/2.0
            values[2] = self.t*(self.a+self.b)/2.0
            values[3] = 1.0-self.b*self.t
            
        def value_shape(self):
            return (2,2)
        
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    class sigx(UserExpression):
        '''
        stress, for u - linear in space, linear in time, 2D
        '''
        def __init__(self, coeffs, degree, t, lmbda, G, **kwargs):
            super().__init__(**kwargs)
            self.degree = degree
            self.a      = coeffs["a"]
            self.b      = coeffs["b"]
            self.t      = t
            self.lmbda  = lmbda
            self.G      = G
        
        def eval(self, values, x):
            values[0] = self.lmbda*(self.a*self.t-self.b*self.t+2.0)+self.G*(self.a*self.t+1.0)*2.0
            values[1] = self.G*self.t*(self.a+self.b)
            values[2] = self.G*self.t*(self.a+self.b)
            values[3] = self.lmbda*(self.a*self.t-self.b*self.t+2.0)-self.G*(self.b*self.t-1.0)*2.0
            
        def value_shape(self):
            return (2,2)
    
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    class f(UserExpression):
        '''
        body force, for u - linear in space, linear in time, 2D
        '''
        def __init__(self, coeffs, degree, t, lmbda, G, **kwargs):
            super().__init__(**kwargs)
            self.degree = degree
            self.a      = coeffs["a"]
            self.b      = coeffs["b"]
            self.t      = t
            self.lmbda  = lmbda
            self.G      = G
        
        def eval(self, values, x):
            values[0] = 0.0
            values[1] = 0.0
            
        def value_shape(self):
            return (2,)
    
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    class g(UserExpression):
        '''
        surface traction, not used for exact solutions (use \sigma\dot n instead)
        '''
        def __init__(self, coeffs, degree, t, lmbda, G, **kwargs):
            super().__init__(**kwargs)
            self.degree = degree
            self.a      = coeffs["a"]
            self.b      = coeffs["b"]
            self.t      = t
            self.lmbda  = lmbda
            self.G      = G
        
        def eval(self, values, x):
            values[0] = 0.0
            values[1] = 0.0
            
        def value_shape(self):
            return (2,)


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

class non_lin_2D:
    '''
    FEniCS Expressions for nonlinear in space and linear in time 2D solution data 
    '''
    
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    class u(UserExpression):
        '''
        u - nonlinear in space, linear in time, 2D
        '''
        def __init__(self, coeffs, degree, t, **kwargs):
            super().__init__(**kwargs)
            self.degree = degree
            self.a      = coeffs["a"]
            self.b      = coeffs["b"]
            self.t      = t
        
        def eval(self, values, x):
            values[0] = sin(x[0]+x[1])*(self.a*self.t+1.0)
            values[1] = cos(x[0]*x[1])*(self.b*self.t-1.0)
              
        def value_shape(self):
            return (2,)
        
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    class epsx(UserExpression):
        '''
        strain, for u - nonlinear in space, linear in time, 2D
        '''
        def __init__(self, coeffs, degree, t, **kwargs):
            super().__init__(**kwargs)
            self.degree = degree
            self.a      = coeffs["a"]
            self.b      = coeffs["b"]
            self.t      = t
        
        def eval(self, values, x):
            a=self.a; b=self.b; t=self.t
            values[0] = cos(x[0]+x[1])*(a*t+1.0)
            values[1] = cos(x[0]+x[1])*(a*t+1.0)/2.0+sin(x[0]*x[1])*(1.0-b*t)*x[1]/2.0
            values[2] = cos(x[0]+x[1])*(a*t+1.0)/2.0+sin(x[0]*x[1])*(1.0-b*t)*x[1]/2.0
            values[3] = sin(x[0]*x[1])*(1.0-b*t)*x[0]

        def value_shape(self):
            return (2,2)
        
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    class sigx(UserExpression):
        '''
        stress, for u - nonlinear in space, linear in time, 2D
        '''
        def __init__(self, coeffs, degree, t, lmbda, G, **kwargs):
            super().__init__(**kwargs)
            self.degree = degree
            self.a      = coeffs["a"]
            self.b      = coeffs["b"]
            self.t      = t
            self.lmbda  = lmbda
            self.G      = G
        
        def eval(self, values, x):
            a=self.a; b=self.b; t=self.t; lmbda = self.lmbda; G=self.G
            values[0]  = lmbda*(cos(x[0]+x[1])*(a*t+1.0)+sin(x[0]*x[1])*(1.0-b*t)*x[0])
            values[0] += G*cos(x[0]+x[1])*(a*t+1.0)*2.0
            values[1]  = G*(cos(x[0]+x[1])*(a*t+1.0)+sin(x[0]*x[1])*(1.0-b*t)*x[1])
            values[2]  = G*(cos(x[0]+x[1])*(a*t+1.0)+sin(x[0]*x[1])*(1.0-b*t)*x[1])
            values[3]  = lmbda*(cos(x[0]+x[1])*(a*t+1.0)+sin(x[0]*x[1])*(1.0-b*t)*x[0])
            values[3] -= G*sin(x[0]*x[1])*(b*t-1.0)*x[0]*2.0

        def value_shape(self):
            return (2,2)

    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    class f(UserExpression):
        '''
        body force, for u - nonlinear in space, linear in time, 2D
        '''
        def __init__(self, coeffs, degree, t, lmbda, G, **kwargs):
            super().__init__(**kwargs)
            self.degree = degree
            self.a      = coeffs["a"]
            self.b      = coeffs["b"]
            self.t      = t
            self.lmbda  = lmbda
            self.G      = G
        
        def eval(self, values, x):
            a=self.a; b=self.b; t=self.t; lmbda = self.lmbda; G=self.G
            values[0] = G*sin(x[0]+x[1])*(a*t+1.0)*2.0+G*((sin(x[0]+x[1])*(a*t+1.0))             \
                            +(sin(x[0]*x[1])*(b*t-1.0))+(x[0]*x[1]*cos(x[0]*x[1])*(b*t-1.0)))    \
                            +lmbda*(sin(x[0]+x[1])*(a*t+1.0)+sin(x[0]*x[1])*(b*t-1.0)            \
                            +x[0]*x[1]*cos(x[0]*x[1])*(b*t-1.0))
            values[1] = lmbda*(sin(x[0]+x[1])*(a*t+1.0)+x[0]*x[0]*cos(x[0]*x[1])*(b*t-1.0))      \
                            +2.0*G*((sin(x[0]+x[1])*(a*t+1.0))/2.0                               \
                            +(cos(x[0]*x[1])*(b*t-1.0)*(x[1]*x[1]))/2.0)                         \
                            +G*x[0]*x[0]*cos(x[0]*x[1])*(b*t-1.0)*2.0

        def value_shape(self):
            return (2,)

    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    class g(UserExpression):
        '''
        surface traction, not used for exact solutions (use \sigma\dot n instead)
        '''
        def __init__(self, coeffs, degree, t, lmbda, G, **kwargs):
            super().__init__(**kwargs)
            self.degree = degree
            self.a      = coeffs["a"]
            self.b      = coeffs["b"]
            self.t      = t
            self.lmbda  = lmbda
            self.G      = G
        
        def eval(self, values, x):
            values[0] = 0.0
            values[1] = 0.0
            
        def value_shape(self):
            return (2,)


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

class lin_lin_3D:
    '''
    FEniCS Expressions for linear in space and linear in time 3D solution data 
    '''
    
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    class u(UserExpression):
        '''
        u - linear in space, linear in time, 3D
        '''
        def __init__(self, coeffs, degree, t, **kwargs):
            super().__init__(**kwargs)
            self.degree = degree
            self.a      = coeffs["a"]
            self.b      = coeffs["b"]
            self.c      = coeffs["c"]
            self.t      = t
        
        def eval(self, values, x):
            a=self.a; b=self.b; c=self.c; t=self.t;
            values[0] = (a*t+1.0)*(x[0]+x[1]+x[2])
            values[1] = (b*t-1.0)*(x[0]-x[1]+x[2])
            values[2] = (c*t+2.0)*(x[0]+x[1]-x[2])

        def value_shape(self):
            return (3,)
        
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    class epsx(UserExpression):
        '''
        strain, for u - linear in space, linear in time, 3D
        '''
        def __init__(self, coeffs, degree, t, **kwargs):
            super().__init__(**kwargs)
            self.degree = degree
            self.a      = coeffs["a"]
            self.b      = coeffs["b"]
            self.c      = coeffs["c"]
            self.t      = t
        
        def eval(self, values, x):
            a=self.a; b=self.b; c=self.c; t=self.t
            values[0] = a*t+1.0
            values[1] = t*(a+b)/2.0
            values[2] = a*t/2.0+c*t/2.0+3.0/2.0
            values[3] = t*(a+b)/2.0
            values[4] = -b*t+1.0
            values[5] = b*t/2.0+c*t/2.0+1.0/2.0
            values[6] = a*t/2.0+c*t/2.0+3.0/2.0
            values[7] = b*t/2.0+c*t/2.0+1.0/2.0
            values[8] = -c*t-2.0

        def value_shape(self):
            return (3,3)
        
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    class sigx(UserExpression):
        '''
        stress, for u - linear in space, linear in time, 3D
        '''
        def __init__(self, coeffs, degree, t, lmbda, G, **kwargs):
            super().__init__(**kwargs)
            self.degree = degree
            self.a      = coeffs["a"]
            self.b      = coeffs["b"]
            self.c      = coeffs["c"]
            self.t      = t
            self.lmbda  = lmbda
            self.G      = G
        
        def eval(self, values, x):
            a=self.a; b=self.b; c=self.c; t=self.t; lmbda = self.lmbda; G=self.G
            values[0]  = G*(a*t+1.0)*2.0-lmbda*(b*t+c*t-a*t)
            values[1]  = G*t*(a+b)
            values[2]  = G*(a*t+c*t+3.0)
            values[3]  = G*t*(a+b)
            values[4]  = G*(1.0-b*t)*2.0-lmbda*(b*t+c*t-a*t)
            values[5]  = G*(b*t+c*t+1.0)
            values[6]  = G*(a*t+c*t+3.0)
            values[7]  = G*(b*t+c*t+1.0)
            values[8]  = -G*(c*t+2.0)*2.0-lmbda*(b*t+c*t-a*t)

        def value_shape(self):
            return (3,3)

    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    class f(UserExpression):
        '''
        body force, for u - linear in space, linear in time, 3D
        '''
        def __init__(self, coeffs, degree, t, lmbda, G, **kwargs):
            super().__init__(**kwargs)
            self.degree = degree
            self.a      = coeffs["a"]
            self.b      = coeffs["b"]
            self.c      = coeffs["c"]
            self.t      = t
            self.lmbda  = lmbda
            self.G      = G
        
        def eval(self, values, x):
            a=self.a; b=self.b; c=self.c; t=self.t; lmbda = self.lmbda; G=self.G
            values[0] = 0.0
            values[1] = 0.0
            values[2] = 0.0

        def value_shape(self):
            return (3,)

    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    class g(UserExpression):
        '''
        surface traction, not used for exact solutions (use \sigma\dot n instead)
        '''
        def __init__(self, coeffs, degree, t, lmbda, G, **kwargs):
            super().__init__(**kwargs)
            self.degree = degree
            self.a      = coeffs["a"]
            self.b      = coeffs["b"]
            self.c      = coeffs["c"]
            self.t      = t
            self.lmbda  = lmbda
            self.G      = G
        
        def eval(self, values, x):
            values[0] = 0.0
            values[1] = 0.0
            values[3] = 0.0
            
        def value_shape(self):
            return (3,)

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

class non_lin_3D:
    '''
    FEniCS Expressions for nonlinear in space and linear in time 3D solution data 
    '''
    
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    class u(UserExpression):
        '''
        u - nonlinear in space, linear in time, 3D
        '''
        def __init__(self, coeffs, degree, t, **kwargs):
            super().__init__(**kwargs)
            self.degree = degree
            self.a      = coeffs["a"]
            self.b      = coeffs["b"]
            self.c      = coeffs["c"]
            self.t      = t
        
        def eval(self, values, x):
            a=self.a; b=self.b; c=self.c; t=self.t;
            values[0] = sin(x[0]+x[1]+x[2])*(a*t+1.0)
            values[1] = cos(x[0]*x[1]*x[2])*(b*t-1.0)
            values[2] = (c*t-1.0)*(x[0]*x[0]*x[0]*x[0]+x[1]*x[1]*x[1]-x[2]*x[2])
            
        def value_shape(self):
            return (3,)
        
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    class epsx(UserExpression):
        '''
        strain, for u - nonlinear in space, linear in time, 3D
        '''
        def __init__(self, coeffs, degree, t, **kwargs):
            super().__init__(**kwargs)
            self.degree = degree
            self.a      = coeffs["a"]
            self.b      = coeffs["b"]
            self.c      = coeffs["c"]
            self.t      = t
        
        def eval(self, values, x):
            a=self.a; b=self.b; c=self.c; t=self.t
            values[0] = cos(x[0]+x[1]+x[2])*(a*t+1.0)
            values[1] = (cos(x[0]+x[1]+x[2])*(a*t+1.0))/2.0+(sin(x[0]*x[1]*x[2])*(1.0-b*t)*x[1]*x[2])/2.0
            values[2] = x[0]*x[0]*x[0]*(c*t-1.0)*2.0+(cos(x[0]+x[1]+x[2])*(a*t+1.0))/2.0
            values[3] = (cos(x[0]+x[1]+x[2])*(a*t+1.0))/2.0+(sin(x[0]*x[1]*x[2])*(1.0-b*t)*x[1]*x[2])/2.0
            values[4] = sin(x[0]*x[1]*x[2])*(1.0-b*t)*x[0]*x[2]
            values[5] = x[1]*x[1]*(c*t-1.0)*(3.0/2.0)+(sin(x[0]*x[1]*x[2])*(1.0-b*t)*x[0]*x[1])/2.0
            values[6] = x[0]*x[0]*x[0]*(c*t-1.0)*2.0+(cos(x[0]+x[1]+x[2])*(a*t+1.0))/2.0
            values[7] = x[1]*x[1]*(c*t-1.0)*(3.0/2.0)+(sin(x[0]*x[1]*x[2])*(1.0-b*t)*x[0]*x[1])/2.0
            values[8] = (c*t-1.0)*x[2]*-2.0

        def value_shape(self):
            return (3,3)
        
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    class sigx(UserExpression):
        '''
        stress, for u - nonlinear in space, linear in time, 3D
        '''
        def __init__(self, coeffs, degree, t, lmbda, G, **kwargs):
            super().__init__(**kwargs)
            self.degree = degree
            self.a      = coeffs["a"]
            self.b      = coeffs["b"]
            self.c      = coeffs["c"]
            self.t      = t
            self.lmbda  = lmbda
            self.G      = G
        
        def eval(self, values, x):
            t2 = self.a*self.t;
            t3 = self.b*self.t;
            t4 = self.c*self.t;
            t5 = x[0]*x[0]*x[0];
            t6 = x[1]*x[1];
            t7 = x[0]*x[1]*x[2];
            t8 = x[0]+x[1]+x[2];
            t9 = cos(t8);
            t10 = sin(t7);
            t11 = t2+1.0;
            t12 = t3-1.0;
            t13 = t4-1.0;
            t14 = t13*x[2]*2.0;
            t15 = t9*t11;
            t16 = t5*t13*2.0;
            t18 = t6*t13*(3.0/2.0);
            t20 = t10*t12*x[0]*x[2];
            t21 = (t10*t12*x[0]*x[1])/2.0;
            t22 = (t10*t12*x[1]*x[2])/2.0;
            t17 = -t15;
            t19 = t15/2.0;
            t23 = -t21;
            t24 = -t22;
            t25 = t16+t19;
            t27 = t18+t23;
            t28 = t19+t24;
            t31 = t14+t17+t20;
            t26 = self.G*t25*2.0;
            t29 = self.G*t27*2.0;
            t30 = self.G*t28*2.0;
            t32 = self.lmbda*t31;
            t33 = -t32;

            values[0] = t33+self.G*t15*2.0;
            values[1] = t30;
            values[2] = t26;
            values[3] = t30;
            values[4] = t33-self.G*t20*2.0;
            values[5] = t29;
            values[6] = t26;
            values[7] = t29;
            values[8] = t33-self.G*t13*x[2]*4.0;

        def value_shape(self):
            return (3,3)

    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    class f(UserExpression):
        '''
        body force, for u - nonlinear in space, linear in time, 3D
        '''
        def __init__(self, coeffs, degree, t, lmbda, G, **kwargs):
            super().__init__(**kwargs)
            self.degree = degree
            self.a      = coeffs["a"]
            self.b      = coeffs["b"]
            self.c      = coeffs["c"]
            self.t      = t
            self.lmbda  = lmbda
            self.G      = G
        
        def eval(self, values, x):
            t2 = self.a*self.t;
            t3 = self.b*self.t;
            t4 = self.c*self.t;
            t5 = x[0]*x[0];
            t6 = x[1]*x[1];
            t7 = x[2]*x[2];
            t8 = x[0]*x[1]*x[2];
            t9 = x[0]+x[1]+x[2];
            t10 = cos(t8);
            t11 = sin(t8);
            t12 = sin(t9);
            t13 = t2+1.0;
            t14 = t3-1.0;
            t15 = t4-1.0;
            t16 = t12*t13;
            t17 = t16/2.0;
            values[0] = self.lmbda*(t16+t11*t14*x[2]+t7*t10*t14*x[0]*x[1])                    \
                         +self.G*t16*3.0+self.G*(t17+(t11*t14*x[2])/2.0                       \
                         +(t7*t10*t14*x[0]*x[1])/2.0)*2.0
            values[1] = self.G*(t17+(t6*t7*t10*t14)/2.0)*2.0                                  \
                         +self.lmbda*(t16+t5*t7*t10*t14)+self.G*t5*t6*t10*t14                 \
                         +self.G*t5*t7*t10*t14*2.0
            values[2] = self.G*t15*4.0+self.G*(t15*x[1]*-3.0+(t11*t14*x[0])/2.0               \
                         +(t5*t10*t14*x[1]*x[2])/2.0)*2.0+self.G*(t17-t5*t15*6.0)*2.0         \
                         +self.lmbda*(t15*2.0+t16+t11*t14*x[0]+t5*t10*t14*x[1]*x[2]);

        def value_shape(self):
            return (3,)

    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    class g(UserExpression):
        '''
        surface traction, not used for exact solutions (use \sigma\dot n instead)
        '''
        def __init__(self, coeffs, degree, t, lmbda, G, **kwargs):
            super().__init__(**kwargs)
            self.degree = degree
            self.a      = coeffs["a"]
            self.b      = coeffs["b"]
            self.c      = coeffs["c"]
            self.t      = t
            self.lmbda  = lmbda
            self.G      = G
        
        def eval(self, values, x):
            values[0] = 0.0
            values[1] = 0.0
            values[3] = 0.0
            
        def value_shape(self):
            return (3,)


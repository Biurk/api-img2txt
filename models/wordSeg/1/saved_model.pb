؈$
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�"serve*2.2.02v2.2.0-0-g2b96f3662b8�
}
conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv1/kernel
v
 conv1/kernel/Read/ReadVariableOpReadVariableOpconv1/kernel*'
_output_shapes
:�*
dtype0
m

conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name
conv1/bias
f
conv1/bias/Read/ReadVariableOpReadVariableOp
conv1/bias*
_output_shapes	
:�*
dtype0
�
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�**
shared_namebatch_normalization/gamma
�
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes	
:�*
dtype0
�
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*)
shared_namebatch_normalization/beta
�
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes	
:�*
dtype0
�
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*0
shared_name!batch_normalization/moving_mean
�
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes	
:�*
dtype0
�
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#batch_normalization/moving_variance
�
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes	
:�*
dtype0
~
conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*
shared_nameconv2/kernel
w
 conv2/kernel/Read/ReadVariableOpReadVariableOpconv2/kernel*(
_output_shapes
:��*
dtype0
m

conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name
conv2/bias
f
conv2/bias/Read/ReadVariableOpReadVariableOp
conv2/bias*
_output_shapes	
:�*
dtype0
�
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_1/gamma
�
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes	
:�*
dtype0
�
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_namebatch_normalization_1/beta
�
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes	
:�*
dtype0
�
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!batch_normalization_1/moving_mean
�
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes	
:�*
dtype0
�
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*6
shared_name'%batch_normalization_1/moving_variance
�
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes	
:�*
dtype0
~
conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*
shared_nameconv3/kernel
w
 conv3/kernel/Read/ReadVariableOpReadVariableOpconv3/kernel*(
_output_shapes
:��*
dtype0
m

conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name
conv3/bias
f
conv3/bias/Read/ReadVariableOpReadVariableOp
conv3/bias*
_output_shapes	
:�*
dtype0
�
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_2/gamma
�
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes	
:�*
dtype0
�
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_namebatch_normalization_2/beta
�
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes	
:�*
dtype0
�
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!batch_normalization_2/moving_mean
�
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes	
:�*
dtype0
�
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*6
shared_name'%batch_normalization_2/moving_variance
�
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes	
:�*
dtype0
~
conv4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*
shared_nameconv4/kernel
w
 conv4/kernel/Read/ReadVariableOpReadVariableOpconv4/kernel*(
_output_shapes
:��*
dtype0
m

conv4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name
conv4/bias
f
conv4/bias/Read/ReadVariableOpReadVariableOp
conv4/bias*
_output_shapes	
:�*
dtype0
�
batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_3/gamma
�
/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes	
:�*
dtype0
�
batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_namebatch_normalization_3/beta
�
.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes	
:�*
dtype0
�
!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!batch_normalization_3/moving_mean
�
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes	
:�*
dtype0
�
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*6
shared_name'%batch_normalization_3/moving_variance
�
9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes	
:�*
dtype0
~
conv5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*
shared_nameconv5/kernel
w
 conv5/kernel/Read/ReadVariableOpReadVariableOpconv5/kernel*(
_output_shapes
:��*
dtype0
m

conv5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name
conv5/bias
f
conv5/bias/Read/ReadVariableOpReadVariableOp
conv5/bias*
_output_shapes	
:�*
dtype0
�
batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_4/gamma
�
/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
_output_shapes	
:�*
dtype0
�
batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_namebatch_normalization_4/beta
�
.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
_output_shapes	
:�*
dtype0
�
!batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!batch_normalization_4/moving_mean
�
5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
_output_shapes	
:�*
dtype0
�
%batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*6
shared_name'%batch_normalization_4/moving_variance
�
9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
_output_shapes	
:�*
dtype0
w
dense1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*
shared_namedense1/kernel
p
!dense1/kernel/Read/ReadVariableOpReadVariableOpdense1/kernel*
_output_shapes
:	�@*
dtype0
n
dense1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense1/bias
g
dense1/bias/Read/ReadVariableOpReadVariableOpdense1/bias*
_output_shapes
:@*
dtype0
v
dense2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense2/kernel
o
!dense2/kernel/Read/ReadVariableOpReadVariableOpdense2/kernel*
_output_shapes

:@*
dtype0
n
dense2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense2/bias
g
dense2/bias/Read/ReadVariableOpReadVariableOpdense2/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
�
Adam/conv1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameAdam/conv1/kernel/m
�
'Adam/conv1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1/kernel/m*'
_output_shapes
:�*
dtype0
{
Adam/conv1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*"
shared_nameAdam/conv1/bias/m
t
%Adam/conv1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1/bias/m*
_output_shapes	
:�*
dtype0
�
 Adam/batch_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" Adam/batch_normalization/gamma/m
�
4Adam/batch_normalization/gamma/m/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/m*
_output_shapes	
:�*
dtype0
�
Adam/batch_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*0
shared_name!Adam/batch_normalization/beta/m
�
3Adam/batch_normalization/beta/m/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/m*
_output_shapes	
:�*
dtype0
�
Adam/conv2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*$
shared_nameAdam/conv2/kernel/m
�
'Adam/conv2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2/kernel/m*(
_output_shapes
:��*
dtype0
{
Adam/conv2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*"
shared_nameAdam/conv2/bias/m
t
%Adam/conv2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2/bias/m*
_output_shapes	
:�*
dtype0
�
"Adam/batch_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Adam/batch_normalization_1/gamma/m
�
6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/m*
_output_shapes	
:�*
dtype0
�
!Adam/batch_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!Adam/batch_normalization_1/beta/m
�
5Adam/batch_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/m*
_output_shapes	
:�*
dtype0
�
Adam/conv3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*$
shared_nameAdam/conv3/kernel/m
�
'Adam/conv3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3/kernel/m*(
_output_shapes
:��*
dtype0
{
Adam/conv3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*"
shared_nameAdam/conv3/bias/m
t
%Adam/conv3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3/bias/m*
_output_shapes	
:�*
dtype0
�
"Adam/batch_normalization_2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Adam/batch_normalization_2/gamma/m
�
6Adam/batch_normalization_2/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_2/gamma/m*
_output_shapes	
:�*
dtype0
�
!Adam/batch_normalization_2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!Adam/batch_normalization_2/beta/m
�
5Adam/batch_normalization_2/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_2/beta/m*
_output_shapes	
:�*
dtype0
�
Adam/conv4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*$
shared_nameAdam/conv4/kernel/m
�
'Adam/conv4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv4/kernel/m*(
_output_shapes
:��*
dtype0
{
Adam/conv4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*"
shared_nameAdam/conv4/bias/m
t
%Adam/conv4/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv4/bias/m*
_output_shapes	
:�*
dtype0
�
"Adam/batch_normalization_3/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Adam/batch_normalization_3/gamma/m
�
6Adam/batch_normalization_3/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_3/gamma/m*
_output_shapes	
:�*
dtype0
�
!Adam/batch_normalization_3/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!Adam/batch_normalization_3/beta/m
�
5Adam/batch_normalization_3/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_3/beta/m*
_output_shapes	
:�*
dtype0
�
Adam/conv5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*$
shared_nameAdam/conv5/kernel/m
�
'Adam/conv5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv5/kernel/m*(
_output_shapes
:��*
dtype0
{
Adam/conv5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*"
shared_nameAdam/conv5/bias/m
t
%Adam/conv5/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv5/bias/m*
_output_shapes	
:�*
dtype0
�
"Adam/batch_normalization_4/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Adam/batch_normalization_4/gamma/m
�
6Adam/batch_normalization_4/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_4/gamma/m*
_output_shapes	
:�*
dtype0
�
!Adam/batch_normalization_4/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!Adam/batch_normalization_4/beta/m
�
5Adam/batch_normalization_4/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_4/beta/m*
_output_shapes	
:�*
dtype0
�
Adam/dense1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*%
shared_nameAdam/dense1/kernel/m
~
(Adam/dense1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense1/kernel/m*
_output_shapes
:	�@*
dtype0
|
Adam/dense1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/dense1/bias/m
u
&Adam/dense1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense1/bias/m*
_output_shapes
:@*
dtype0
�
Adam/dense2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*%
shared_nameAdam/dense2/kernel/m
}
(Adam/dense2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense2/kernel/m*
_output_shapes

:@*
dtype0
|
Adam/dense2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/dense2/bias/m
u
&Adam/dense2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense2/bias/m*
_output_shapes
:*
dtype0
�
Adam/conv1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameAdam/conv1/kernel/v
�
'Adam/conv1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1/kernel/v*'
_output_shapes
:�*
dtype0
{
Adam/conv1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*"
shared_nameAdam/conv1/bias/v
t
%Adam/conv1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1/bias/v*
_output_shapes	
:�*
dtype0
�
 Adam/batch_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" Adam/batch_normalization/gamma/v
�
4Adam/batch_normalization/gamma/v/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/v*
_output_shapes	
:�*
dtype0
�
Adam/batch_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*0
shared_name!Adam/batch_normalization/beta/v
�
3Adam/batch_normalization/beta/v/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/v*
_output_shapes	
:�*
dtype0
�
Adam/conv2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*$
shared_nameAdam/conv2/kernel/v
�
'Adam/conv2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2/kernel/v*(
_output_shapes
:��*
dtype0
{
Adam/conv2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*"
shared_nameAdam/conv2/bias/v
t
%Adam/conv2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2/bias/v*
_output_shapes	
:�*
dtype0
�
"Adam/batch_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Adam/batch_normalization_1/gamma/v
�
6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/v*
_output_shapes	
:�*
dtype0
�
!Adam/batch_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!Adam/batch_normalization_1/beta/v
�
5Adam/batch_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/v*
_output_shapes	
:�*
dtype0
�
Adam/conv3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*$
shared_nameAdam/conv3/kernel/v
�
'Adam/conv3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3/kernel/v*(
_output_shapes
:��*
dtype0
{
Adam/conv3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*"
shared_nameAdam/conv3/bias/v
t
%Adam/conv3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3/bias/v*
_output_shapes	
:�*
dtype0
�
"Adam/batch_normalization_2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Adam/batch_normalization_2/gamma/v
�
6Adam/batch_normalization_2/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_2/gamma/v*
_output_shapes	
:�*
dtype0
�
!Adam/batch_normalization_2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!Adam/batch_normalization_2/beta/v
�
5Adam/batch_normalization_2/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_2/beta/v*
_output_shapes	
:�*
dtype0
�
Adam/conv4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*$
shared_nameAdam/conv4/kernel/v
�
'Adam/conv4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv4/kernel/v*(
_output_shapes
:��*
dtype0
{
Adam/conv4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*"
shared_nameAdam/conv4/bias/v
t
%Adam/conv4/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv4/bias/v*
_output_shapes	
:�*
dtype0
�
"Adam/batch_normalization_3/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Adam/batch_normalization_3/gamma/v
�
6Adam/batch_normalization_3/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_3/gamma/v*
_output_shapes	
:�*
dtype0
�
!Adam/batch_normalization_3/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!Adam/batch_normalization_3/beta/v
�
5Adam/batch_normalization_3/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_3/beta/v*
_output_shapes	
:�*
dtype0
�
Adam/conv5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*$
shared_nameAdam/conv5/kernel/v
�
'Adam/conv5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv5/kernel/v*(
_output_shapes
:��*
dtype0
{
Adam/conv5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*"
shared_nameAdam/conv5/bias/v
t
%Adam/conv5/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv5/bias/v*
_output_shapes	
:�*
dtype0
�
"Adam/batch_normalization_4/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Adam/batch_normalization_4/gamma/v
�
6Adam/batch_normalization_4/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_4/gamma/v*
_output_shapes	
:�*
dtype0
�
!Adam/batch_normalization_4/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!Adam/batch_normalization_4/beta/v
�
5Adam/batch_normalization_4/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_4/beta/v*
_output_shapes	
:�*
dtype0
�
Adam/dense1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*%
shared_nameAdam/dense1/kernel/v
~
(Adam/dense1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense1/kernel/v*
_output_shapes
:	�@*
dtype0
|
Adam/dense1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/dense1/bias/v
u
&Adam/dense1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense1/bias/v*
_output_shapes
:@*
dtype0
�
Adam/dense2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*%
shared_nameAdam/dense2/kernel/v
}
(Adam/dense2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense2/kernel/v*
_output_shapes

:@*
dtype0
|
Adam/dense2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/dense2/bias/v
u
&Adam/dense2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense2/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value��B�� B��
�
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
layer-10
layer_with_weights-7
layer-11
layer_with_weights-8
layer-12
layer-13
layer_with_weights-9
layer-14
layer-15
layer_with_weights-10
layer-16
layer-17
layer_with_weights-11
layer-18
layer-19
layer-20
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
h

kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
R
"	variables
#trainable_variables
$regularization_losses
%	keras_api
�
&axis
	'gamma
(beta
)moving_mean
*moving_variance
+	variables
,trainable_variables
-regularization_losses
.	keras_api
h

/kernel
0bias
1	variables
2trainable_variables
3regularization_losses
4	keras_api
R
5	variables
6trainable_variables
7regularization_losses
8	keras_api
�
9axis
	:gamma
;beta
<moving_mean
=moving_variance
>	variables
?trainable_variables
@regularization_losses
A	keras_api
h

Bkernel
Cbias
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
R
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
�
Laxis
	Mgamma
Nbeta
Omoving_mean
Pmoving_variance
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
h

Ukernel
Vbias
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
R
[	variables
\trainable_variables
]regularization_losses
^	keras_api
�
_axis
	`gamma
abeta
bmoving_mean
cmoving_variance
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h

hkernel
ibias
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
R
n	variables
otrainable_variables
pregularization_losses
q	keras_api
�
raxis
	sgamma
tbeta
umoving_mean
vmoving_variance
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
R
{	variables
|trainable_variables
}regularization_losses
~	keras_api
m

kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
V
�	variables
�trainable_variables
�regularization_losses
�	keras_api
n
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
V
�	variables
�trainable_variables
�regularization_losses
�	keras_api
V
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�
	�iter
�beta_1
�beta_2

�decay
�learning_ratem�m�'m�(m�/m�0m�:m�;m�Bm�Cm�Mm�Nm�Um�Vm�`m�am�hm�im�sm�tm�m�	�m�	�m�	�m�v�v�'v�(v�/v�0v�:v�;v�Bv�Cv�Mv�Nv�Uv�Vv�`v�av�hv�iv�sv�tv�v�	�v�	�v�	�v�
�
0
1
'2
(3
)4
*5
/6
07
:8
;9
<10
=11
B12
C13
M14
N15
O16
P17
U18
V19
`20
a21
b22
c23
h24
i25
s26
t27
u28
v29
30
�31
�32
�33
�
0
1
'2
(3
/4
05
:6
;7
B8
C9
M10
N11
U12
V13
`14
a15
h16
i17
s18
t19
20
�21
�22
�23
 
�
�non_trainable_variables
	variables
�layer_metrics
trainable_variables
�metrics
�layers
regularization_losses
 �layer_regularization_losses
 
XV
VARIABLE_VALUEconv1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
�non_trainable_variables
	variables
�layer_metrics
trainable_variables
�metrics
�layers
 regularization_losses
 �layer_regularization_losses
 
 
 
�
�non_trainable_variables
"	variables
�layer_metrics
#trainable_variables
�metrics
�layers
$regularization_losses
 �layer_regularization_losses
 
db
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

'0
(1
)2
*3

'0
(1
 
�
�non_trainable_variables
+	variables
�layer_metrics
,trainable_variables
�metrics
�layers
-regularization_losses
 �layer_regularization_losses
XV
VARIABLE_VALUEconv2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

/0
01

/0
01
 
�
�non_trainable_variables
1	variables
�layer_metrics
2trainable_variables
�metrics
�layers
3regularization_losses
 �layer_regularization_losses
 
 
 
�
�non_trainable_variables
5	variables
�layer_metrics
6trainable_variables
�metrics
�layers
7regularization_losses
 �layer_regularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

:0
;1
<2
=3

:0
;1
 
�
�non_trainable_variables
>	variables
�layer_metrics
?trainable_variables
�metrics
�layers
@regularization_losses
 �layer_regularization_losses
XV
VARIABLE_VALUEconv3/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv3/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

B0
C1

B0
C1
 
�
�non_trainable_variables
D	variables
�layer_metrics
Etrainable_variables
�metrics
�layers
Fregularization_losses
 �layer_regularization_losses
 
 
 
�
�non_trainable_variables
H	variables
�layer_metrics
Itrainable_variables
�metrics
�layers
Jregularization_losses
 �layer_regularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_2/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_2/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_2/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_2/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

M0
N1
O2
P3

M0
N1
 
�
�non_trainable_variables
Q	variables
�layer_metrics
Rtrainable_variables
�metrics
�layers
Sregularization_losses
 �layer_regularization_losses
XV
VARIABLE_VALUEconv4/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv4/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

U0
V1

U0
V1
 
�
�non_trainable_variables
W	variables
�layer_metrics
Xtrainable_variables
�metrics
�layers
Yregularization_losses
 �layer_regularization_losses
 
 
 
�
�non_trainable_variables
[	variables
�layer_metrics
\trainable_variables
�metrics
�layers
]regularization_losses
 �layer_regularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_3/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_3/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_3/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_3/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

`0
a1
b2
c3

`0
a1
 
�
�non_trainable_variables
d	variables
�layer_metrics
etrainable_variables
�metrics
�layers
fregularization_losses
 �layer_regularization_losses
XV
VARIABLE_VALUEconv5/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv5/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

h0
i1

h0
i1
 
�
�non_trainable_variables
j	variables
�layer_metrics
ktrainable_variables
�metrics
�layers
lregularization_losses
 �layer_regularization_losses
 
 
 
�
�non_trainable_variables
n	variables
�layer_metrics
otrainable_variables
�metrics
�layers
pregularization_losses
 �layer_regularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_4/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_4/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_4/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_4/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

s0
t1
u2
v3

s0
t1
 
�
�non_trainable_variables
w	variables
�layer_metrics
xtrainable_variables
�metrics
�layers
yregularization_losses
 �layer_regularization_losses
 
 
 
�
�non_trainable_variables
{	variables
�layer_metrics
|trainable_variables
�metrics
�layers
}regularization_losses
 �layer_regularization_losses
ZX
VARIABLE_VALUEdense1/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense1/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

0
�1

0
�1
 
�
�non_trainable_variables
�	variables
�layer_metrics
�trainable_variables
�metrics
�layers
�regularization_losses
 �layer_regularization_losses
 
 
 
�
�non_trainable_variables
�	variables
�layer_metrics
�trainable_variables
�metrics
�layers
�regularization_losses
 �layer_regularization_losses
ZX
VARIABLE_VALUEdense2/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense2/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�0
�1
 
�
�non_trainable_variables
�	variables
�layer_metrics
�trainable_variables
�metrics
�layers
�regularization_losses
 �layer_regularization_losses
 
 
 
�
�non_trainable_variables
�	variables
�layer_metrics
�trainable_variables
�metrics
�layers
�regularization_losses
 �layer_regularization_losses
 
 
 
�
�non_trainable_variables
�	variables
�layer_metrics
�trainable_variables
�metrics
�layers
�regularization_losses
 �layer_regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
F
)0
*1
<2
=3
O4
P5
b6
c7
u8
v9
 

�0
�1
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
 
 
 
 
 
 
 
 
 
 
 

)0
*1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

<0
=1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

O0
P1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

b0
c1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

u0
v1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

�total

�count
�	variables
�	keras_api
I

�total

�count
�
_fn_kwargs
�	variables
�	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�	variables
{y
VARIABLE_VALUEAdam/conv1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE Adam/batch_normalization/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/batch_normalization/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_1/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE!Adam/batch_normalization_1/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv3/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv3/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_2/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE!Adam/batch_normalization_2/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv4/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv4/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_3/gamma/mQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE!Adam/batch_normalization_3/beta/mPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv5/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv5/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_4/gamma/mQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE!Adam/batch_normalization_4/beta/mPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense1/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense1/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense2/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense2/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE Adam/batch_normalization/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/batch_normalization/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_1/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE!Adam/batch_normalization_1/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv3/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv3/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_2/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE!Adam/batch_normalization_2/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv4/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv4/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_3/gamma/vQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE!Adam/batch_normalization_3/beta/vPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv5/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv5/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_4/gamma/vQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE!Adam/batch_normalization_4/beta/vPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense1/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense1/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense2/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense2/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_the_inputsPlaceholder*0
_output_shapes
:����������#*
dtype0*%
shape:����������#
�	
StatefulPartitionedCallStatefulPartitionedCallserving_default_the_inputsconv1/kernel
conv1/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2/kernel
conv2/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv3/kernel
conv3/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv4/kernel
conv4/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv5/kernel
conv5/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variancedense1/kerneldense1/biasdense2/kerneldense2/bias*.
Tin'
%2#*
Tout
2*,
_output_shapes
:����������*D
_read_only_resource_inputs&
$"	
 !"*-
config_proto

CPU

GPU2*0J 8*+
f&R$
"__inference_signature_wrapper_7290
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�"
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename conv1/kernel/Read/ReadVariableOpconv1/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp conv2/kernel/Read/ReadVariableOpconv2/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp conv3/kernel/Read/ReadVariableOpconv3/bias/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp conv4/kernel/Read/ReadVariableOpconv4/bias/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOp conv5/kernel/Read/ReadVariableOpconv5/bias/Read/ReadVariableOp/batch_normalization_4/gamma/Read/ReadVariableOp.batch_normalization_4/beta/Read/ReadVariableOp5batch_normalization_4/moving_mean/Read/ReadVariableOp9batch_normalization_4/moving_variance/Read/ReadVariableOp!dense1/kernel/Read/ReadVariableOpdense1/bias/Read/ReadVariableOp!dense2/kernel/Read/ReadVariableOpdense2/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp'Adam/conv1/kernel/m/Read/ReadVariableOp%Adam/conv1/bias/m/Read/ReadVariableOp4Adam/batch_normalization/gamma/m/Read/ReadVariableOp3Adam/batch_normalization/beta/m/Read/ReadVariableOp'Adam/conv2/kernel/m/Read/ReadVariableOp%Adam/conv2/bias/m/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_1/beta/m/Read/ReadVariableOp'Adam/conv3/kernel/m/Read/ReadVariableOp%Adam/conv3/bias/m/Read/ReadVariableOp6Adam/batch_normalization_2/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_2/beta/m/Read/ReadVariableOp'Adam/conv4/kernel/m/Read/ReadVariableOp%Adam/conv4/bias/m/Read/ReadVariableOp6Adam/batch_normalization_3/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_3/beta/m/Read/ReadVariableOp'Adam/conv5/kernel/m/Read/ReadVariableOp%Adam/conv5/bias/m/Read/ReadVariableOp6Adam/batch_normalization_4/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_4/beta/m/Read/ReadVariableOp(Adam/dense1/kernel/m/Read/ReadVariableOp&Adam/dense1/bias/m/Read/ReadVariableOp(Adam/dense2/kernel/m/Read/ReadVariableOp&Adam/dense2/bias/m/Read/ReadVariableOp'Adam/conv1/kernel/v/Read/ReadVariableOp%Adam/conv1/bias/v/Read/ReadVariableOp4Adam/batch_normalization/gamma/v/Read/ReadVariableOp3Adam/batch_normalization/beta/v/Read/ReadVariableOp'Adam/conv2/kernel/v/Read/ReadVariableOp%Adam/conv2/bias/v/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_1/beta/v/Read/ReadVariableOp'Adam/conv3/kernel/v/Read/ReadVariableOp%Adam/conv3/bias/v/Read/ReadVariableOp6Adam/batch_normalization_2/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_2/beta/v/Read/ReadVariableOp'Adam/conv4/kernel/v/Read/ReadVariableOp%Adam/conv4/bias/v/Read/ReadVariableOp6Adam/batch_normalization_3/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_3/beta/v/Read/ReadVariableOp'Adam/conv5/kernel/v/Read/ReadVariableOp%Adam/conv5/bias/v/Read/ReadVariableOp6Adam/batch_normalization_4/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_4/beta/v/Read/ReadVariableOp(Adam/dense1/kernel/v/Read/ReadVariableOp&Adam/dense1/bias/v/Read/ReadVariableOp(Adam/dense2/kernel/v/Read/ReadVariableOp&Adam/dense2/bias/v/Read/ReadVariableOpConst*h
Tina
_2]	*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*&
f!R
__inference__traced_save_9144
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1/kernel
conv1/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2/kernel
conv2/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv3/kernel
conv3/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv4/kernel
conv4/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv5/kernel
conv5/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variancedense1/kerneldense1/biasdense2/kerneldense2/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv1/kernel/mAdam/conv1/bias/m Adam/batch_normalization/gamma/mAdam/batch_normalization/beta/mAdam/conv2/kernel/mAdam/conv2/bias/m"Adam/batch_normalization_1/gamma/m!Adam/batch_normalization_1/beta/mAdam/conv3/kernel/mAdam/conv3/bias/m"Adam/batch_normalization_2/gamma/m!Adam/batch_normalization_2/beta/mAdam/conv4/kernel/mAdam/conv4/bias/m"Adam/batch_normalization_3/gamma/m!Adam/batch_normalization_3/beta/mAdam/conv5/kernel/mAdam/conv5/bias/m"Adam/batch_normalization_4/gamma/m!Adam/batch_normalization_4/beta/mAdam/dense1/kernel/mAdam/dense1/bias/mAdam/dense2/kernel/mAdam/dense2/bias/mAdam/conv1/kernel/vAdam/conv1/bias/v Adam/batch_normalization/gamma/vAdam/batch_normalization/beta/vAdam/conv2/kernel/vAdam/conv2/bias/v"Adam/batch_normalization_1/gamma/v!Adam/batch_normalization_1/beta/vAdam/conv3/kernel/vAdam/conv3/bias/v"Adam/batch_normalization_2/gamma/v!Adam/batch_normalization_2/beta/vAdam/conv4/kernel/vAdam/conv4/bias/v"Adam/batch_normalization_3/gamma/v!Adam/batch_normalization_3/beta/vAdam/conv5/kernel/vAdam/conv5/bias/v"Adam/batch_normalization_4/gamma/v!Adam/batch_normalization_4/beta/vAdam/dense1/kernel/vAdam/dense1/bias/vAdam/dense2/kernel/vAdam/dense2/bias/v*g
Tin`
^2\*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*)
f$R"
 __inference__traced_restore_9429ԡ
�$
�
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_8240

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��#AssignMovingAvg/AssignSubVariableOp�%AssignMovingAvg_1/AssignSubVariableOpu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:�����������:�:�:�:�:*
epsilon%o�:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:�����������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
"__inference_signature_wrapper_7290

the_inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall
the_inputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32*.
Tin'
%2#*
Tout
2*,
_output_shapes
:����������*D
_read_only_resource_inputs&
$"	
 !"*-
config_proto

CPU

GPU2*0J 8*(
f#R!
__inference__wrapped_model_52792
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:����������#::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
0
_output_shapes
:����������#
$
_user_specified_name
the_inputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: 
�$
�
M__inference_batch_normalization_layer_call_and_return_conditional_losses_7991

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��#AssignMovingAvg/AssignSubVariableOp�%AssignMovingAvg_1/AssignSubVariableOpu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8096

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������:::::j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�$
�
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_5877

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��#AssignMovingAvg/AssignSubVariableOp�%AssignMovingAvg_1/AssignSubVariableOpu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
4__inference_batch_normalization_1_layer_call_fn_8122

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_55882
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
`
A__inference_dropout_layer_call_and_return_conditional_losses_8695

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout/Const}
dropout/MulMulinputsdropout/Const:output:0*
T0*1
_output_shapes
:�����������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*1
_output_shapes
:�����������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:�����������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:�����������2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*1
_output_shapes
:�����������2
dropout/Mul_1o
IdentityIdentitydropout/Mul_1:z:0*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*0
_input_shapes
:�����������:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
>
"__inference_ma3_layer_call_fn_5633

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*F
fAR?
=__inference_ma3_layer_call_and_return_conditional_losses_56272
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
4__inference_batch_normalization_2_layer_call_fn_8271

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_62992
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:�����������::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�$
�
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_6299

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��#AssignMovingAvg/AssignSubVariableOp�%AssignMovingAvg_1/AssignSubVariableOpu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:�����������:�:�:�:�:*
epsilon%o�:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:�����������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
��
�
__inference__wrapped_model_5279

the_inputs3
/sequential_conv1_conv2d_readvariableop_resource4
0sequential_conv1_biasadd_readvariableop_resource:
6sequential_batch_normalization_readvariableop_resource<
8sequential_batch_normalization_readvariableop_1_resourceK
Gsequential_batch_normalization_fusedbatchnormv3_readvariableop_resourceM
Isequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource3
/sequential_conv2_conv2d_readvariableop_resource4
0sequential_conv2_biasadd_readvariableop_resource<
8sequential_batch_normalization_1_readvariableop_resource>
:sequential_batch_normalization_1_readvariableop_1_resourceM
Isequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resourceO
Ksequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource3
/sequential_conv3_conv2d_readvariableop_resource4
0sequential_conv3_biasadd_readvariableop_resource<
8sequential_batch_normalization_2_readvariableop_resource>
:sequential_batch_normalization_2_readvariableop_1_resourceM
Isequential_batch_normalization_2_fusedbatchnormv3_readvariableop_resourceO
Ksequential_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource3
/sequential_conv4_conv2d_readvariableop_resource4
0sequential_conv4_biasadd_readvariableop_resource<
8sequential_batch_normalization_3_readvariableop_resource>
:sequential_batch_normalization_3_readvariableop_1_resourceM
Isequential_batch_normalization_3_fusedbatchnormv3_readvariableop_resourceO
Ksequential_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource3
/sequential_conv5_conv2d_readvariableop_resource4
0sequential_conv5_biasadd_readvariableop_resource<
8sequential_batch_normalization_4_readvariableop_resource>
:sequential_batch_normalization_4_readvariableop_1_resourceM
Isequential_batch_normalization_4_fusedbatchnormv3_readvariableop_resourceO
Ksequential_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7
3sequential_dense1_tensordot_readvariableop_resource5
1sequential_dense1_biasadd_readvariableop_resource7
3sequential_dense2_tensordot_readvariableop_resource5
1sequential_dense2_biasadd_readvariableop_resource
identity��
&sequential/conv1/Conv2D/ReadVariableOpReadVariableOp/sequential_conv1_conv2d_readvariableop_resource*'
_output_shapes
:�*
dtype02(
&sequential/conv1/Conv2D/ReadVariableOp�
sequential/conv1/Conv2DConv2D
the_inputs.sequential/conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������#�*
paddingSAME*
strides
2
sequential/conv1/Conv2D�
'sequential/conv1/BiasAdd/ReadVariableOpReadVariableOp0sequential_conv1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02)
'sequential/conv1/BiasAdd/ReadVariableOp�
sequential/conv1/BiasAddBiasAdd sequential/conv1/Conv2D:output:0/sequential/conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������#�2
sequential/conv1/BiasAdd�
sequential/conv1/ReluRelu!sequential/conv1/BiasAdd:output:0*
T0*1
_output_shapes
:����������#�2
sequential/conv1/Relu�
sequential/max1/MaxPoolMaxPool#sequential/conv1/Relu:activations:0*1
_output_shapes
:�����������*
ksize
*
paddingVALID*
strides
2
sequential/max1/MaxPool�
-sequential/batch_normalization/ReadVariableOpReadVariableOp6sequential_batch_normalization_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential/batch_normalization/ReadVariableOp�
/sequential/batch_normalization/ReadVariableOp_1ReadVariableOp8sequential_batch_normalization_readvariableop_1_resource*
_output_shapes	
:�*
dtype021
/sequential/batch_normalization/ReadVariableOp_1�
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpGsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02@
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp�
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIsequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02B
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1�
/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV3 sequential/max1/MaxPool:output:05sequential/batch_normalization/ReadVariableOp:value:07sequential/batch_normalization/ReadVariableOp_1:value:0Fsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Hsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:�����������:�:�:�:�:*
epsilon%o�:*
is_training( 21
/sequential/batch_normalization/FusedBatchNormV3�
&sequential/conv2/Conv2D/ReadVariableOpReadVariableOp/sequential_conv2_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02(
&sequential/conv2/Conv2D/ReadVariableOp�
sequential/conv2/Conv2DConv2D3sequential/batch_normalization/FusedBatchNormV3:y:0.sequential/conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
2
sequential/conv2/Conv2D�
'sequential/conv2/BiasAdd/ReadVariableOpReadVariableOp0sequential_conv2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02)
'sequential/conv2/BiasAdd/ReadVariableOp�
sequential/conv2/BiasAddBiasAdd sequential/conv2/Conv2D:output:0/sequential/conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2
sequential/conv2/BiasAdd�
sequential/conv2/ReluRelu!sequential/conv2/BiasAdd:output:0*
T0*1
_output_shapes
:�����������2
sequential/conv2/Relu�
sequential/max2/MaxPoolMaxPool#sequential/conv2/Relu:activations:0*1
_output_shapes
:�����������*
ksize
*
paddingVALID*
strides
2
sequential/max2/MaxPool�
/sequential/batch_normalization_1/ReadVariableOpReadVariableOp8sequential_batch_normalization_1_readvariableop_resource*
_output_shapes	
:�*
dtype021
/sequential/batch_normalization_1/ReadVariableOp�
1sequential/batch_normalization_1/ReadVariableOp_1ReadVariableOp:sequential_batch_normalization_1_readvariableop_1_resource*
_output_shapes	
:�*
dtype023
1sequential/batch_normalization_1/ReadVariableOp_1�
@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpIsequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02B
@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp�
Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKsequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02D
Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�
1sequential/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3 sequential/max2/MaxPool:output:07sequential/batch_normalization_1/ReadVariableOp:value:09sequential/batch_normalization_1/ReadVariableOp_1:value:0Hsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Jsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:�����������:�:�:�:�:*
epsilon%o�:*
is_training( 23
1sequential/batch_normalization_1/FusedBatchNormV3�
&sequential/conv3/Conv2D/ReadVariableOpReadVariableOp/sequential_conv3_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02(
&sequential/conv3/Conv2D/ReadVariableOp�
sequential/conv3/Conv2DConv2D5sequential/batch_normalization_1/FusedBatchNormV3:y:0.sequential/conv3/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
2
sequential/conv3/Conv2D�
'sequential/conv3/BiasAdd/ReadVariableOpReadVariableOp0sequential_conv3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02)
'sequential/conv3/BiasAdd/ReadVariableOp�
sequential/conv3/BiasAddBiasAdd sequential/conv3/Conv2D:output:0/sequential/conv3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2
sequential/conv3/BiasAdd�
sequential/conv3/ReluRelu!sequential/conv3/BiasAdd:output:0*
T0*1
_output_shapes
:�����������2
sequential/conv3/Relu�
sequential/ma3/MaxPoolMaxPool#sequential/conv3/Relu:activations:0*1
_output_shapes
:�����������*
ksize
*
paddingVALID*
strides
2
sequential/ma3/MaxPool�
/sequential/batch_normalization_2/ReadVariableOpReadVariableOp8sequential_batch_normalization_2_readvariableop_resource*
_output_shapes	
:�*
dtype021
/sequential/batch_normalization_2/ReadVariableOp�
1sequential/batch_normalization_2/ReadVariableOp_1ReadVariableOp:sequential_batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:�*
dtype023
1sequential/batch_normalization_2/ReadVariableOp_1�
@sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpIsequential_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02B
@sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp�
Bsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKsequential_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02D
Bsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1�
1sequential/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3sequential/ma3/MaxPool:output:07sequential/batch_normalization_2/ReadVariableOp:value:09sequential/batch_normalization_2/ReadVariableOp_1:value:0Hsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Jsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:�����������:�:�:�:�:*
epsilon%o�:*
is_training( 23
1sequential/batch_normalization_2/FusedBatchNormV3�
&sequential/conv4/Conv2D/ReadVariableOpReadVariableOp/sequential_conv4_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02(
&sequential/conv4/Conv2D/ReadVariableOp�
sequential/conv4/Conv2DConv2D5sequential/batch_normalization_2/FusedBatchNormV3:y:0.sequential/conv4/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
2
sequential/conv4/Conv2D�
'sequential/conv4/BiasAdd/ReadVariableOpReadVariableOp0sequential_conv4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02)
'sequential/conv4/BiasAdd/ReadVariableOp�
sequential/conv4/BiasAddBiasAdd sequential/conv4/Conv2D:output:0/sequential/conv4/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2
sequential/conv4/BiasAdd�
sequential/conv4/ReluRelu!sequential/conv4/BiasAdd:output:0*
T0*1
_output_shapes
:�����������2
sequential/conv4/Relu�
sequential/ma4/MaxPoolMaxPool#sequential/conv4/Relu:activations:0*1
_output_shapes
:�����������*
ksize
*
paddingVALID*
strides
2
sequential/ma4/MaxPool�
/sequential/batch_normalization_3/ReadVariableOpReadVariableOp8sequential_batch_normalization_3_readvariableop_resource*
_output_shapes	
:�*
dtype021
/sequential/batch_normalization_3/ReadVariableOp�
1sequential/batch_normalization_3/ReadVariableOp_1ReadVariableOp:sequential_batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:�*
dtype023
1sequential/batch_normalization_3/ReadVariableOp_1�
@sequential/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpIsequential_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02B
@sequential/batch_normalization_3/FusedBatchNormV3/ReadVariableOp�
Bsequential/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKsequential_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02D
Bsequential/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�
1sequential/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3sequential/ma4/MaxPool:output:07sequential/batch_normalization_3/ReadVariableOp:value:09sequential/batch_normalization_3/ReadVariableOp_1:value:0Hsequential/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Jsequential/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:�����������:�:�:�:�:*
epsilon%o�:*
is_training( 23
1sequential/batch_normalization_3/FusedBatchNormV3�
&sequential/conv5/Conv2D/ReadVariableOpReadVariableOp/sequential_conv5_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02(
&sequential/conv5/Conv2D/ReadVariableOp�
sequential/conv5/Conv2DConv2D5sequential/batch_normalization_3/FusedBatchNormV3:y:0.sequential/conv5/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
2
sequential/conv5/Conv2D�
'sequential/conv5/BiasAdd/ReadVariableOpReadVariableOp0sequential_conv5_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02)
'sequential/conv5/BiasAdd/ReadVariableOp�
sequential/conv5/BiasAddBiasAdd sequential/conv5/Conv2D:output:0/sequential/conv5/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2
sequential/conv5/BiasAdd�
sequential/conv5/ReluRelu!sequential/conv5/BiasAdd:output:0*
T0*1
_output_shapes
:�����������2
sequential/conv5/Relu�
sequential/ma5/MaxPoolMaxPool#sequential/conv5/Relu:activations:0*1
_output_shapes
:�����������*
ksize
*
paddingVALID*
strides
2
sequential/ma5/MaxPool�
/sequential/batch_normalization_4/ReadVariableOpReadVariableOp8sequential_batch_normalization_4_readvariableop_resource*
_output_shapes	
:�*
dtype021
/sequential/batch_normalization_4/ReadVariableOp�
1sequential/batch_normalization_4/ReadVariableOp_1ReadVariableOp:sequential_batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:�*
dtype023
1sequential/batch_normalization_4/ReadVariableOp_1�
@sequential/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpIsequential_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02B
@sequential/batch_normalization_4/FusedBatchNormV3/ReadVariableOp�
Bsequential/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKsequential_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02D
Bsequential/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1�
1sequential/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3sequential/ma5/MaxPool:output:07sequential/batch_normalization_4/ReadVariableOp:value:09sequential/batch_normalization_4/ReadVariableOp_1:value:0Hsequential/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Jsequential/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:�����������:�:�:�:�:*
epsilon%o�:*
is_training( 23
1sequential/batch_normalization_4/FusedBatchNormV3�
sequential/dropout/IdentityIdentity5sequential/batch_normalization_4/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:�����������2
sequential/dropout/Identity�
*sequential/dense1/Tensordot/ReadVariableOpReadVariableOp3sequential_dense1_tensordot_readvariableop_resource*
_output_shapes
:	�@*
dtype02,
*sequential/dense1/Tensordot/ReadVariableOp�
 sequential/dense1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2"
 sequential/dense1/Tensordot/axes�
 sequential/dense1/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 sequential/dense1/Tensordot/free�
!sequential/dense1/Tensordot/ShapeShape$sequential/dropout/Identity:output:0*
T0*
_output_shapes
:2#
!sequential/dense1/Tensordot/Shape�
)sequential/dense1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)sequential/dense1/Tensordot/GatherV2/axis�
$sequential/dense1/Tensordot/GatherV2GatherV2*sequential/dense1/Tensordot/Shape:output:0)sequential/dense1/Tensordot/free:output:02sequential/dense1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2&
$sequential/dense1/Tensordot/GatherV2�
+sequential/dense1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential/dense1/Tensordot/GatherV2_1/axis�
&sequential/dense1/Tensordot/GatherV2_1GatherV2*sequential/dense1/Tensordot/Shape:output:0)sequential/dense1/Tensordot/axes:output:04sequential/dense1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&sequential/dense1/Tensordot/GatherV2_1�
!sequential/dense1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2#
!sequential/dense1/Tensordot/Const�
 sequential/dense1/Tensordot/ProdProd-sequential/dense1/Tensordot/GatherV2:output:0*sequential/dense1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2"
 sequential/dense1/Tensordot/Prod�
#sequential/dense1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2%
#sequential/dense1/Tensordot/Const_1�
"sequential/dense1/Tensordot/Prod_1Prod/sequential/dense1/Tensordot/GatherV2_1:output:0,sequential/dense1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2$
"sequential/dense1/Tensordot/Prod_1�
'sequential/dense1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential/dense1/Tensordot/concat/axis�
"sequential/dense1/Tensordot/concatConcatV2)sequential/dense1/Tensordot/free:output:0)sequential/dense1/Tensordot/axes:output:00sequential/dense1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2$
"sequential/dense1/Tensordot/concat�
!sequential/dense1/Tensordot/stackPack)sequential/dense1/Tensordot/Prod:output:0+sequential/dense1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2#
!sequential/dense1/Tensordot/stack�
%sequential/dense1/Tensordot/transpose	Transpose$sequential/dropout/Identity:output:0+sequential/dense1/Tensordot/concat:output:0*
T0*1
_output_shapes
:�����������2'
%sequential/dense1/Tensordot/transpose�
#sequential/dense1/Tensordot/ReshapeReshape)sequential/dense1/Tensordot/transpose:y:0*sequential/dense1/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2%
#sequential/dense1/Tensordot/Reshape�
"sequential/dense1/Tensordot/MatMulMatMul,sequential/dense1/Tensordot/Reshape:output:02sequential/dense1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2$
"sequential/dense1/Tensordot/MatMul�
#sequential/dense1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2%
#sequential/dense1/Tensordot/Const_2�
)sequential/dense1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)sequential/dense1/Tensordot/concat_1/axis�
$sequential/dense1/Tensordot/concat_1ConcatV2-sequential/dense1/Tensordot/GatherV2:output:0,sequential/dense1/Tensordot/Const_2:output:02sequential/dense1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2&
$sequential/dense1/Tensordot/concat_1�
sequential/dense1/TensordotReshape,sequential/dense1/Tensordot/MatMul:product:0-sequential/dense1/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:����������@2
sequential/dense1/Tensordot�
(sequential/dense1/BiasAdd/ReadVariableOpReadVariableOp1sequential_dense1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(sequential/dense1/BiasAdd/ReadVariableOp�
sequential/dense1/BiasAddBiasAdd$sequential/dense1/Tensordot:output:00sequential/dense1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������@2
sequential/dense1/BiasAdd�
sequential/dense1/ReluRelu"sequential/dense1/BiasAdd:output:0*
T0*0
_output_shapes
:����������@2
sequential/dense1/Relu�
sequential/dropout_1/IdentityIdentity$sequential/dense1/Relu:activations:0*
T0*0
_output_shapes
:����������@2
sequential/dropout_1/Identity�
*sequential/dense2/Tensordot/ReadVariableOpReadVariableOp3sequential_dense2_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype02,
*sequential/dense2/Tensordot/ReadVariableOp�
 sequential/dense2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2"
 sequential/dense2/Tensordot/axes�
 sequential/dense2/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 sequential/dense2/Tensordot/free�
!sequential/dense2/Tensordot/ShapeShape&sequential/dropout_1/Identity:output:0*
T0*
_output_shapes
:2#
!sequential/dense2/Tensordot/Shape�
)sequential/dense2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)sequential/dense2/Tensordot/GatherV2/axis�
$sequential/dense2/Tensordot/GatherV2GatherV2*sequential/dense2/Tensordot/Shape:output:0)sequential/dense2/Tensordot/free:output:02sequential/dense2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2&
$sequential/dense2/Tensordot/GatherV2�
+sequential/dense2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential/dense2/Tensordot/GatherV2_1/axis�
&sequential/dense2/Tensordot/GatherV2_1GatherV2*sequential/dense2/Tensordot/Shape:output:0)sequential/dense2/Tensordot/axes:output:04sequential/dense2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&sequential/dense2/Tensordot/GatherV2_1�
!sequential/dense2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2#
!sequential/dense2/Tensordot/Const�
 sequential/dense2/Tensordot/ProdProd-sequential/dense2/Tensordot/GatherV2:output:0*sequential/dense2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2"
 sequential/dense2/Tensordot/Prod�
#sequential/dense2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2%
#sequential/dense2/Tensordot/Const_1�
"sequential/dense2/Tensordot/Prod_1Prod/sequential/dense2/Tensordot/GatherV2_1:output:0,sequential/dense2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2$
"sequential/dense2/Tensordot/Prod_1�
'sequential/dense2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential/dense2/Tensordot/concat/axis�
"sequential/dense2/Tensordot/concatConcatV2)sequential/dense2/Tensordot/free:output:0)sequential/dense2/Tensordot/axes:output:00sequential/dense2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2$
"sequential/dense2/Tensordot/concat�
!sequential/dense2/Tensordot/stackPack)sequential/dense2/Tensordot/Prod:output:0+sequential/dense2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2#
!sequential/dense2/Tensordot/stack�
%sequential/dense2/Tensordot/transpose	Transpose&sequential/dropout_1/Identity:output:0+sequential/dense2/Tensordot/concat:output:0*
T0*0
_output_shapes
:����������@2'
%sequential/dense2/Tensordot/transpose�
#sequential/dense2/Tensordot/ReshapeReshape)sequential/dense2/Tensordot/transpose:y:0*sequential/dense2/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2%
#sequential/dense2/Tensordot/Reshape�
"sequential/dense2/Tensordot/MatMulMatMul,sequential/dense2/Tensordot/Reshape:output:02sequential/dense2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2$
"sequential/dense2/Tensordot/MatMul�
#sequential/dense2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#sequential/dense2/Tensordot/Const_2�
)sequential/dense2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)sequential/dense2/Tensordot/concat_1/axis�
$sequential/dense2/Tensordot/concat_1ConcatV2-sequential/dense2/Tensordot/GatherV2:output:0,sequential/dense2/Tensordot/Const_2:output:02sequential/dense2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2&
$sequential/dense2/Tensordot/concat_1�
sequential/dense2/TensordotReshape,sequential/dense2/Tensordot/MatMul:product:0-sequential/dense2/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:����������2
sequential/dense2/Tensordot�
(sequential/dense2/BiasAdd/ReadVariableOpReadVariableOp1sequential_dense2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(sequential/dense2/BiasAdd/ReadVariableOp�
sequential/dense2/BiasAddBiasAdd$sequential/dense2/Tensordot:output:00sequential/dense2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
sequential/dense2/BiasAdd�
sequential/sigmoid/SigmoidSigmoid"sequential/dense2/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
sequential/sigmoid/Sigmoid�
sequential/reshape/ShapeShapesequential/sigmoid/Sigmoid:y:0*
T0*
_output_shapes
:2
sequential/reshape/Shape�
&sequential/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential/reshape/strided_slice/stack�
(sequential/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential/reshape/strided_slice/stack_1�
(sequential/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential/reshape/strided_slice/stack_2�
 sequential/reshape/strided_sliceStridedSlice!sequential/reshape/Shape:output:0/sequential/reshape/strided_slice/stack:output:01sequential/reshape/strided_slice/stack_1:output:01sequential/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 sequential/reshape/strided_slice�
"sequential/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :�2$
"sequential/reshape/Reshape/shape/1�
"sequential/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2$
"sequential/reshape/Reshape/shape/2�
 sequential/reshape/Reshape/shapePack)sequential/reshape/strided_slice:output:0+sequential/reshape/Reshape/shape/1:output:0+sequential/reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2"
 sequential/reshape/Reshape/shape�
sequential/reshape/ReshapeReshapesequential/sigmoid/Sigmoid:y:0)sequential/reshape/Reshape/shape:output:0*
T0*,
_output_shapes
:����������2
sequential/reshape/Reshape|
IdentityIdentity#sequential/reshape/Reshape:output:0*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:����������#:::::::::::::::::::::::::::::::::::\ X
0
_output_shapes
:����������#
$
_user_specified_name
the_inputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: 
�
a
C__inference_dropout_1_layer_call_and_return_conditional_losses_6627

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:����������@2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:����������@2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:����������@:X T
0
_output_shapes
:����������@
 
_user_specified_nameinputs
�
�
M__inference_batch_normalization_layer_call_and_return_conditional_losses_8009

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������:::::j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�

�
?__inference_conv2_layer_call_and_return_conditional_losses_5451

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,����������������������������2
Relu�
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,����������������������������:::j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5748

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������:::::j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�$
�
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_8315

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��#AssignMovingAvg/AssignSubVariableOp�%AssignMovingAvg_1/AssignSubVariableOpu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�$
�
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_8477

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��#AssignMovingAvg/AssignSubVariableOp�%AssignMovingAvg_1/AssignSubVariableOpu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:�����������:�:�:�:�:*
epsilon%o�:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:�����������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_6317

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:�����������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3r
IdentityIdentityFusedBatchNormV3:y:0*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:�����������:::::Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_8657

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:�����������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3r
IdentityIdentityFusedBatchNormV3:y:0*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:�����������:::::Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_5588

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������:::::j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
B
&__inference_sigmoid_layer_call_fn_8826

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_sigmoid_layer_call_and_return_conditional_losses_66912
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�f
�
D__inference_sequential_layer_call_and_return_conditional_losses_6910

inputs

conv1_6820

conv1_6822
batch_normalization_6826
batch_normalization_6828
batch_normalization_6830
batch_normalization_6832

conv2_6835

conv2_6837
batch_normalization_1_6841
batch_normalization_1_6843
batch_normalization_1_6845
batch_normalization_1_6847

conv3_6850

conv3_6852
batch_normalization_2_6856
batch_normalization_2_6858
batch_normalization_2_6860
batch_normalization_2_6862

conv4_6865

conv4_6867
batch_normalization_3_6871
batch_normalization_3_6873
batch_normalization_3_6875
batch_normalization_3_6877

conv5_6880

conv5_6882
batch_normalization_4_6886
batch_normalization_4_6888
batch_normalization_4_6890
batch_normalization_4_6892
dense1_6896
dense1_6898
dense2_6902
dense2_6904
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�-batch_normalization_3/StatefulPartitionedCall�-batch_normalization_4/StatefulPartitionedCall�conv1/StatefulPartitionedCall�conv2/StatefulPartitionedCall�conv3/StatefulPartitionedCall�conv4/StatefulPartitionedCall�conv5/StatefulPartitionedCall�dense1/StatefulPartitionedCall�dense2/StatefulPartitionedCall�dropout/StatefulPartitionedCall�!dropout_1/StatefulPartitionedCall�
conv1/StatefulPartitionedCallStatefulPartitionedCallinputs
conv1_6820
conv1_6822*
Tin
2*
Tout
2*1
_output_shapes
:����������#�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_conv1_layer_call_and_return_conditional_losses_52912
conv1/StatefulPartitionedCall�
max1/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_max1_layer_call_and_return_conditional_losses_53072
max1/PartitionedCall�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCallmax1/PartitionedCall:output:0batch_normalization_6826batch_normalization_6828batch_normalization_6830batch_normalization_6832*
Tin	
2*
Tout
2*1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_61192-
+batch_normalization/StatefulPartitionedCall�
conv2/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0
conv2_6835
conv2_6837*
Tin
2*
Tout
2*1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_conv2_layer_call_and_return_conditional_losses_54512
conv2/StatefulPartitionedCall�
max2/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_max2_layer_call_and_return_conditional_losses_54672
max2/PartitionedCall�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCallmax2/PartitionedCall:output:0batch_normalization_1_6841batch_normalization_1_6843batch_normalization_1_6845batch_normalization_1_6847*
Tin	
2*
Tout
2*1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_62092/
-batch_normalization_1/StatefulPartitionedCall�
conv3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0
conv3_6850
conv3_6852*
Tin
2*
Tout
2*1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_conv3_layer_call_and_return_conditional_losses_56112
conv3/StatefulPartitionedCall�
ma3/PartitionedCallPartitionedCall&conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*F
fAR?
=__inference_ma3_layer_call_and_return_conditional_losses_56272
ma3/PartitionedCall�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCallma3/PartitionedCall:output:0batch_normalization_2_6856batch_normalization_2_6858batch_normalization_2_6860batch_normalization_2_6862*
Tin	
2*
Tout
2*1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_62992/
-batch_normalization_2/StatefulPartitionedCall�
conv4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0
conv4_6865
conv4_6867*
Tin
2*
Tout
2*1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_conv4_layer_call_and_return_conditional_losses_57712
conv4/StatefulPartitionedCall�
ma4/PartitionedCallPartitionedCall&conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*F
fAR?
=__inference_ma4_layer_call_and_return_conditional_losses_57872
ma4/PartitionedCall�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCallma4/PartitionedCall:output:0batch_normalization_3_6871batch_normalization_3_6873batch_normalization_3_6875batch_normalization_3_6877*
Tin	
2*
Tout
2*1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_63892/
-batch_normalization_3/StatefulPartitionedCall�
conv5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0
conv5_6880
conv5_6882*
Tin
2*
Tout
2*1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_conv5_layer_call_and_return_conditional_losses_59312
conv5/StatefulPartitionedCall�
ma5/PartitionedCallPartitionedCall&conv5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*F
fAR?
=__inference_ma5_layer_call_and_return_conditional_losses_59472
ma5/PartitionedCall�
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCallma5/PartitionedCall:output:0batch_normalization_4_6886batch_normalization_4_6888batch_normalization_4_6890batch_normalization_4_6892*
Tin	
2*
Tout
2*1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_64792/
-batch_normalization_4/StatefulPartitionedCall�
dropout/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_65452!
dropout/StatefulPartitionedCall�
dense1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense1_6896dense1_6898*
Tin
2*
Tout
2*0
_output_shapes
:����������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_dense1_layer_call_and_return_conditional_losses_65942 
dense1/StatefulPartitionedCall�
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*0
_output_shapes
:����������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_66222#
!dropout_1/StatefulPartitionedCall�
dense2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense2_6902dense2_6904*
Tin
2*
Tout
2*0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_dense2_layer_call_and_return_conditional_losses_66702 
dense2/StatefulPartitionedCall�
sigmoid/PartitionedCallPartitionedCall'dense2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_sigmoid_layer_call_and_return_conditional_losses_66912
sigmoid/PartitionedCall�
reshape/PartitionedCallPartitionedCall sigmoid/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_67122
reshape/PartitionedCall�
IdentityIdentity reshape/PartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv3/StatefulPartitionedCall^conv4/StatefulPartitionedCall^conv5/StatefulPartitionedCall^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:����������#::::::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2>
conv4/StatefulPartitionedCallconv4/StatefulPartitionedCall2>
conv5/StatefulPartitionedCallconv5/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:X T
0
_output_shapes
:����������#
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: 
�$
�
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_8564

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��#AssignMovingAvg/AssignSubVariableOp�%AssignMovingAvg_1/AssignSubVariableOpu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
4__inference_batch_normalization_3_layer_call_fn_8446

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_59082
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
]
A__inference_reshape_layer_call_and_return_conditional_losses_8839

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicee
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :�2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapet
ReshapeReshapeinputsReshape/shape:output:0*
T0*,
_output_shapes
:����������2	
Reshapei
IdentityIdentityReshape:output:0*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
@__inference_dense1_layer_call_and_return_conditional_losses_8741

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	�@*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesu
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*1
_output_shapes
:�����������2
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*0
_output_shapes
:����������@2
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������@2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������@2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:�����������:::Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ȅ
�2
 __inference__traced_restore_9429
file_prefix!
assignvariableop_conv1_kernel!
assignvariableop_1_conv1_bias0
,assignvariableop_2_batch_normalization_gamma/
+assignvariableop_3_batch_normalization_beta6
2assignvariableop_4_batch_normalization_moving_mean:
6assignvariableop_5_batch_normalization_moving_variance#
assignvariableop_6_conv2_kernel!
assignvariableop_7_conv2_bias2
.assignvariableop_8_batch_normalization_1_gamma1
-assignvariableop_9_batch_normalization_1_beta9
5assignvariableop_10_batch_normalization_1_moving_mean=
9assignvariableop_11_batch_normalization_1_moving_variance$
 assignvariableop_12_conv3_kernel"
assignvariableop_13_conv3_bias3
/assignvariableop_14_batch_normalization_2_gamma2
.assignvariableop_15_batch_normalization_2_beta9
5assignvariableop_16_batch_normalization_2_moving_mean=
9assignvariableop_17_batch_normalization_2_moving_variance$
 assignvariableop_18_conv4_kernel"
assignvariableop_19_conv4_bias3
/assignvariableop_20_batch_normalization_3_gamma2
.assignvariableop_21_batch_normalization_3_beta9
5assignvariableop_22_batch_normalization_3_moving_mean=
9assignvariableop_23_batch_normalization_3_moving_variance$
 assignvariableop_24_conv5_kernel"
assignvariableop_25_conv5_bias3
/assignvariableop_26_batch_normalization_4_gamma2
.assignvariableop_27_batch_normalization_4_beta9
5assignvariableop_28_batch_normalization_4_moving_mean=
9assignvariableop_29_batch_normalization_4_moving_variance%
!assignvariableop_30_dense1_kernel#
assignvariableop_31_dense1_bias%
!assignvariableop_32_dense2_kernel#
assignvariableop_33_dense2_bias!
assignvariableop_34_adam_iter#
assignvariableop_35_adam_beta_1#
assignvariableop_36_adam_beta_2"
assignvariableop_37_adam_decay*
&assignvariableop_38_adam_learning_rate
assignvariableop_39_total
assignvariableop_40_count
assignvariableop_41_total_1
assignvariableop_42_count_1+
'assignvariableop_43_adam_conv1_kernel_m)
%assignvariableop_44_adam_conv1_bias_m8
4assignvariableop_45_adam_batch_normalization_gamma_m7
3assignvariableop_46_adam_batch_normalization_beta_m+
'assignvariableop_47_adam_conv2_kernel_m)
%assignvariableop_48_adam_conv2_bias_m:
6assignvariableop_49_adam_batch_normalization_1_gamma_m9
5assignvariableop_50_adam_batch_normalization_1_beta_m+
'assignvariableop_51_adam_conv3_kernel_m)
%assignvariableop_52_adam_conv3_bias_m:
6assignvariableop_53_adam_batch_normalization_2_gamma_m9
5assignvariableop_54_adam_batch_normalization_2_beta_m+
'assignvariableop_55_adam_conv4_kernel_m)
%assignvariableop_56_adam_conv4_bias_m:
6assignvariableop_57_adam_batch_normalization_3_gamma_m9
5assignvariableop_58_adam_batch_normalization_3_beta_m+
'assignvariableop_59_adam_conv5_kernel_m)
%assignvariableop_60_adam_conv5_bias_m:
6assignvariableop_61_adam_batch_normalization_4_gamma_m9
5assignvariableop_62_adam_batch_normalization_4_beta_m,
(assignvariableop_63_adam_dense1_kernel_m*
&assignvariableop_64_adam_dense1_bias_m,
(assignvariableop_65_adam_dense2_kernel_m*
&assignvariableop_66_adam_dense2_bias_m+
'assignvariableop_67_adam_conv1_kernel_v)
%assignvariableop_68_adam_conv1_bias_v8
4assignvariableop_69_adam_batch_normalization_gamma_v7
3assignvariableop_70_adam_batch_normalization_beta_v+
'assignvariableop_71_adam_conv2_kernel_v)
%assignvariableop_72_adam_conv2_bias_v:
6assignvariableop_73_adam_batch_normalization_1_gamma_v9
5assignvariableop_74_adam_batch_normalization_1_beta_v+
'assignvariableop_75_adam_conv3_kernel_v)
%assignvariableop_76_adam_conv3_bias_v:
6assignvariableop_77_adam_batch_normalization_2_gamma_v9
5assignvariableop_78_adam_batch_normalization_2_beta_v+
'assignvariableop_79_adam_conv4_kernel_v)
%assignvariableop_80_adam_conv4_bias_v:
6assignvariableop_81_adam_batch_normalization_3_gamma_v9
5assignvariableop_82_adam_batch_normalization_3_beta_v+
'assignvariableop_83_adam_conv5_kernel_v)
%assignvariableop_84_adam_conv5_bias_v:
6assignvariableop_85_adam_batch_normalization_4_gamma_v9
5assignvariableop_86_adam_batch_normalization_4_beta_v,
(assignvariableop_87_adam_dense1_kernel_v*
&assignvariableop_88_adam_dense1_bias_v,
(assignvariableop_89_adam_dense2_kernel_v*
&assignvariableop_90_adam_dense2_bias_v
identity_92��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_69�AssignVariableOp_7�AssignVariableOp_70�AssignVariableOp_71�AssignVariableOp_72�AssignVariableOp_73�AssignVariableOp_74�AssignVariableOp_75�AssignVariableOp_76�AssignVariableOp_77�AssignVariableOp_78�AssignVariableOp_79�AssignVariableOp_8�AssignVariableOp_80�AssignVariableOp_81�AssignVariableOp_82�AssignVariableOp_83�AssignVariableOp_84�AssignVariableOp_85�AssignVariableOp_86�AssignVariableOp_87�AssignVariableOp_88�AssignVariableOp_89�AssignVariableOp_9�AssignVariableOp_90�	RestoreV2�RestoreV2_1�2
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:[*
dtype0*�1
value�1B�1[B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:[*
dtype0*�
value�B�[B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*i
dtypes_
]2[	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpassignvariableop_conv1_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv1_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp,assignvariableop_2_batch_normalization_gammaIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp+assignvariableop_3_batch_normalization_betaIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp2assignvariableop_4_batch_normalization_moving_meanIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp6assignvariableop_5_batch_normalization_moving_varianceIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOpassignvariableop_6_conv2_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpassignvariableop_7_conv2_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_1_gammaIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batch_normalization_1_betaIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp5assignvariableop_10_batch_normalization_1_moving_meanIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp9assignvariableop_11_batch_normalization_1_moving_varianceIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp assignvariableop_12_conv3_kernelIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOpassignvariableop_13_conv3_biasIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp/assignvariableop_14_batch_normalization_2_gammaIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp.assignvariableop_15_batch_normalization_2_betaIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp5assignvariableop_16_batch_normalization_2_moving_meanIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp9assignvariableop_17_batch_normalization_2_moving_varianceIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp assignvariableop_18_conv4_kernelIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOpassignvariableop_19_conv4_biasIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp/assignvariableop_20_batch_normalization_3_gammaIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp.assignvariableop_21_batch_normalization_3_betaIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp5assignvariableop_22_batch_normalization_3_moving_meanIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp9assignvariableop_23_batch_normalization_3_moving_varianceIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp assignvariableop_24_conv5_kernelIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOpassignvariableop_25_conv5_biasIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp/assignvariableop_26_batch_normalization_4_gammaIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp.assignvariableop_27_batch_normalization_4_betaIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp5assignvariableop_28_batch_normalization_4_moving_meanIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp9assignvariableop_29_batch_normalization_4_moving_varianceIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp!assignvariableop_30_dense1_kernelIdentity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOpassignvariableop_31_dense1_biasIdentity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp!assignvariableop_32_dense2_kernelIdentity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOpassignvariableop_33_dense2_biasIdentity_33:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_33_
Identity_34IdentityRestoreV2:tensors:34*
T0	*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOpassignvariableop_34_adam_iterIdentity_34:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_34_
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOpassignvariableop_35_adam_beta_1Identity_35:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_35_
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOpassignvariableop_36_adam_beta_2Identity_36:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_36_
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOpassignvariableop_37_adam_decayIdentity_37:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_37_
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOp&assignvariableop_38_adam_learning_rateIdentity_38:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_38_
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:2
Identity_39�
AssignVariableOp_39AssignVariableOpassignvariableop_39_totalIdentity_39:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_39_
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:2
Identity_40�
AssignVariableOp_40AssignVariableOpassignvariableop_40_countIdentity_40:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_40_
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:2
Identity_41�
AssignVariableOp_41AssignVariableOpassignvariableop_41_total_1Identity_41:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_41_
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:2
Identity_42�
AssignVariableOp_42AssignVariableOpassignvariableop_42_count_1Identity_42:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_42_
Identity_43IdentityRestoreV2:tensors:43*
T0*
_output_shapes
:2
Identity_43�
AssignVariableOp_43AssignVariableOp'assignvariableop_43_adam_conv1_kernel_mIdentity_43:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_43_
Identity_44IdentityRestoreV2:tensors:44*
T0*
_output_shapes
:2
Identity_44�
AssignVariableOp_44AssignVariableOp%assignvariableop_44_adam_conv1_bias_mIdentity_44:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_44_
Identity_45IdentityRestoreV2:tensors:45*
T0*
_output_shapes
:2
Identity_45�
AssignVariableOp_45AssignVariableOp4assignvariableop_45_adam_batch_normalization_gamma_mIdentity_45:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_45_
Identity_46IdentityRestoreV2:tensors:46*
T0*
_output_shapes
:2
Identity_46�
AssignVariableOp_46AssignVariableOp3assignvariableop_46_adam_batch_normalization_beta_mIdentity_46:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_46_
Identity_47IdentityRestoreV2:tensors:47*
T0*
_output_shapes
:2
Identity_47�
AssignVariableOp_47AssignVariableOp'assignvariableop_47_adam_conv2_kernel_mIdentity_47:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_47_
Identity_48IdentityRestoreV2:tensors:48*
T0*
_output_shapes
:2
Identity_48�
AssignVariableOp_48AssignVariableOp%assignvariableop_48_adam_conv2_bias_mIdentity_48:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_48_
Identity_49IdentityRestoreV2:tensors:49*
T0*
_output_shapes
:2
Identity_49�
AssignVariableOp_49AssignVariableOp6assignvariableop_49_adam_batch_normalization_1_gamma_mIdentity_49:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_49_
Identity_50IdentityRestoreV2:tensors:50*
T0*
_output_shapes
:2
Identity_50�
AssignVariableOp_50AssignVariableOp5assignvariableop_50_adam_batch_normalization_1_beta_mIdentity_50:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_50_
Identity_51IdentityRestoreV2:tensors:51*
T0*
_output_shapes
:2
Identity_51�
AssignVariableOp_51AssignVariableOp'assignvariableop_51_adam_conv3_kernel_mIdentity_51:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_51_
Identity_52IdentityRestoreV2:tensors:52*
T0*
_output_shapes
:2
Identity_52�
AssignVariableOp_52AssignVariableOp%assignvariableop_52_adam_conv3_bias_mIdentity_52:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_52_
Identity_53IdentityRestoreV2:tensors:53*
T0*
_output_shapes
:2
Identity_53�
AssignVariableOp_53AssignVariableOp6assignvariableop_53_adam_batch_normalization_2_gamma_mIdentity_53:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_53_
Identity_54IdentityRestoreV2:tensors:54*
T0*
_output_shapes
:2
Identity_54�
AssignVariableOp_54AssignVariableOp5assignvariableop_54_adam_batch_normalization_2_beta_mIdentity_54:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_54_
Identity_55IdentityRestoreV2:tensors:55*
T0*
_output_shapes
:2
Identity_55�
AssignVariableOp_55AssignVariableOp'assignvariableop_55_adam_conv4_kernel_mIdentity_55:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_55_
Identity_56IdentityRestoreV2:tensors:56*
T0*
_output_shapes
:2
Identity_56�
AssignVariableOp_56AssignVariableOp%assignvariableop_56_adam_conv4_bias_mIdentity_56:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_56_
Identity_57IdentityRestoreV2:tensors:57*
T0*
_output_shapes
:2
Identity_57�
AssignVariableOp_57AssignVariableOp6assignvariableop_57_adam_batch_normalization_3_gamma_mIdentity_57:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_57_
Identity_58IdentityRestoreV2:tensors:58*
T0*
_output_shapes
:2
Identity_58�
AssignVariableOp_58AssignVariableOp5assignvariableop_58_adam_batch_normalization_3_beta_mIdentity_58:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_58_
Identity_59IdentityRestoreV2:tensors:59*
T0*
_output_shapes
:2
Identity_59�
AssignVariableOp_59AssignVariableOp'assignvariableop_59_adam_conv5_kernel_mIdentity_59:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_59_
Identity_60IdentityRestoreV2:tensors:60*
T0*
_output_shapes
:2
Identity_60�
AssignVariableOp_60AssignVariableOp%assignvariableop_60_adam_conv5_bias_mIdentity_60:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_60_
Identity_61IdentityRestoreV2:tensors:61*
T0*
_output_shapes
:2
Identity_61�
AssignVariableOp_61AssignVariableOp6assignvariableop_61_adam_batch_normalization_4_gamma_mIdentity_61:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_61_
Identity_62IdentityRestoreV2:tensors:62*
T0*
_output_shapes
:2
Identity_62�
AssignVariableOp_62AssignVariableOp5assignvariableop_62_adam_batch_normalization_4_beta_mIdentity_62:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_62_
Identity_63IdentityRestoreV2:tensors:63*
T0*
_output_shapes
:2
Identity_63�
AssignVariableOp_63AssignVariableOp(assignvariableop_63_adam_dense1_kernel_mIdentity_63:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_63_
Identity_64IdentityRestoreV2:tensors:64*
T0*
_output_shapes
:2
Identity_64�
AssignVariableOp_64AssignVariableOp&assignvariableop_64_adam_dense1_bias_mIdentity_64:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_64_
Identity_65IdentityRestoreV2:tensors:65*
T0*
_output_shapes
:2
Identity_65�
AssignVariableOp_65AssignVariableOp(assignvariableop_65_adam_dense2_kernel_mIdentity_65:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_65_
Identity_66IdentityRestoreV2:tensors:66*
T0*
_output_shapes
:2
Identity_66�
AssignVariableOp_66AssignVariableOp&assignvariableop_66_adam_dense2_bias_mIdentity_66:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_66_
Identity_67IdentityRestoreV2:tensors:67*
T0*
_output_shapes
:2
Identity_67�
AssignVariableOp_67AssignVariableOp'assignvariableop_67_adam_conv1_kernel_vIdentity_67:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_67_
Identity_68IdentityRestoreV2:tensors:68*
T0*
_output_shapes
:2
Identity_68�
AssignVariableOp_68AssignVariableOp%assignvariableop_68_adam_conv1_bias_vIdentity_68:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_68_
Identity_69IdentityRestoreV2:tensors:69*
T0*
_output_shapes
:2
Identity_69�
AssignVariableOp_69AssignVariableOp4assignvariableop_69_adam_batch_normalization_gamma_vIdentity_69:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_69_
Identity_70IdentityRestoreV2:tensors:70*
T0*
_output_shapes
:2
Identity_70�
AssignVariableOp_70AssignVariableOp3assignvariableop_70_adam_batch_normalization_beta_vIdentity_70:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_70_
Identity_71IdentityRestoreV2:tensors:71*
T0*
_output_shapes
:2
Identity_71�
AssignVariableOp_71AssignVariableOp'assignvariableop_71_adam_conv2_kernel_vIdentity_71:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_71_
Identity_72IdentityRestoreV2:tensors:72*
T0*
_output_shapes
:2
Identity_72�
AssignVariableOp_72AssignVariableOp%assignvariableop_72_adam_conv2_bias_vIdentity_72:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_72_
Identity_73IdentityRestoreV2:tensors:73*
T0*
_output_shapes
:2
Identity_73�
AssignVariableOp_73AssignVariableOp6assignvariableop_73_adam_batch_normalization_1_gamma_vIdentity_73:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_73_
Identity_74IdentityRestoreV2:tensors:74*
T0*
_output_shapes
:2
Identity_74�
AssignVariableOp_74AssignVariableOp5assignvariableop_74_adam_batch_normalization_1_beta_vIdentity_74:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_74_
Identity_75IdentityRestoreV2:tensors:75*
T0*
_output_shapes
:2
Identity_75�
AssignVariableOp_75AssignVariableOp'assignvariableop_75_adam_conv3_kernel_vIdentity_75:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_75_
Identity_76IdentityRestoreV2:tensors:76*
T0*
_output_shapes
:2
Identity_76�
AssignVariableOp_76AssignVariableOp%assignvariableop_76_adam_conv3_bias_vIdentity_76:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_76_
Identity_77IdentityRestoreV2:tensors:77*
T0*
_output_shapes
:2
Identity_77�
AssignVariableOp_77AssignVariableOp6assignvariableop_77_adam_batch_normalization_2_gamma_vIdentity_77:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_77_
Identity_78IdentityRestoreV2:tensors:78*
T0*
_output_shapes
:2
Identity_78�
AssignVariableOp_78AssignVariableOp5assignvariableop_78_adam_batch_normalization_2_beta_vIdentity_78:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_78_
Identity_79IdentityRestoreV2:tensors:79*
T0*
_output_shapes
:2
Identity_79�
AssignVariableOp_79AssignVariableOp'assignvariableop_79_adam_conv4_kernel_vIdentity_79:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_79_
Identity_80IdentityRestoreV2:tensors:80*
T0*
_output_shapes
:2
Identity_80�
AssignVariableOp_80AssignVariableOp%assignvariableop_80_adam_conv4_bias_vIdentity_80:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_80_
Identity_81IdentityRestoreV2:tensors:81*
T0*
_output_shapes
:2
Identity_81�
AssignVariableOp_81AssignVariableOp6assignvariableop_81_adam_batch_normalization_3_gamma_vIdentity_81:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_81_
Identity_82IdentityRestoreV2:tensors:82*
T0*
_output_shapes
:2
Identity_82�
AssignVariableOp_82AssignVariableOp5assignvariableop_82_adam_batch_normalization_3_beta_vIdentity_82:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_82_
Identity_83IdentityRestoreV2:tensors:83*
T0*
_output_shapes
:2
Identity_83�
AssignVariableOp_83AssignVariableOp'assignvariableop_83_adam_conv5_kernel_vIdentity_83:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_83_
Identity_84IdentityRestoreV2:tensors:84*
T0*
_output_shapes
:2
Identity_84�
AssignVariableOp_84AssignVariableOp%assignvariableop_84_adam_conv5_bias_vIdentity_84:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_84_
Identity_85IdentityRestoreV2:tensors:85*
T0*
_output_shapes
:2
Identity_85�
AssignVariableOp_85AssignVariableOp6assignvariableop_85_adam_batch_normalization_4_gamma_vIdentity_85:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_85_
Identity_86IdentityRestoreV2:tensors:86*
T0*
_output_shapes
:2
Identity_86�
AssignVariableOp_86AssignVariableOp5assignvariableop_86_adam_batch_normalization_4_beta_vIdentity_86:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_86_
Identity_87IdentityRestoreV2:tensors:87*
T0*
_output_shapes
:2
Identity_87�
AssignVariableOp_87AssignVariableOp(assignvariableop_87_adam_dense1_kernel_vIdentity_87:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_87_
Identity_88IdentityRestoreV2:tensors:88*
T0*
_output_shapes
:2
Identity_88�
AssignVariableOp_88AssignVariableOp&assignvariableop_88_adam_dense1_bias_vIdentity_88:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_88_
Identity_89IdentityRestoreV2:tensors:89*
T0*
_output_shapes
:2
Identity_89�
AssignVariableOp_89AssignVariableOp(assignvariableop_89_adam_dense2_kernel_vIdentity_89:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_89_
Identity_90IdentityRestoreV2:tensors:90*
T0*
_output_shapes
:2
Identity_90�
AssignVariableOp_90AssignVariableOp&assignvariableop_90_adam_dense2_bias_vIdentity_90:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_90�
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names�
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices�
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_91Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_91�
Identity_92IdentityIdentity_91:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_92"#
identity_92Identity_92:output:0*�
_input_shapes�
�: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: :1

_output_shapes
: :2

_output_shapes
: :3

_output_shapes
: :4

_output_shapes
: :5

_output_shapes
: :6

_output_shapes
: :7

_output_shapes
: :8

_output_shapes
: :9

_output_shapes
: ::

_output_shapes
: :;

_output_shapes
: :<

_output_shapes
: :=

_output_shapes
: :>

_output_shapes
: :?

_output_shapes
: :@

_output_shapes
: :A

_output_shapes
: :B

_output_shapes
: :C

_output_shapes
: :D

_output_shapes
: :E

_output_shapes
: :F

_output_shapes
: :G

_output_shapes
: :H

_output_shapes
: :I

_output_shapes
: :J

_output_shapes
: :K

_output_shapes
: :L

_output_shapes
: :M

_output_shapes
: :N

_output_shapes
: :O

_output_shapes
: :P

_output_shapes
: :Q

_output_shapes
: :R

_output_shapes
: :S

_output_shapes
: :T

_output_shapes
: :U

_output_shapes
: :V

_output_shapes
: :W

_output_shapes
: :X

_output_shapes
: :Y

_output_shapes
: :Z

_output_shapes
: :[

_output_shapes
: 
�

�
?__inference_conv5_layer_call_and_return_conditional_losses_5931

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,����������������������������2
Relu�
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,����������������������������:::j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
4__inference_batch_normalization_4_layer_call_fn_8608

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_60682
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�$
�
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_6037

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��#AssignMovingAvg/AssignSubVariableOp�%AssignMovingAvg_1/AssignSubVariableOpu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
?
#__inference_max2_layer_call_fn_5473

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_max2_layer_call_and_return_conditional_losses_54672
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
4__inference_batch_normalization_2_layer_call_fn_8284

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*1
_output_shapes
:�����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_63172
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:�����������::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�d
�
D__inference_sequential_layer_call_and_return_conditional_losses_6814

the_inputs

conv1_6724

conv1_6726
batch_normalization_6730
batch_normalization_6732
batch_normalization_6734
batch_normalization_6736

conv2_6739

conv2_6741
batch_normalization_1_6745
batch_normalization_1_6747
batch_normalization_1_6749
batch_normalization_1_6751

conv3_6754

conv3_6756
batch_normalization_2_6760
batch_normalization_2_6762
batch_normalization_2_6764
batch_normalization_2_6766

conv4_6769

conv4_6771
batch_normalization_3_6775
batch_normalization_3_6777
batch_normalization_3_6779
batch_normalization_3_6781

conv5_6784

conv5_6786
batch_normalization_4_6790
batch_normalization_4_6792
batch_normalization_4_6794
batch_normalization_4_6796
dense1_6800
dense1_6802
dense2_6806
dense2_6808
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�-batch_normalization_3/StatefulPartitionedCall�-batch_normalization_4/StatefulPartitionedCall�conv1/StatefulPartitionedCall�conv2/StatefulPartitionedCall�conv3/StatefulPartitionedCall�conv4/StatefulPartitionedCall�conv5/StatefulPartitionedCall�dense1/StatefulPartitionedCall�dense2/StatefulPartitionedCall�
conv1/StatefulPartitionedCallStatefulPartitionedCall
the_inputs
conv1_6724
conv1_6726*
Tin
2*
Tout
2*1
_output_shapes
:����������#�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_conv1_layer_call_and_return_conditional_losses_52912
conv1/StatefulPartitionedCall�
max1/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_max1_layer_call_and_return_conditional_losses_53072
max1/PartitionedCall�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCallmax1/PartitionedCall:output:0batch_normalization_6730batch_normalization_6732batch_normalization_6734batch_normalization_6736*
Tin	
2*
Tout
2*1
_output_shapes
:�����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_61372-
+batch_normalization/StatefulPartitionedCall�
conv2/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0
conv2_6739
conv2_6741*
Tin
2*
Tout
2*1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_conv2_layer_call_and_return_conditional_losses_54512
conv2/StatefulPartitionedCall�
max2/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_max2_layer_call_and_return_conditional_losses_54672
max2/PartitionedCall�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCallmax2/PartitionedCall:output:0batch_normalization_1_6745batch_normalization_1_6747batch_normalization_1_6749batch_normalization_1_6751*
Tin	
2*
Tout
2*1
_output_shapes
:�����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_62272/
-batch_normalization_1/StatefulPartitionedCall�
conv3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0
conv3_6754
conv3_6756*
Tin
2*
Tout
2*1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_conv3_layer_call_and_return_conditional_losses_56112
conv3/StatefulPartitionedCall�
ma3/PartitionedCallPartitionedCall&conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*F
fAR?
=__inference_ma3_layer_call_and_return_conditional_losses_56272
ma3/PartitionedCall�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCallma3/PartitionedCall:output:0batch_normalization_2_6760batch_normalization_2_6762batch_normalization_2_6764batch_normalization_2_6766*
Tin	
2*
Tout
2*1
_output_shapes
:�����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_63172/
-batch_normalization_2/StatefulPartitionedCall�
conv4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0
conv4_6769
conv4_6771*
Tin
2*
Tout
2*1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_conv4_layer_call_and_return_conditional_losses_57712
conv4/StatefulPartitionedCall�
ma4/PartitionedCallPartitionedCall&conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*F
fAR?
=__inference_ma4_layer_call_and_return_conditional_losses_57872
ma4/PartitionedCall�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCallma4/PartitionedCall:output:0batch_normalization_3_6775batch_normalization_3_6777batch_normalization_3_6779batch_normalization_3_6781*
Tin	
2*
Tout
2*1
_output_shapes
:�����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_64072/
-batch_normalization_3/StatefulPartitionedCall�
conv5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0
conv5_6784
conv5_6786*
Tin
2*
Tout
2*1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_conv5_layer_call_and_return_conditional_losses_59312
conv5/StatefulPartitionedCall�
ma5/PartitionedCallPartitionedCall&conv5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*F
fAR?
=__inference_ma5_layer_call_and_return_conditional_losses_59472
ma5/PartitionedCall�
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCallma5/PartitionedCall:output:0batch_normalization_4_6790batch_normalization_4_6792batch_normalization_4_6794batch_normalization_4_6796*
Tin	
2*
Tout
2*1
_output_shapes
:�����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_64972/
-batch_normalization_4/StatefulPartitionedCall�
dropout/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_65502
dropout/PartitionedCall�
dense1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense1_6800dense1_6802*
Tin
2*
Tout
2*0
_output_shapes
:����������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_dense1_layer_call_and_return_conditional_losses_65942 
dense1/StatefulPartitionedCall�
dropout_1/PartitionedCallPartitionedCall'dense1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:����������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_66272
dropout_1/PartitionedCall�
dense2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense2_6806dense2_6808*
Tin
2*
Tout
2*0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_dense2_layer_call_and_return_conditional_losses_66702 
dense2/StatefulPartitionedCall�
sigmoid/PartitionedCallPartitionedCall'dense2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_sigmoid_layer_call_and_return_conditional_losses_66912
sigmoid/PartitionedCall�
reshape/PartitionedCallPartitionedCall sigmoid/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_67122
reshape/PartitionedCall�
IdentityIdentity reshape/PartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv3/StatefulPartitionedCall^conv4/StatefulPartitionedCall^conv5/StatefulPartitionedCall^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:����������#::::::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2>
conv4/StatefulPartitionedCallconv4/StatefulPartitionedCall2>
conv5/StatefulPartitionedCallconv5/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall:\ X
0
_output_shapes
:����������#
$
_user_specified_name
the_inputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: 
�
�
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_6068

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������:::::j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
]
A__inference_sigmoid_layer_call_and_return_conditional_losses_8821

inputs
identity`
SigmoidSigmoidinputs*
T0*0
_output_shapes
:����������2	
Sigmoidh
IdentityIdentitySigmoid:y:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�f
�
D__inference_sequential_layer_call_and_return_conditional_losses_6721

the_inputs

conv1_6083

conv1_6085
batch_normalization_6164
batch_normalization_6166
batch_normalization_6168
batch_normalization_6170

conv2_6173

conv2_6175
batch_normalization_1_6254
batch_normalization_1_6256
batch_normalization_1_6258
batch_normalization_1_6260

conv3_6263

conv3_6265
batch_normalization_2_6344
batch_normalization_2_6346
batch_normalization_2_6348
batch_normalization_2_6350

conv4_6353

conv4_6355
batch_normalization_3_6434
batch_normalization_3_6436
batch_normalization_3_6438
batch_normalization_3_6440

conv5_6443

conv5_6445
batch_normalization_4_6524
batch_normalization_4_6526
batch_normalization_4_6528
batch_normalization_4_6530
dense1_6605
dense1_6607
dense2_6681
dense2_6683
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�-batch_normalization_3/StatefulPartitionedCall�-batch_normalization_4/StatefulPartitionedCall�conv1/StatefulPartitionedCall�conv2/StatefulPartitionedCall�conv3/StatefulPartitionedCall�conv4/StatefulPartitionedCall�conv5/StatefulPartitionedCall�dense1/StatefulPartitionedCall�dense2/StatefulPartitionedCall�dropout/StatefulPartitionedCall�!dropout_1/StatefulPartitionedCall�
conv1/StatefulPartitionedCallStatefulPartitionedCall
the_inputs
conv1_6083
conv1_6085*
Tin
2*
Tout
2*1
_output_shapes
:����������#�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_conv1_layer_call_and_return_conditional_losses_52912
conv1/StatefulPartitionedCall�
max1/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_max1_layer_call_and_return_conditional_losses_53072
max1/PartitionedCall�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCallmax1/PartitionedCall:output:0batch_normalization_6164batch_normalization_6166batch_normalization_6168batch_normalization_6170*
Tin	
2*
Tout
2*1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_61192-
+batch_normalization/StatefulPartitionedCall�
conv2/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0
conv2_6173
conv2_6175*
Tin
2*
Tout
2*1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_conv2_layer_call_and_return_conditional_losses_54512
conv2/StatefulPartitionedCall�
max2/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_max2_layer_call_and_return_conditional_losses_54672
max2/PartitionedCall�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCallmax2/PartitionedCall:output:0batch_normalization_1_6254batch_normalization_1_6256batch_normalization_1_6258batch_normalization_1_6260*
Tin	
2*
Tout
2*1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_62092/
-batch_normalization_1/StatefulPartitionedCall�
conv3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0
conv3_6263
conv3_6265*
Tin
2*
Tout
2*1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_conv3_layer_call_and_return_conditional_losses_56112
conv3/StatefulPartitionedCall�
ma3/PartitionedCallPartitionedCall&conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*F
fAR?
=__inference_ma3_layer_call_and_return_conditional_losses_56272
ma3/PartitionedCall�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCallma3/PartitionedCall:output:0batch_normalization_2_6344batch_normalization_2_6346batch_normalization_2_6348batch_normalization_2_6350*
Tin	
2*
Tout
2*1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_62992/
-batch_normalization_2/StatefulPartitionedCall�
conv4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0
conv4_6353
conv4_6355*
Tin
2*
Tout
2*1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_conv4_layer_call_and_return_conditional_losses_57712
conv4/StatefulPartitionedCall�
ma4/PartitionedCallPartitionedCall&conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*F
fAR?
=__inference_ma4_layer_call_and_return_conditional_losses_57872
ma4/PartitionedCall�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCallma4/PartitionedCall:output:0batch_normalization_3_6434batch_normalization_3_6436batch_normalization_3_6438batch_normalization_3_6440*
Tin	
2*
Tout
2*1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_63892/
-batch_normalization_3/StatefulPartitionedCall�
conv5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0
conv5_6443
conv5_6445*
Tin
2*
Tout
2*1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_conv5_layer_call_and_return_conditional_losses_59312
conv5/StatefulPartitionedCall�
ma5/PartitionedCallPartitionedCall&conv5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*F
fAR?
=__inference_ma5_layer_call_and_return_conditional_losses_59472
ma5/PartitionedCall�
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCallma5/PartitionedCall:output:0batch_normalization_4_6524batch_normalization_4_6526batch_normalization_4_6528batch_normalization_4_6530*
Tin	
2*
Tout
2*1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_64792/
-batch_normalization_4/StatefulPartitionedCall�
dropout/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_65452!
dropout/StatefulPartitionedCall�
dense1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense1_6605dense1_6607*
Tin
2*
Tout
2*0
_output_shapes
:����������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_dense1_layer_call_and_return_conditional_losses_65942 
dense1/StatefulPartitionedCall�
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*0
_output_shapes
:����������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_66222#
!dropout_1/StatefulPartitionedCall�
dense2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense2_6681dense2_6683*
Tin
2*
Tout
2*0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_dense2_layer_call_and_return_conditional_losses_66702 
dense2/StatefulPartitionedCall�
sigmoid/PartitionedCallPartitionedCall'dense2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_sigmoid_layer_call_and_return_conditional_losses_66912
sigmoid/PartitionedCall�
reshape/PartitionedCallPartitionedCall sigmoid/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_67122
reshape/PartitionedCall�
IdentityIdentity reshape/PartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv3/StatefulPartitionedCall^conv4/StatefulPartitionedCall^conv5/StatefulPartitionedCall^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:����������#::::::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2>
conv4/StatefulPartitionedCallconv4/StatefulPartitionedCall2>
conv5/StatefulPartitionedCallconv5/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:\ X
0
_output_shapes
:����������#
$
_user_specified_name
the_inputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: 
�
Y
=__inference_ma5_layer_call_and_return_conditional_losses_5947

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�

�
?__inference_conv1_layer_call_and_return_conditional_losses_5291

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:�*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,����������������������������2
Relu�
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������:::i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
)__inference_sequential_layer_call_fn_7873

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32*.
Tin'
%2#*
Tout
2*,
_output_shapes
:����������*D
_read_only_resource_inputs&
$"	
 !"*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_70762
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:����������#::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������#
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: 
�
�
M__inference_batch_normalization_layer_call_and_return_conditional_losses_5428

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������:::::j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
_
&__inference_dropout_layer_call_fn_8705

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_65452
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*0
_input_shapes
:�����������22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
_
A__inference_dropout_layer_call_and_return_conditional_losses_6550

inputs

identity_1d
IdentityIdentityinputs*
T0*1
_output_shapes
:�����������2

Identitys

Identity_1IdentityIdentity:output:0*
T0*1
_output_shapes
:�����������2

Identity_1"!

identity_1Identity_1:output:0*0
_input_shapes
:�����������:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
y
$__inference_conv3_layer_call_fn_5621

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_conv3_layer_call_and_return_conditional_losses_56112
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,����������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�$
�
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8078

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��#AssignMovingAvg/AssignSubVariableOp�%AssignMovingAvg_1/AssignSubVariableOpu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�$
�
M__inference_batch_normalization_layer_call_and_return_conditional_losses_7916

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��#AssignMovingAvg/AssignSubVariableOp�%AssignMovingAvg_1/AssignSubVariableOpu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:�����������:�:�:�:�:*
epsilon%o�:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:�����������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
]
A__inference_sigmoid_layer_call_and_return_conditional_losses_6691

inputs
identity`
SigmoidSigmoidinputs*
T0*0
_output_shapes
:����������2	
Sigmoidh
IdentityIdentitySigmoid:y:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
B
&__inference_reshape_layer_call_fn_8844

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_67122
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
y
$__inference_conv4_layer_call_fn_5781

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_conv4_layer_call_and_return_conditional_losses_57712
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,����������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�$
�
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8153

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��#AssignMovingAvg/AssignSubVariableOp�%AssignMovingAvg_1/AssignSubVariableOpu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:�����������:�:�:�:�:*
epsilon%o�:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:�����������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
��
�
D__inference_sequential_layer_call_and_return_conditional_losses_7727

inputs(
$conv1_conv2d_readvariableop_resource)
%conv1_biasadd_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource@
<batch_normalization_fusedbatchnormv3_readvariableop_resourceB
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource(
$conv2_conv2d_readvariableop_resource)
%conv2_biasadd_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceB
>batch_normalization_1_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource(
$conv3_conv2d_readvariableop_resource)
%conv3_biasadd_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resourceB
>batch_normalization_2_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource(
$conv4_conv2d_readvariableop_resource)
%conv4_biasadd_readvariableop_resource1
-batch_normalization_3_readvariableop_resource3
/batch_normalization_3_readvariableop_1_resourceB
>batch_normalization_3_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource(
$conv5_conv2d_readvariableop_resource)
%conv5_biasadd_readvariableop_resource1
-batch_normalization_4_readvariableop_resource3
/batch_normalization_4_readvariableop_1_resourceB
>batch_normalization_4_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource,
(dense1_tensordot_readvariableop_resource*
&dense1_biasadd_readvariableop_resource,
(dense2_tensordot_readvariableop_resource*
&dense2_biasadd_readvariableop_resource
identity��
conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*'
_output_shapes
:�*
dtype02
conv1/Conv2D/ReadVariableOp�
conv1/Conv2DConv2Dinputs#conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������#�*
paddingSAME*
strides
2
conv1/Conv2D�
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
conv1/BiasAdd/ReadVariableOp�
conv1/BiasAddBiasAddconv1/Conv2D:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������#�2
conv1/BiasAddt

conv1/ReluReluconv1/BiasAdd:output:0*
T0*1
_output_shapes
:����������#�2

conv1/Relu�
max1/MaxPoolMaxPoolconv1/Relu:activations:0*1
_output_shapes
:�����������*
ksize
*
paddingVALID*
strides
2
max1/MaxPool�
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes	
:�*
dtype02$
"batch_normalization/ReadVariableOp�
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes	
:�*
dtype02&
$batch_normalization/ReadVariableOp_1�
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp�
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1�
$batch_normalization/FusedBatchNormV3FusedBatchNormV3max1/MaxPool:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:�����������:�:�:�:�:*
epsilon%o�:*
is_training( 2&
$batch_normalization/FusedBatchNormV3�
conv2/Conv2D/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
conv2/Conv2D/ReadVariableOp�
conv2/Conv2DConv2D(batch_normalization/FusedBatchNormV3:y:0#conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
2
conv2/Conv2D�
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
conv2/BiasAdd/ReadVariableOp�
conv2/BiasAddBiasAddconv2/Conv2D:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2
conv2/BiasAddt

conv2/ReluReluconv2/BiasAdd:output:0*
T0*1
_output_shapes
:�����������2

conv2/Relu�
max2/MaxPoolMaxPoolconv2/Relu:activations:0*1
_output_shapes
:�����������*
ksize
*
paddingVALID*
strides
2
max2/MaxPool�
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes	
:�*
dtype02&
$batch_normalization_1/ReadVariableOp�
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes	
:�*
dtype02(
&batch_normalization_1/ReadVariableOp_1�
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3max2/MaxPool:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:�����������:�:�:�:�:*
epsilon%o�:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3�
conv3/Conv2D/ReadVariableOpReadVariableOp$conv3_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
conv3/Conv2D/ReadVariableOp�
conv3/Conv2DConv2D*batch_normalization_1/FusedBatchNormV3:y:0#conv3/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
2
conv3/Conv2D�
conv3/BiasAdd/ReadVariableOpReadVariableOp%conv3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
conv3/BiasAdd/ReadVariableOp�
conv3/BiasAddBiasAddconv3/Conv2D:output:0$conv3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2
conv3/BiasAddt

conv3/ReluReluconv3/BiasAdd:output:0*
T0*1
_output_shapes
:�����������2

conv3/Relu�
ma3/MaxPoolMaxPoolconv3/Relu:activations:0*1
_output_shapes
:�����������*
ksize
*
paddingVALID*
strides
2
ma3/MaxPool�
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes	
:�*
dtype02&
$batch_normalization_2/ReadVariableOp�
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:�*
dtype02(
&batch_normalization_2/ReadVariableOp_1�
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3ma3/MaxPool:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:�����������:�:�:�:�:*
epsilon%o�:*
is_training( 2(
&batch_normalization_2/FusedBatchNormV3�
conv4/Conv2D/ReadVariableOpReadVariableOp$conv4_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
conv4/Conv2D/ReadVariableOp�
conv4/Conv2DConv2D*batch_normalization_2/FusedBatchNormV3:y:0#conv4/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
2
conv4/Conv2D�
conv4/BiasAdd/ReadVariableOpReadVariableOp%conv4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
conv4/BiasAdd/ReadVariableOp�
conv4/BiasAddBiasAddconv4/Conv2D:output:0$conv4/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2
conv4/BiasAddt

conv4/ReluReluconv4/BiasAdd:output:0*
T0*1
_output_shapes
:�����������2

conv4/Relu�
ma4/MaxPoolMaxPoolconv4/Relu:activations:0*1
_output_shapes
:�����������*
ksize
*
paddingVALID*
strides
2
ma4/MaxPool�
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes	
:�*
dtype02&
$batch_normalization_3/ReadVariableOp�
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02(
&batch_normalization_3/ReadVariableOp_1�
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3ma4/MaxPool:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:�����������:�:�:�:�:*
epsilon%o�:*
is_training( 2(
&batch_normalization_3/FusedBatchNormV3�
conv5/Conv2D/ReadVariableOpReadVariableOp$conv5_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
conv5/Conv2D/ReadVariableOp�
conv5/Conv2DConv2D*batch_normalization_3/FusedBatchNormV3:y:0#conv5/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
2
conv5/Conv2D�
conv5/BiasAdd/ReadVariableOpReadVariableOp%conv5_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
conv5/BiasAdd/ReadVariableOp�
conv5/BiasAddBiasAddconv5/Conv2D:output:0$conv5/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2
conv5/BiasAddt

conv5/ReluReluconv5/BiasAdd:output:0*
T0*1
_output_shapes
:�����������2

conv5/Relu�
ma5/MaxPoolMaxPoolconv5/Relu:activations:0*1
_output_shapes
:�����������*
ksize
*
paddingVALID*
strides
2
ma5/MaxPool�
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes	
:�*
dtype02&
$batch_normalization_4/ReadVariableOp�
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:�*
dtype02(
&batch_normalization_4/ReadVariableOp_1�
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3ma5/MaxPool:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:�����������:�:�:�:�:*
epsilon%o�:*
is_training( 2(
&batch_normalization_4/FusedBatchNormV3�
dropout/IdentityIdentity*batch_normalization_4/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:�����������2
dropout/Identity�
dense1/Tensordot/ReadVariableOpReadVariableOp(dense1_tensordot_readvariableop_resource*
_output_shapes
:	�@*
dtype02!
dense1/Tensordot/ReadVariableOpx
dense1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense1/Tensordot/axes�
dense1/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
dense1/Tensordot/freey
dense1/Tensordot/ShapeShapedropout/Identity:output:0*
T0*
_output_shapes
:2
dense1/Tensordot/Shape�
dense1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense1/Tensordot/GatherV2/axis�
dense1/Tensordot/GatherV2GatherV2dense1/Tensordot/Shape:output:0dense1/Tensordot/free:output:0'dense1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense1/Tensordot/GatherV2�
 dense1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense1/Tensordot/GatherV2_1/axis�
dense1/Tensordot/GatherV2_1GatherV2dense1/Tensordot/Shape:output:0dense1/Tensordot/axes:output:0)dense1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense1/Tensordot/GatherV2_1z
dense1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense1/Tensordot/Const�
dense1/Tensordot/ProdProd"dense1/Tensordot/GatherV2:output:0dense1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense1/Tensordot/Prod~
dense1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense1/Tensordot/Const_1�
dense1/Tensordot/Prod_1Prod$dense1/Tensordot/GatherV2_1:output:0!dense1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense1/Tensordot/Prod_1~
dense1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense1/Tensordot/concat/axis�
dense1/Tensordot/concatConcatV2dense1/Tensordot/free:output:0dense1/Tensordot/axes:output:0%dense1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense1/Tensordot/concat�
dense1/Tensordot/stackPackdense1/Tensordot/Prod:output:0 dense1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense1/Tensordot/stack�
dense1/Tensordot/transpose	Transposedropout/Identity:output:0 dense1/Tensordot/concat:output:0*
T0*1
_output_shapes
:�����������2
dense1/Tensordot/transpose�
dense1/Tensordot/ReshapeReshapedense1/Tensordot/transpose:y:0dense1/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
dense1/Tensordot/Reshape�
dense1/Tensordot/MatMulMatMul!dense1/Tensordot/Reshape:output:0'dense1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense1/Tensordot/MatMul~
dense1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense1/Tensordot/Const_2�
dense1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense1/Tensordot/concat_1/axis�
dense1/Tensordot/concat_1ConcatV2"dense1/Tensordot/GatherV2:output:0!dense1/Tensordot/Const_2:output:0'dense1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense1/Tensordot/concat_1�
dense1/TensordotReshape!dense1/Tensordot/MatMul:product:0"dense1/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:����������@2
dense1/Tensordot�
dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense1/BiasAdd/ReadVariableOp�
dense1/BiasAddBiasAdddense1/Tensordot:output:0%dense1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������@2
dense1/BiasAddv
dense1/ReluReludense1/BiasAdd:output:0*
T0*0
_output_shapes
:����������@2
dense1/Relu�
dropout_1/IdentityIdentitydense1/Relu:activations:0*
T0*0
_output_shapes
:����������@2
dropout_1/Identity�
dense2/Tensordot/ReadVariableOpReadVariableOp(dense2_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype02!
dense2/Tensordot/ReadVariableOpx
dense2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense2/Tensordot/axes�
dense2/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
dense2/Tensordot/free{
dense2/Tensordot/ShapeShapedropout_1/Identity:output:0*
T0*
_output_shapes
:2
dense2/Tensordot/Shape�
dense2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense2/Tensordot/GatherV2/axis�
dense2/Tensordot/GatherV2GatherV2dense2/Tensordot/Shape:output:0dense2/Tensordot/free:output:0'dense2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense2/Tensordot/GatherV2�
 dense2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense2/Tensordot/GatherV2_1/axis�
dense2/Tensordot/GatherV2_1GatherV2dense2/Tensordot/Shape:output:0dense2/Tensordot/axes:output:0)dense2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense2/Tensordot/GatherV2_1z
dense2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense2/Tensordot/Const�
dense2/Tensordot/ProdProd"dense2/Tensordot/GatherV2:output:0dense2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense2/Tensordot/Prod~
dense2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense2/Tensordot/Const_1�
dense2/Tensordot/Prod_1Prod$dense2/Tensordot/GatherV2_1:output:0!dense2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense2/Tensordot/Prod_1~
dense2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense2/Tensordot/concat/axis�
dense2/Tensordot/concatConcatV2dense2/Tensordot/free:output:0dense2/Tensordot/axes:output:0%dense2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense2/Tensordot/concat�
dense2/Tensordot/stackPackdense2/Tensordot/Prod:output:0 dense2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense2/Tensordot/stack�
dense2/Tensordot/transpose	Transposedropout_1/Identity:output:0 dense2/Tensordot/concat:output:0*
T0*0
_output_shapes
:����������@2
dense2/Tensordot/transpose�
dense2/Tensordot/ReshapeReshapedense2/Tensordot/transpose:y:0dense2/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
dense2/Tensordot/Reshape�
dense2/Tensordot/MatMulMatMul!dense2/Tensordot/Reshape:output:0'dense2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense2/Tensordot/MatMul~
dense2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense2/Tensordot/Const_2�
dense2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense2/Tensordot/concat_1/axis�
dense2/Tensordot/concat_1ConcatV2"dense2/Tensordot/GatherV2:output:0!dense2/Tensordot/Const_2:output:0'dense2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense2/Tensordot/concat_1�
dense2/TensordotReshape!dense2/Tensordot/MatMul:product:0"dense2/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:����������2
dense2/Tensordot�
dense2/BiasAdd/ReadVariableOpReadVariableOp&dense2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense2/BiasAdd/ReadVariableOp�
dense2/BiasAddBiasAdddense2/Tensordot:output:0%dense2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
dense2/BiasAdd�
sigmoid/SigmoidSigmoiddense2/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
sigmoid/Sigmoida
reshape/ShapeShapesigmoid/Sigmoid:y:0*
T0*
_output_shapes
:2
reshape/Shape�
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack�
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1�
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2�
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_sliceu
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :�2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2�
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape�
reshape/ReshapeReshapesigmoid/Sigmoid:y:0reshape/Reshape/shape:output:0*
T0*,
_output_shapes
:����������2
reshape/Reshapeq
IdentityIdentityreshape/Reshape:output:0*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:����������#:::::::::::::::::::::::::::::::::::X T
0
_output_shapes
:����������#
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: 
�$
�
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5717

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��#AssignMovingAvg/AssignSubVariableOp�%AssignMovingAvg_1/AssignSubVariableOpu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
b
C__inference_dropout_1_layer_call_and_return_conditional_losses_6622

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:����������@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:����������@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:����������@2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:����������@2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:����������@2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������@:X T
0
_output_shapes
:����������@
 
_user_specified_nameinputs
�
�
4__inference_batch_normalization_2_layer_call_fn_8346

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_57172
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
4__inference_batch_normalization_3_layer_call_fn_8508

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_63892
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:�����������::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
@__inference_dense2_layer_call_and_return_conditional_losses_8807

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesu
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*0
_output_shapes
:����������@2
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*0
_output_shapes
:����������2
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������@:::X T
0
_output_shapes
:����������@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
]
A__inference_reshape_layer_call_and_return_conditional_losses_6712

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicee
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :�2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapet
ReshapeReshapeinputsReshape/shape:output:0*
T0*,
_output_shapes
:����������2	
Reshapei
IdentityIdentityReshape:output:0*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
B
&__inference_dropout_layer_call_fn_8710

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_65502
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*0
_input_shapes
:�����������:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_5908

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������:::::j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_8420

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������:::::j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
?
#__inference_max1_layer_call_fn_5313

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_max1_layer_call_and_return_conditional_losses_53072
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
D
(__inference_dropout_1_layer_call_fn_8777

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*0
_output_shapes
:����������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_66272
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������@:X T
0
_output_shapes
:����������@
 
_user_specified_nameinputs
�
�
2__inference_batch_normalization_layer_call_fn_8022

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_53972
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
2__inference_batch_normalization_layer_call_fn_7960

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*1
_output_shapes
:�����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_61372
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:�����������::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�c
�
D__inference_sequential_layer_call_and_return_conditional_losses_7076

inputs

conv1_6986

conv1_6988
batch_normalization_6992
batch_normalization_6994
batch_normalization_6996
batch_normalization_6998

conv2_7001

conv2_7003
batch_normalization_1_7007
batch_normalization_1_7009
batch_normalization_1_7011
batch_normalization_1_7013

conv3_7016

conv3_7018
batch_normalization_2_7022
batch_normalization_2_7024
batch_normalization_2_7026
batch_normalization_2_7028

conv4_7031

conv4_7033
batch_normalization_3_7037
batch_normalization_3_7039
batch_normalization_3_7041
batch_normalization_3_7043

conv5_7046

conv5_7048
batch_normalization_4_7052
batch_normalization_4_7054
batch_normalization_4_7056
batch_normalization_4_7058
dense1_7062
dense1_7064
dense2_7068
dense2_7070
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�-batch_normalization_3/StatefulPartitionedCall�-batch_normalization_4/StatefulPartitionedCall�conv1/StatefulPartitionedCall�conv2/StatefulPartitionedCall�conv3/StatefulPartitionedCall�conv4/StatefulPartitionedCall�conv5/StatefulPartitionedCall�dense1/StatefulPartitionedCall�dense2/StatefulPartitionedCall�
conv1/StatefulPartitionedCallStatefulPartitionedCallinputs
conv1_6986
conv1_6988*
Tin
2*
Tout
2*1
_output_shapes
:����������#�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_conv1_layer_call_and_return_conditional_losses_52912
conv1/StatefulPartitionedCall�
max1/PartitionedCallPartitionedCall&conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_max1_layer_call_and_return_conditional_losses_53072
max1/PartitionedCall�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCallmax1/PartitionedCall:output:0batch_normalization_6992batch_normalization_6994batch_normalization_6996batch_normalization_6998*
Tin	
2*
Tout
2*1
_output_shapes
:�����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_61372-
+batch_normalization/StatefulPartitionedCall�
conv2/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0
conv2_7001
conv2_7003*
Tin
2*
Tout
2*1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_conv2_layer_call_and_return_conditional_losses_54512
conv2/StatefulPartitionedCall�
max2/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_max2_layer_call_and_return_conditional_losses_54672
max2/PartitionedCall�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCallmax2/PartitionedCall:output:0batch_normalization_1_7007batch_normalization_1_7009batch_normalization_1_7011batch_normalization_1_7013*
Tin	
2*
Tout
2*1
_output_shapes
:�����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_62272/
-batch_normalization_1/StatefulPartitionedCall�
conv3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0
conv3_7016
conv3_7018*
Tin
2*
Tout
2*1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_conv3_layer_call_and_return_conditional_losses_56112
conv3/StatefulPartitionedCall�
ma3/PartitionedCallPartitionedCall&conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*F
fAR?
=__inference_ma3_layer_call_and_return_conditional_losses_56272
ma3/PartitionedCall�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCallma3/PartitionedCall:output:0batch_normalization_2_7022batch_normalization_2_7024batch_normalization_2_7026batch_normalization_2_7028*
Tin	
2*
Tout
2*1
_output_shapes
:�����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_63172/
-batch_normalization_2/StatefulPartitionedCall�
conv4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0
conv4_7031
conv4_7033*
Tin
2*
Tout
2*1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_conv4_layer_call_and_return_conditional_losses_57712
conv4/StatefulPartitionedCall�
ma4/PartitionedCallPartitionedCall&conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*F
fAR?
=__inference_ma4_layer_call_and_return_conditional_losses_57872
ma4/PartitionedCall�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCallma4/PartitionedCall:output:0batch_normalization_3_7037batch_normalization_3_7039batch_normalization_3_7041batch_normalization_3_7043*
Tin	
2*
Tout
2*1
_output_shapes
:�����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_64072/
-batch_normalization_3/StatefulPartitionedCall�
conv5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0
conv5_7046
conv5_7048*
Tin
2*
Tout
2*1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_conv5_layer_call_and_return_conditional_losses_59312
conv5/StatefulPartitionedCall�
ma5/PartitionedCallPartitionedCall&conv5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*F
fAR?
=__inference_ma5_layer_call_and_return_conditional_losses_59472
ma5/PartitionedCall�
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCallma5/PartitionedCall:output:0batch_normalization_4_7052batch_normalization_4_7054batch_normalization_4_7056batch_normalization_4_7058*
Tin	
2*
Tout
2*1
_output_shapes
:�����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_64972/
-batch_normalization_4/StatefulPartitionedCall�
dropout/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_65502
dropout/PartitionedCall�
dense1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense1_7062dense1_7064*
Tin
2*
Tout
2*0
_output_shapes
:����������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_dense1_layer_call_and_return_conditional_losses_65942 
dense1/StatefulPartitionedCall�
dropout_1/PartitionedCallPartitionedCall'dense1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:����������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_66272
dropout_1/PartitionedCall�
dense2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense2_7068dense2_7070*
Tin
2*
Tout
2*0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_dense2_layer_call_and_return_conditional_losses_66702 
dense2/StatefulPartitionedCall�
sigmoid/PartitionedCallPartitionedCall'dense2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_sigmoid_layer_call_and_return_conditional_losses_66912
sigmoid/PartitionedCall�
reshape/PartitionedCallPartitionedCall sigmoid/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_67122
reshape/PartitionedCall�
IdentityIdentity reshape/PartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv3/StatefulPartitionedCall^conv4/StatefulPartitionedCall^conv5/StatefulPartitionedCall^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:����������#::::::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2>
conv4/StatefulPartitionedCallconv4/StatefulPartitionedCall2>
conv5/StatefulPartitionedCallconv5/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall:X T
0
_output_shapes
:����������#
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: 
�$
�
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_8402

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��#AssignMovingAvg/AssignSubVariableOp�%AssignMovingAvg_1/AssignSubVariableOpu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
4__inference_batch_normalization_4_layer_call_fn_8683

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*1
_output_shapes
:�����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_64972
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:�����������::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
��
�(
__inference__traced_save_9144
file_prefix+
'savev2_conv1_kernel_read_readvariableop)
%savev2_conv1_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop+
'savev2_conv2_kernel_read_readvariableop)
%savev2_conv2_bias_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop+
'savev2_conv3_kernel_read_readvariableop)
%savev2_conv3_bias_read_readvariableop:
6savev2_batch_normalization_2_gamma_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableop+
'savev2_conv4_kernel_read_readvariableop)
%savev2_conv4_bias_read_readvariableop:
6savev2_batch_normalization_3_gamma_read_readvariableop9
5savev2_batch_normalization_3_beta_read_readvariableop@
<savev2_batch_normalization_3_moving_mean_read_readvariableopD
@savev2_batch_normalization_3_moving_variance_read_readvariableop+
'savev2_conv5_kernel_read_readvariableop)
%savev2_conv5_bias_read_readvariableop:
6savev2_batch_normalization_4_gamma_read_readvariableop9
5savev2_batch_normalization_4_beta_read_readvariableop@
<savev2_batch_normalization_4_moving_mean_read_readvariableopD
@savev2_batch_normalization_4_moving_variance_read_readvariableop,
(savev2_dense1_kernel_read_readvariableop*
&savev2_dense1_bias_read_readvariableop,
(savev2_dense2_kernel_read_readvariableop*
&savev2_dense2_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop2
.savev2_adam_conv1_kernel_m_read_readvariableop0
,savev2_adam_conv1_bias_m_read_readvariableop?
;savev2_adam_batch_normalization_gamma_m_read_readvariableop>
:savev2_adam_batch_normalization_beta_m_read_readvariableop2
.savev2_adam_conv2_kernel_m_read_readvariableop0
,savev2_adam_conv2_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_1_beta_m_read_readvariableop2
.savev2_adam_conv3_kernel_m_read_readvariableop0
,savev2_adam_conv3_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_2_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_2_beta_m_read_readvariableop2
.savev2_adam_conv4_kernel_m_read_readvariableop0
,savev2_adam_conv4_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_3_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_3_beta_m_read_readvariableop2
.savev2_adam_conv5_kernel_m_read_readvariableop0
,savev2_adam_conv5_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_4_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_4_beta_m_read_readvariableop3
/savev2_adam_dense1_kernel_m_read_readvariableop1
-savev2_adam_dense1_bias_m_read_readvariableop3
/savev2_adam_dense2_kernel_m_read_readvariableop1
-savev2_adam_dense2_bias_m_read_readvariableop2
.savev2_adam_conv1_kernel_v_read_readvariableop0
,savev2_adam_conv1_bias_v_read_readvariableop?
;savev2_adam_batch_normalization_gamma_v_read_readvariableop>
:savev2_adam_batch_normalization_beta_v_read_readvariableop2
.savev2_adam_conv2_kernel_v_read_readvariableop0
,savev2_adam_conv2_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_1_beta_v_read_readvariableop2
.savev2_adam_conv3_kernel_v_read_readvariableop0
,savev2_adam_conv3_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_2_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_2_beta_v_read_readvariableop2
.savev2_adam_conv4_kernel_v_read_readvariableop0
,savev2_adam_conv4_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_3_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_3_beta_v_read_readvariableop2
.savev2_adam_conv5_kernel_v_read_readvariableop0
,savev2_adam_conv5_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_4_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_4_beta_v_read_readvariableop3
/savev2_adam_dense1_kernel_v_read_readvariableop1
-savev2_adam_dense1_bias_v_read_readvariableop3
/savev2_adam_dense2_kernel_v_read_readvariableop1
-savev2_adam_dense2_bias_v_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const�
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_81a105a38d3141cc962d7dd0b2cbfbe7/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�2
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:[*
dtype0*�1
value�1B�1[B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:[*
dtype0*�
value�B�[B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�&
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_conv1_kernel_read_readvariableop%savev2_conv1_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop'savev2_conv2_kernel_read_readvariableop%savev2_conv2_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop'savev2_conv3_kernel_read_readvariableop%savev2_conv3_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop'savev2_conv4_kernel_read_readvariableop%savev2_conv4_bias_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop'savev2_conv5_kernel_read_readvariableop%savev2_conv5_bias_read_readvariableop6savev2_batch_normalization_4_gamma_read_readvariableop5savev2_batch_normalization_4_beta_read_readvariableop<savev2_batch_normalization_4_moving_mean_read_readvariableop@savev2_batch_normalization_4_moving_variance_read_readvariableop(savev2_dense1_kernel_read_readvariableop&savev2_dense1_bias_read_readvariableop(savev2_dense2_kernel_read_readvariableop&savev2_dense2_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop.savev2_adam_conv1_kernel_m_read_readvariableop,savev2_adam_conv1_bias_m_read_readvariableop;savev2_adam_batch_normalization_gamma_m_read_readvariableop:savev2_adam_batch_normalization_beta_m_read_readvariableop.savev2_adam_conv2_kernel_m_read_readvariableop,savev2_adam_conv2_bias_m_read_readvariableop=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop<savev2_adam_batch_normalization_1_beta_m_read_readvariableop.savev2_adam_conv3_kernel_m_read_readvariableop,savev2_adam_conv3_bias_m_read_readvariableop=savev2_adam_batch_normalization_2_gamma_m_read_readvariableop<savev2_adam_batch_normalization_2_beta_m_read_readvariableop.savev2_adam_conv4_kernel_m_read_readvariableop,savev2_adam_conv4_bias_m_read_readvariableop=savev2_adam_batch_normalization_3_gamma_m_read_readvariableop<savev2_adam_batch_normalization_3_beta_m_read_readvariableop.savev2_adam_conv5_kernel_m_read_readvariableop,savev2_adam_conv5_bias_m_read_readvariableop=savev2_adam_batch_normalization_4_gamma_m_read_readvariableop<savev2_adam_batch_normalization_4_beta_m_read_readvariableop/savev2_adam_dense1_kernel_m_read_readvariableop-savev2_adam_dense1_bias_m_read_readvariableop/savev2_adam_dense2_kernel_m_read_readvariableop-savev2_adam_dense2_bias_m_read_readvariableop.savev2_adam_conv1_kernel_v_read_readvariableop,savev2_adam_conv1_bias_v_read_readvariableop;savev2_adam_batch_normalization_gamma_v_read_readvariableop:savev2_adam_batch_normalization_beta_v_read_readvariableop.savev2_adam_conv2_kernel_v_read_readvariableop,savev2_adam_conv2_bias_v_read_readvariableop=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop<savev2_adam_batch_normalization_1_beta_v_read_readvariableop.savev2_adam_conv3_kernel_v_read_readvariableop,savev2_adam_conv3_bias_v_read_readvariableop=savev2_adam_batch_normalization_2_gamma_v_read_readvariableop<savev2_adam_batch_normalization_2_beta_v_read_readvariableop.savev2_adam_conv4_kernel_v_read_readvariableop,savev2_adam_conv4_bias_v_read_readvariableop=savev2_adam_batch_normalization_3_gamma_v_read_readvariableop<savev2_adam_batch_normalization_3_beta_v_read_readvariableop.savev2_adam_conv5_kernel_v_read_readvariableop,savev2_adam_conv5_bias_v_read_readvariableop=savev2_adam_batch_normalization_4_gamma_v_read_readvariableop<savev2_adam_batch_normalization_4_beta_v_read_readvariableop/savev2_adam_dense1_kernel_v_read_readvariableop-savev2_adam_dense1_bias_v_read_readvariableop/savev2_adam_dense2_kernel_v_read_readvariableop-savev2_adam_dense2_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *i
dtypes_
]2[	2
SaveV2�
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard�
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1�
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names�
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices�
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity�

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: :�:�:�:�:�:�:��:�:�:�:�:�:��:�:�:�:�:�:��:�:�:�:�:�:��:�:�:�:�:�:	�@:@:@:: : : : : : : : : :�:�:�:�:��:�:�:�:��:�:�:�:��:�:�:�:��:�:�:�:	�@:@:@::�:�:�:�:��:�:�:�:��:�:�:�:��:�:�:�:��:�:�:�:	�@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:-)
'
_output_shapes
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:.*
(
_output_shapes
:��:!

_output_shapes	
:�:!	

_output_shapes	
:�:!


_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:.*
(
_output_shapes
:��:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:.*
(
_output_shapes
:��:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:.*
(
_output_shapes
:��:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:%!

_output_shapes
:	�@:  

_output_shapes
:@:$! 

_output_shapes

:@: "

_output_shapes
::#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :-,)
'
_output_shapes
:�:!-

_output_shapes	
:�:!.

_output_shapes	
:�:!/

_output_shapes	
:�:.0*
(
_output_shapes
:��:!1

_output_shapes	
:�:!2

_output_shapes	
:�:!3

_output_shapes	
:�:.4*
(
_output_shapes
:��:!5

_output_shapes	
:�:!6

_output_shapes	
:�:!7

_output_shapes	
:�:.8*
(
_output_shapes
:��:!9

_output_shapes	
:�:!:

_output_shapes	
:�:!;

_output_shapes	
:�:.<*
(
_output_shapes
:��:!=

_output_shapes	
:�:!>

_output_shapes	
:�:!?

_output_shapes	
:�:%@!

_output_shapes
:	�@: A

_output_shapes
:@:$B 

_output_shapes

:@: C

_output_shapes
::-D)
'
_output_shapes
:�:!E

_output_shapes	
:�:!F

_output_shapes	
:�:!G

_output_shapes	
:�:.H*
(
_output_shapes
:��:!I

_output_shapes	
:�:!J

_output_shapes	
:�:!K

_output_shapes	
:�:.L*
(
_output_shapes
:��:!M

_output_shapes	
:�:!N

_output_shapes	
:�:!O

_output_shapes	
:�:.P*
(
_output_shapes
:��:!Q

_output_shapes	
:�:!R

_output_shapes	
:�:!S

_output_shapes	
:�:.T*
(
_output_shapes
:��:!U

_output_shapes	
:�:!V

_output_shapes	
:�:!W

_output_shapes	
:�:%X!

_output_shapes
:	�@: Y

_output_shapes
:@:$Z 

_output_shapes

:@: [

_output_shapes
::\

_output_shapes
: 
�
�
@__inference_dense1_layer_call_and_return_conditional_losses_6594

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	�@*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesu
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*1
_output_shapes
:�����������2
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*0
_output_shapes
:����������@2
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������@2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������@2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:�����������:::Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_8258

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:�����������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3r
IdentityIdentityFusedBatchNormV3:y:0*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:�����������:::::Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
Y
=__inference_ma3_layer_call_and_return_conditional_losses_5627

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_6227

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:�����������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3r
IdentityIdentityFusedBatchNormV3:y:0*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:�����������:::::Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�$
�
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_5557

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��#AssignMovingAvg/AssignSubVariableOp�%AssignMovingAvg_1/AssignSubVariableOpu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
4__inference_batch_normalization_2_layer_call_fn_8359

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_57482
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
y
$__inference_conv1_layer_call_fn_5301

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_conv1_layer_call_and_return_conditional_losses_52912
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
M__inference_batch_normalization_layer_call_and_return_conditional_losses_7934

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:�����������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3r
IdentityIdentityFusedBatchNormV3:y:0*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:�����������:::::Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8171

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:�����������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3r
IdentityIdentityFusedBatchNormV3:y:0*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:�����������:::::Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
z
%__inference_dense1_layer_call_fn_8750

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*0
_output_shapes
:����������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_dense1_layer_call_and_return_conditional_losses_65942
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:�����������::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
4__inference_batch_normalization_1_layer_call_fn_8109

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_55572
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
Y
=__inference_ma4_layer_call_and_return_conditional_losses_5787

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
)__inference_sequential_layer_call_fn_7147

the_inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall
the_inputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32*.
Tin'
%2#*
Tout
2*,
_output_shapes
:����������*D
_read_only_resource_inputs&
$"	
 !"*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_70762
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:����������#::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
0
_output_shapes
:����������#
$
_user_specified_name
the_inputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: 
�
�
4__inference_batch_normalization_3_layer_call_fn_8521

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*1
_output_shapes
:�����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_64072
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:�����������::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
)__inference_sequential_layer_call_fn_6981

the_inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall
the_inputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32*.
Tin'
%2#*
Tout
2*,
_output_shapes
:����������*:
_read_only_resource_inputs
	
 !"*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_69102
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:����������#::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
0
_output_shapes
:����������#
$
_user_specified_name
the_inputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: 
�$
�
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_6479

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��#AssignMovingAvg/AssignSubVariableOp�%AssignMovingAvg_1/AssignSubVariableOpu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:�����������:�:�:�:�:*
epsilon%o�:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:�����������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
4__inference_batch_normalization_1_layer_call_fn_8184

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_62092
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:�����������::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
>
"__inference_ma4_layer_call_fn_5793

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*F
fAR?
=__inference_ma4_layer_call_and_return_conditional_losses_57872
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�$
�
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_6389

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��#AssignMovingAvg/AssignSubVariableOp�%AssignMovingAvg_1/AssignSubVariableOpu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:�����������:�:�:�:�:*
epsilon%o�:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:�����������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_8582

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������:::::j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
4__inference_batch_normalization_4_layer_call_fn_8595

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_60372
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
Z
>__inference_max2_layer_call_and_return_conditional_losses_5467

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_8333

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������:::::j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
4__inference_batch_normalization_1_layer_call_fn_8197

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*1
_output_shapes
:�����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_62272
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:�����������::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
˄
�
D__inference_sequential_layer_call_and_return_conditional_losses_7548

inputs(
$conv1_conv2d_readvariableop_resource)
%conv1_biasadd_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource@
<batch_normalization_fusedbatchnormv3_readvariableop_resourceB
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource(
$conv2_conv2d_readvariableop_resource)
%conv2_biasadd_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceB
>batch_normalization_1_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource(
$conv3_conv2d_readvariableop_resource)
%conv3_biasadd_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resourceB
>batch_normalization_2_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource(
$conv4_conv2d_readvariableop_resource)
%conv4_biasadd_readvariableop_resource1
-batch_normalization_3_readvariableop_resource3
/batch_normalization_3_readvariableop_1_resourceB
>batch_normalization_3_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource(
$conv5_conv2d_readvariableop_resource)
%conv5_biasadd_readvariableop_resource1
-batch_normalization_4_readvariableop_resource3
/batch_normalization_4_readvariableop_1_resourceB
>batch_normalization_4_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource,
(dense1_tensordot_readvariableop_resource*
&dense1_biasadd_readvariableop_resource,
(dense2_tensordot_readvariableop_resource*
&dense2_biasadd_readvariableop_resource
identity��7batch_normalization/AssignMovingAvg/AssignSubVariableOp�9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp�9batch_normalization_1/AssignMovingAvg/AssignSubVariableOp�;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp�9batch_normalization_2/AssignMovingAvg/AssignSubVariableOp�;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp�9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp�;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp�9batch_normalization_4/AssignMovingAvg/AssignSubVariableOp�;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp�
conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*'
_output_shapes
:�*
dtype02
conv1/Conv2D/ReadVariableOp�
conv1/Conv2DConv2Dinputs#conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������#�*
paddingSAME*
strides
2
conv1/Conv2D�
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
conv1/BiasAdd/ReadVariableOp�
conv1/BiasAddBiasAddconv1/Conv2D:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������#�2
conv1/BiasAddt

conv1/ReluReluconv1/BiasAdd:output:0*
T0*1
_output_shapes
:����������#�2

conv1/Relu�
max1/MaxPoolMaxPoolconv1/Relu:activations:0*1
_output_shapes
:�����������*
ksize
*
paddingVALID*
strides
2
max1/MaxPool�
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes	
:�*
dtype02$
"batch_normalization/ReadVariableOp�
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes	
:�*
dtype02&
$batch_normalization/ReadVariableOp_1�
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp�
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1�
$batch_normalization/FusedBatchNormV3FusedBatchNormV3max1/MaxPool:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:�����������:�:�:�:�:*
epsilon%o�:2&
$batch_normalization/FusedBatchNormV3{
batch_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
batch_normalization/Const�
)batch_normalization/AssignMovingAvg/sub/xConst*O
_classE
CAloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2+
)batch_normalization/AssignMovingAvg/sub/x�
'batch_normalization/AssignMovingAvg/subSub2batch_normalization/AssignMovingAvg/sub/x:output:0"batch_normalization/Const:output:0*
T0*O
_classE
CAloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2)
'batch_normalization/AssignMovingAvg/sub�
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype024
2batch_normalization/AssignMovingAvg/ReadVariableOp�
)batch_normalization/AssignMovingAvg/sub_1Sub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:01batch_normalization/FusedBatchNormV3:batch_mean:0*
T0*O
_classE
CAloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2+
)batch_normalization/AssignMovingAvg/sub_1�
'batch_normalization/AssignMovingAvg/mulMul-batch_normalization/AssignMovingAvg/sub_1:z:0+batch_normalization/AssignMovingAvg/sub:z:0*
T0*O
_classE
CAloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2)
'batch_normalization/AssignMovingAvg/mul�
7batch_normalization/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp*O
_classE
CAloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype029
7batch_normalization/AssignMovingAvg/AssignSubVariableOp�
+batch_normalization/AssignMovingAvg_1/sub/xConst*Q
_classG
ECloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2-
+batch_normalization/AssignMovingAvg_1/sub/x�
)batch_normalization/AssignMovingAvg_1/subSub4batch_normalization/AssignMovingAvg_1/sub/x:output:0"batch_normalization/Const:output:0*
T0*Q
_classG
ECloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2+
)batch_normalization/AssignMovingAvg_1/sub�
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype026
4batch_normalization/AssignMovingAvg_1/ReadVariableOp�
+batch_normalization/AssignMovingAvg_1/sub_1Sub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:05batch_normalization/FusedBatchNormV3:batch_variance:0*
T0*Q
_classG
ECloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2-
+batch_normalization/AssignMovingAvg_1/sub_1�
)batch_normalization/AssignMovingAvg_1/mulMul/batch_normalization/AssignMovingAvg_1/sub_1:z:0-batch_normalization/AssignMovingAvg_1/sub:z:0*
T0*Q
_classG
ECloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2+
)batch_normalization/AssignMovingAvg_1/mul�
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*Q
_classG
ECloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02;
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp�
conv2/Conv2D/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
conv2/Conv2D/ReadVariableOp�
conv2/Conv2DConv2D(batch_normalization/FusedBatchNormV3:y:0#conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
2
conv2/Conv2D�
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
conv2/BiasAdd/ReadVariableOp�
conv2/BiasAddBiasAddconv2/Conv2D:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2
conv2/BiasAddt

conv2/ReluReluconv2/BiasAdd:output:0*
T0*1
_output_shapes
:�����������2

conv2/Relu�
max2/MaxPoolMaxPoolconv2/Relu:activations:0*1
_output_shapes
:�����������*
ksize
*
paddingVALID*
strides
2
max2/MaxPool�
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes	
:�*
dtype02&
$batch_normalization_1/ReadVariableOp�
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes	
:�*
dtype02(
&batch_normalization_1/ReadVariableOp_1�
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3max2/MaxPool:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:�����������:�:�:�:�:*
epsilon%o�:2(
&batch_normalization_1/FusedBatchNormV3
batch_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
batch_normalization_1/Const�
+batch_normalization_1/AssignMovingAvg/sub/xConst*Q
_classG
ECloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2-
+batch_normalization_1/AssignMovingAvg/sub/x�
)batch_normalization_1/AssignMovingAvg/subSub4batch_normalization_1/AssignMovingAvg/sub/x:output:0$batch_normalization_1/Const:output:0*
T0*Q
_classG
ECloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2+
)batch_normalization_1/AssignMovingAvg/sub�
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype026
4batch_normalization_1/AssignMovingAvg/ReadVariableOp�
+batch_normalization_1/AssignMovingAvg/sub_1Sub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:03batch_normalization_1/FusedBatchNormV3:batch_mean:0*
T0*Q
_classG
ECloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2-
+batch_normalization_1/AssignMovingAvg/sub_1�
)batch_normalization_1/AssignMovingAvg/mulMul/batch_normalization_1/AssignMovingAvg/sub_1:z:0-batch_normalization_1/AssignMovingAvg/sub:z:0*
T0*Q
_classG
ECloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2+
)batch_normalization_1/AssignMovingAvg/mul�
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource-batch_normalization_1/AssignMovingAvg/mul:z:05^batch_normalization_1/AssignMovingAvg/ReadVariableOp6^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02;
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOp�
-batch_normalization_1/AssignMovingAvg_1/sub/xConst*S
_classI
GEloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2/
-batch_normalization_1/AssignMovingAvg_1/sub/x�
+batch_normalization_1/AssignMovingAvg_1/subSub6batch_normalization_1/AssignMovingAvg_1/sub/x:output:0$batch_normalization_1/Const:output:0*
T0*S
_classI
GEloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2-
+batch_normalization_1/AssignMovingAvg_1/sub�
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype028
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp�
-batch_normalization_1/AssignMovingAvg_1/sub_1Sub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:07batch_normalization_1/FusedBatchNormV3:batch_variance:0*
T0*S
_classI
GEloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2/
-batch_normalization_1/AssignMovingAvg_1/sub_1�
+batch_normalization_1/AssignMovingAvg_1/mulMul1batch_normalization_1/AssignMovingAvg_1/sub_1:z:0/batch_normalization_1/AssignMovingAvg_1/sub:z:0*
T0*S
_classI
GEloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2-
+batch_normalization_1/AssignMovingAvg_1/mul�
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource/batch_normalization_1/AssignMovingAvg_1/mul:z:07^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02=
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp�
conv3/Conv2D/ReadVariableOpReadVariableOp$conv3_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
conv3/Conv2D/ReadVariableOp�
conv3/Conv2DConv2D*batch_normalization_1/FusedBatchNormV3:y:0#conv3/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
2
conv3/Conv2D�
conv3/BiasAdd/ReadVariableOpReadVariableOp%conv3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
conv3/BiasAdd/ReadVariableOp�
conv3/BiasAddBiasAddconv3/Conv2D:output:0$conv3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2
conv3/BiasAddt

conv3/ReluReluconv3/BiasAdd:output:0*
T0*1
_output_shapes
:�����������2

conv3/Relu�
ma3/MaxPoolMaxPoolconv3/Relu:activations:0*1
_output_shapes
:�����������*
ksize
*
paddingVALID*
strides
2
ma3/MaxPool�
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes	
:�*
dtype02&
$batch_normalization_2/ReadVariableOp�
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:�*
dtype02(
&batch_normalization_2/ReadVariableOp_1�
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3ma3/MaxPool:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:�����������:�:�:�:�:*
epsilon%o�:2(
&batch_normalization_2/FusedBatchNormV3
batch_normalization_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
batch_normalization_2/Const�
+batch_normalization_2/AssignMovingAvg/sub/xConst*Q
_classG
ECloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2-
+batch_normalization_2/AssignMovingAvg/sub/x�
)batch_normalization_2/AssignMovingAvg/subSub4batch_normalization_2/AssignMovingAvg/sub/x:output:0$batch_normalization_2/Const:output:0*
T0*Q
_classG
ECloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2+
)batch_normalization_2/AssignMovingAvg/sub�
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype026
4batch_normalization_2/AssignMovingAvg/ReadVariableOp�
+batch_normalization_2/AssignMovingAvg/sub_1Sub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:03batch_normalization_2/FusedBatchNormV3:batch_mean:0*
T0*Q
_classG
ECloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2-
+batch_normalization_2/AssignMovingAvg/sub_1�
)batch_normalization_2/AssignMovingAvg/mulMul/batch_normalization_2/AssignMovingAvg/sub_1:z:0-batch_normalization_2/AssignMovingAvg/sub:z:0*
T0*Q
_classG
ECloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2+
)batch_normalization_2/AssignMovingAvg/mul�
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource-batch_normalization_2/AssignMovingAvg/mul:z:05^batch_normalization_2/AssignMovingAvg/ReadVariableOp6^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02;
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOp�
-batch_normalization_2/AssignMovingAvg_1/sub/xConst*S
_classI
GEloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2/
-batch_normalization_2/AssignMovingAvg_1/sub/x�
+batch_normalization_2/AssignMovingAvg_1/subSub6batch_normalization_2/AssignMovingAvg_1/sub/x:output:0$batch_normalization_2/Const:output:0*
T0*S
_classI
GEloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2-
+batch_normalization_2/AssignMovingAvg_1/sub�
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype028
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp�
-batch_normalization_2/AssignMovingAvg_1/sub_1Sub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:07batch_normalization_2/FusedBatchNormV3:batch_variance:0*
T0*S
_classI
GEloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2/
-batch_normalization_2/AssignMovingAvg_1/sub_1�
+batch_normalization_2/AssignMovingAvg_1/mulMul1batch_normalization_2/AssignMovingAvg_1/sub_1:z:0/batch_normalization_2/AssignMovingAvg_1/sub:z:0*
T0*S
_classI
GEloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2-
+batch_normalization_2/AssignMovingAvg_1/mul�
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource/batch_normalization_2/AssignMovingAvg_1/mul:z:07^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02=
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp�
conv4/Conv2D/ReadVariableOpReadVariableOp$conv4_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
conv4/Conv2D/ReadVariableOp�
conv4/Conv2DConv2D*batch_normalization_2/FusedBatchNormV3:y:0#conv4/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
2
conv4/Conv2D�
conv4/BiasAdd/ReadVariableOpReadVariableOp%conv4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
conv4/BiasAdd/ReadVariableOp�
conv4/BiasAddBiasAddconv4/Conv2D:output:0$conv4/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2
conv4/BiasAddt

conv4/ReluReluconv4/BiasAdd:output:0*
T0*1
_output_shapes
:�����������2

conv4/Relu�
ma4/MaxPoolMaxPoolconv4/Relu:activations:0*1
_output_shapes
:�����������*
ksize
*
paddingVALID*
strides
2
ma4/MaxPool�
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes	
:�*
dtype02&
$batch_normalization_3/ReadVariableOp�
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02(
&batch_normalization_3/ReadVariableOp_1�
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3ma4/MaxPool:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:�����������:�:�:�:�:*
epsilon%o�:2(
&batch_normalization_3/FusedBatchNormV3
batch_normalization_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
batch_normalization_3/Const�
+batch_normalization_3/AssignMovingAvg/sub/xConst*Q
_classG
ECloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2-
+batch_normalization_3/AssignMovingAvg/sub/x�
)batch_normalization_3/AssignMovingAvg/subSub4batch_normalization_3/AssignMovingAvg/sub/x:output:0$batch_normalization_3/Const:output:0*
T0*Q
_classG
ECloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2+
)batch_normalization_3/AssignMovingAvg/sub�
4batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype026
4batch_normalization_3/AssignMovingAvg/ReadVariableOp�
+batch_normalization_3/AssignMovingAvg/sub_1Sub<batch_normalization_3/AssignMovingAvg/ReadVariableOp:value:03batch_normalization_3/FusedBatchNormV3:batch_mean:0*
T0*Q
_classG
ECloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2-
+batch_normalization_3/AssignMovingAvg/sub_1�
)batch_normalization_3/AssignMovingAvg/mulMul/batch_normalization_3/AssignMovingAvg/sub_1:z:0-batch_normalization_3/AssignMovingAvg/sub:z:0*
T0*Q
_classG
ECloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2+
)batch_normalization_3/AssignMovingAvg/mul�
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource-batch_normalization_3/AssignMovingAvg/mul:z:05^batch_normalization_3/AssignMovingAvg/ReadVariableOp6^batch_normalization_3/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02;
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp�
-batch_normalization_3/AssignMovingAvg_1/sub/xConst*S
_classI
GEloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2/
-batch_normalization_3/AssignMovingAvg_1/sub/x�
+batch_normalization_3/AssignMovingAvg_1/subSub6batch_normalization_3/AssignMovingAvg_1/sub/x:output:0$batch_normalization_3/Const:output:0*
T0*S
_classI
GEloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2-
+batch_normalization_3/AssignMovingAvg_1/sub�
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype028
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp�
-batch_normalization_3/AssignMovingAvg_1/sub_1Sub>batch_normalization_3/AssignMovingAvg_1/ReadVariableOp:value:07batch_normalization_3/FusedBatchNormV3:batch_variance:0*
T0*S
_classI
GEloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2/
-batch_normalization_3/AssignMovingAvg_1/sub_1�
+batch_normalization_3/AssignMovingAvg_1/mulMul1batch_normalization_3/AssignMovingAvg_1/sub_1:z:0/batch_normalization_3/AssignMovingAvg_1/sub:z:0*
T0*S
_classI
GEloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2-
+batch_normalization_3/AssignMovingAvg_1/mul�
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource/batch_normalization_3/AssignMovingAvg_1/mul:z:07^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02=
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp�
conv5/Conv2D/ReadVariableOpReadVariableOp$conv5_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
conv5/Conv2D/ReadVariableOp�
conv5/Conv2DConv2D*batch_normalization_3/FusedBatchNormV3:y:0#conv5/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
2
conv5/Conv2D�
conv5/BiasAdd/ReadVariableOpReadVariableOp%conv5_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
conv5/BiasAdd/ReadVariableOp�
conv5/BiasAddBiasAddconv5/Conv2D:output:0$conv5/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2
conv5/BiasAddt

conv5/ReluReluconv5/BiasAdd:output:0*
T0*1
_output_shapes
:�����������2

conv5/Relu�
ma5/MaxPoolMaxPoolconv5/Relu:activations:0*1
_output_shapes
:�����������*
ksize
*
paddingVALID*
strides
2
ma5/MaxPool�
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes	
:�*
dtype02&
$batch_normalization_4/ReadVariableOp�
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:�*
dtype02(
&batch_normalization_4/ReadVariableOp_1�
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3ma5/MaxPool:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:�����������:�:�:�:�:*
epsilon%o�:2(
&batch_normalization_4/FusedBatchNormV3
batch_normalization_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
batch_normalization_4/Const�
+batch_normalization_4/AssignMovingAvg/sub/xConst*Q
_classG
ECloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2-
+batch_normalization_4/AssignMovingAvg/sub/x�
)batch_normalization_4/AssignMovingAvg/subSub4batch_normalization_4/AssignMovingAvg/sub/x:output:0$batch_normalization_4/Const:output:0*
T0*Q
_classG
ECloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2+
)batch_normalization_4/AssignMovingAvg/sub�
4batch_normalization_4/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype026
4batch_normalization_4/AssignMovingAvg/ReadVariableOp�
+batch_normalization_4/AssignMovingAvg/sub_1Sub<batch_normalization_4/AssignMovingAvg/ReadVariableOp:value:03batch_normalization_4/FusedBatchNormV3:batch_mean:0*
T0*Q
_classG
ECloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2-
+batch_normalization_4/AssignMovingAvg/sub_1�
)batch_normalization_4/AssignMovingAvg/mulMul/batch_normalization_4/AssignMovingAvg/sub_1:z:0-batch_normalization_4/AssignMovingAvg/sub:z:0*
T0*Q
_classG
ECloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2+
)batch_normalization_4/AssignMovingAvg/mul�
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource-batch_normalization_4/AssignMovingAvg/mul:z:05^batch_normalization_4/AssignMovingAvg/ReadVariableOp6^batch_normalization_4/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02;
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOp�
-batch_normalization_4/AssignMovingAvg_1/sub/xConst*S
_classI
GEloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2/
-batch_normalization_4/AssignMovingAvg_1/sub/x�
+batch_normalization_4/AssignMovingAvg_1/subSub6batch_normalization_4/AssignMovingAvg_1/sub/x:output:0$batch_normalization_4/Const:output:0*
T0*S
_classI
GEloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2-
+batch_normalization_4/AssignMovingAvg_1/sub�
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype028
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp�
-batch_normalization_4/AssignMovingAvg_1/sub_1Sub>batch_normalization_4/AssignMovingAvg_1/ReadVariableOp:value:07batch_normalization_4/FusedBatchNormV3:batch_variance:0*
T0*S
_classI
GEloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2/
-batch_normalization_4/AssignMovingAvg_1/sub_1�
+batch_normalization_4/AssignMovingAvg_1/mulMul1batch_normalization_4/AssignMovingAvg_1/sub_1:z:0/batch_normalization_4/AssignMovingAvg_1/sub:z:0*
T0*S
_classI
GEloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2-
+batch_normalization_4/AssignMovingAvg_1/mul�
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource/batch_normalization_4/AssignMovingAvg_1/mul:z:07^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02=
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOps
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout/dropout/Const�
dropout/dropout/MulMul*batch_normalization_4/FusedBatchNormV3:y:0dropout/dropout/Const:output:0*
T0*1
_output_shapes
:�����������2
dropout/dropout/Mul�
dropout/dropout/ShapeShape*batch_normalization_4/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
dropout/dropout/Shape�
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*1
_output_shapes
:�����������*
dtype02.
,dropout/dropout/random_uniform/RandomUniform�
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2 
dropout/dropout/GreaterEqual/y�
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:�����������2
dropout/dropout/GreaterEqual�
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:�����������2
dropout/dropout/Cast�
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*1
_output_shapes
:�����������2
dropout/dropout/Mul_1�
dense1/Tensordot/ReadVariableOpReadVariableOp(dense1_tensordot_readvariableop_resource*
_output_shapes
:	�@*
dtype02!
dense1/Tensordot/ReadVariableOpx
dense1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense1/Tensordot/axes�
dense1/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
dense1/Tensordot/freey
dense1/Tensordot/ShapeShapedropout/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
dense1/Tensordot/Shape�
dense1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense1/Tensordot/GatherV2/axis�
dense1/Tensordot/GatherV2GatherV2dense1/Tensordot/Shape:output:0dense1/Tensordot/free:output:0'dense1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense1/Tensordot/GatherV2�
 dense1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense1/Tensordot/GatherV2_1/axis�
dense1/Tensordot/GatherV2_1GatherV2dense1/Tensordot/Shape:output:0dense1/Tensordot/axes:output:0)dense1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense1/Tensordot/GatherV2_1z
dense1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense1/Tensordot/Const�
dense1/Tensordot/ProdProd"dense1/Tensordot/GatherV2:output:0dense1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense1/Tensordot/Prod~
dense1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense1/Tensordot/Const_1�
dense1/Tensordot/Prod_1Prod$dense1/Tensordot/GatherV2_1:output:0!dense1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense1/Tensordot/Prod_1~
dense1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense1/Tensordot/concat/axis�
dense1/Tensordot/concatConcatV2dense1/Tensordot/free:output:0dense1/Tensordot/axes:output:0%dense1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense1/Tensordot/concat�
dense1/Tensordot/stackPackdense1/Tensordot/Prod:output:0 dense1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense1/Tensordot/stack�
dense1/Tensordot/transpose	Transposedropout/dropout/Mul_1:z:0 dense1/Tensordot/concat:output:0*
T0*1
_output_shapes
:�����������2
dense1/Tensordot/transpose�
dense1/Tensordot/ReshapeReshapedense1/Tensordot/transpose:y:0dense1/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
dense1/Tensordot/Reshape�
dense1/Tensordot/MatMulMatMul!dense1/Tensordot/Reshape:output:0'dense1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense1/Tensordot/MatMul~
dense1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense1/Tensordot/Const_2�
dense1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense1/Tensordot/concat_1/axis�
dense1/Tensordot/concat_1ConcatV2"dense1/Tensordot/GatherV2:output:0!dense1/Tensordot/Const_2:output:0'dense1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense1/Tensordot/concat_1�
dense1/TensordotReshape!dense1/Tensordot/MatMul:product:0"dense1/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:����������@2
dense1/Tensordot�
dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense1/BiasAdd/ReadVariableOp�
dense1/BiasAddBiasAdddense1/Tensordot:output:0%dense1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������@2
dense1/BiasAddv
dense1/ReluReludense1/BiasAdd:output:0*
T0*0
_output_shapes
:����������@2
dense1/Reluw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout_1/dropout/Const�
dropout_1/dropout/MulMuldense1/Relu:activations:0 dropout_1/dropout/Const:output:0*
T0*0
_output_shapes
:����������@2
dropout_1/dropout/Mul{
dropout_1/dropout/ShapeShapedense1/Relu:activations:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape�
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*0
_output_shapes
:����������@*
dtype020
.dropout_1/dropout/random_uniform/RandomUniform�
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2"
 dropout_1/dropout/GreaterEqual/y�
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:����������@2 
dropout_1/dropout/GreaterEqual�
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:����������@2
dropout_1/dropout/Cast�
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*0
_output_shapes
:����������@2
dropout_1/dropout/Mul_1�
dense2/Tensordot/ReadVariableOpReadVariableOp(dense2_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype02!
dense2/Tensordot/ReadVariableOpx
dense2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense2/Tensordot/axes�
dense2/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
dense2/Tensordot/free{
dense2/Tensordot/ShapeShapedropout_1/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
dense2/Tensordot/Shape�
dense2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense2/Tensordot/GatherV2/axis�
dense2/Tensordot/GatherV2GatherV2dense2/Tensordot/Shape:output:0dense2/Tensordot/free:output:0'dense2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense2/Tensordot/GatherV2�
 dense2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense2/Tensordot/GatherV2_1/axis�
dense2/Tensordot/GatherV2_1GatherV2dense2/Tensordot/Shape:output:0dense2/Tensordot/axes:output:0)dense2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense2/Tensordot/GatherV2_1z
dense2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense2/Tensordot/Const�
dense2/Tensordot/ProdProd"dense2/Tensordot/GatherV2:output:0dense2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense2/Tensordot/Prod~
dense2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense2/Tensordot/Const_1�
dense2/Tensordot/Prod_1Prod$dense2/Tensordot/GatherV2_1:output:0!dense2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense2/Tensordot/Prod_1~
dense2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense2/Tensordot/concat/axis�
dense2/Tensordot/concatConcatV2dense2/Tensordot/free:output:0dense2/Tensordot/axes:output:0%dense2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense2/Tensordot/concat�
dense2/Tensordot/stackPackdense2/Tensordot/Prod:output:0 dense2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense2/Tensordot/stack�
dense2/Tensordot/transpose	Transposedropout_1/dropout/Mul_1:z:0 dense2/Tensordot/concat:output:0*
T0*0
_output_shapes
:����������@2
dense2/Tensordot/transpose�
dense2/Tensordot/ReshapeReshapedense2/Tensordot/transpose:y:0dense2/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
dense2/Tensordot/Reshape�
dense2/Tensordot/MatMulMatMul!dense2/Tensordot/Reshape:output:0'dense2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense2/Tensordot/MatMul~
dense2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense2/Tensordot/Const_2�
dense2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense2/Tensordot/concat_1/axis�
dense2/Tensordot/concat_1ConcatV2"dense2/Tensordot/GatherV2:output:0!dense2/Tensordot/Const_2:output:0'dense2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense2/Tensordot/concat_1�
dense2/TensordotReshape!dense2/Tensordot/MatMul:product:0"dense2/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:����������2
dense2/Tensordot�
dense2/BiasAdd/ReadVariableOpReadVariableOp&dense2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense2/BiasAdd/ReadVariableOp�
dense2/BiasAddBiasAdddense2/Tensordot:output:0%dense2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
dense2/BiasAdd�
sigmoid/SigmoidSigmoiddense2/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
sigmoid/Sigmoida
reshape/ShapeShapesigmoid/Sigmoid:y:0*
T0*
_output_shapes
:2
reshape/Shape�
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack�
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1�
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2�
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_sliceu
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :�2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2�
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape�
reshape/ReshapeReshapesigmoid/Sigmoid:y:0reshape/Reshape/shape:output:0*
T0*,
_output_shapes
:����������2
reshape/Reshape�
IdentityIdentityreshape/Reshape:output:08^batch_normalization/AssignMovingAvg/AssignSubVariableOp:^batch_normalization/AssignMovingAvg_1/AssignSubVariableOp:^batch_normalization_1/AssignMovingAvg/AssignSubVariableOp<^batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp:^batch_normalization_2/AssignMovingAvg/AssignSubVariableOp<^batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp:^batch_normalization_3/AssignMovingAvg/AssignSubVariableOp<^batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp:^batch_normalization_4/AssignMovingAvg/AssignSubVariableOp<^batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:����������#::::::::::::::::::::::::::::::::::2r
7batch_normalization/AssignMovingAvg/AssignSubVariableOp7batch_normalization/AssignMovingAvg/AssignSubVariableOp2v
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp2v
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOp9batch_normalization_1/AssignMovingAvg/AssignSubVariableOp2z
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp2v
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOp9batch_normalization_2/AssignMovingAvg/AssignSubVariableOp2z
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp2v
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp2z
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp2v
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOp9batch_normalization_4/AssignMovingAvg/AssignSubVariableOp2z
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp:X T
0
_output_shapes
:����������#
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: 
�
�
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_6407

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:�����������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3r
IdentityIdentityFusedBatchNormV3:y:0*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:�����������:::::Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�$
�
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_6209

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��#AssignMovingAvg/AssignSubVariableOp�%AssignMovingAvg_1/AssignSubVariableOpu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:�����������:�:�:�:�:*
epsilon%o�:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:�����������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�$
�
M__inference_batch_normalization_layer_call_and_return_conditional_losses_5397

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��#AssignMovingAvg/AssignSubVariableOp�%AssignMovingAvg_1/AssignSubVariableOpu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
b
C__inference_dropout_1_layer_call_and_return_conditional_losses_8762

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:����������@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:����������@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:����������@2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:����������@2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:����������@2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������@:X T
0
_output_shapes
:����������@
 
_user_specified_nameinputs
�
z
%__inference_dense2_layer_call_fn_8816

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_dense2_layer_call_and_return_conditional_losses_66702
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������@::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
)__inference_sequential_layer_call_fn_7800

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32*.
Tin'
%2#*
Tout
2*,
_output_shapes
:����������*:
_read_only_resource_inputs
	
 !"*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_69102
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:����������#::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������#
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: 
�
a
C__inference_dropout_1_layer_call_and_return_conditional_losses_8767

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:����������@2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:����������@2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:����������@:X T
0
_output_shapes
:����������@
 
_user_specified_nameinputs
�
�
4__inference_batch_normalization_4_layer_call_fn_8670

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_64792
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:�����������::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�$
�
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_8639

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��#AssignMovingAvg/AssignSubVariableOp�%AssignMovingAvg_1/AssignSubVariableOpu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:�����������:�:�:�:�:*
epsilon%o�:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:�����������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
_
A__inference_dropout_layer_call_and_return_conditional_losses_8700

inputs

identity_1d
IdentityIdentityinputs*
T0*1
_output_shapes
:�����������2

Identitys

Identity_1IdentityIdentity:output:0*
T0*1
_output_shapes
:�����������2

Identity_1"!

identity_1Identity_1:output:0*0
_input_shapes
:�����������:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�$
�
M__inference_batch_normalization_layer_call_and_return_conditional_losses_6119

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��#AssignMovingAvg/AssignSubVariableOp�%AssignMovingAvg_1/AssignSubVariableOpu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:�����������:�:�:�:�:*
epsilon%o�:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:�����������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_8495

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:�����������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3r
IdentityIdentityFusedBatchNormV3:y:0*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:�����������:::::Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
`
A__inference_dropout_layer_call_and_return_conditional_losses_6545

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout/Const}
dropout/MulMulinputsdropout/Const:output:0*
T0*1
_output_shapes
:�����������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*1
_output_shapes
:�����������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:�����������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:�����������2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*1
_output_shapes
:�����������2
dropout/Mul_1o
IdentityIdentitydropout/Mul_1:z:0*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*0
_input_shapes
:�����������:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_6497

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:�����������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3r
IdentityIdentityFusedBatchNormV3:y:0*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:�����������:::::Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
Z
>__inference_max1_layer_call_and_return_conditional_losses_5307

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
a
(__inference_dropout_1_layer_call_fn_8772

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*0
_output_shapes
:����������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_66222
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������@22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������@
 
_user_specified_nameinputs
�

�
?__inference_conv4_layer_call_and_return_conditional_losses_5771

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,����������������������������2
Relu�
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,����������������������������:::j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
2__inference_batch_normalization_layer_call_fn_8035

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_54282
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
@__inference_dense2_layer_call_and_return_conditional_losses_6670

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesu
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*0
_output_shapes
:����������@2
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*0
_output_shapes
:����������2
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������@:::X T
0
_output_shapes
:����������@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
4__inference_batch_normalization_3_layer_call_fn_8433

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_58772
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
y
$__inference_conv2_layer_call_fn_5461

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_conv2_layer_call_and_return_conditional_losses_54512
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,����������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
y
$__inference_conv5_layer_call_fn_5941

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_conv5_layer_call_and_return_conditional_losses_59312
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,����������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
>
"__inference_ma5_layer_call_fn_5953

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*F
fAR?
=__inference_ma5_layer_call_and_return_conditional_losses_59472
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
M__inference_batch_normalization_layer_call_and_return_conditional_losses_6137

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*Q
_output_shapes?
=:�����������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3r
IdentityIdentityFusedBatchNormV3:y:0*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:�����������:::::Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�

�
?__inference_conv3_layer_call_and_return_conditional_losses_5611

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,����������������������������2
Relu�
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,����������������������������:::j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
2__inference_batch_normalization_layer_call_fn_7947

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*1
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_61192
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:�����������::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: "�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
J

the_inputs<
serving_default_the_inputs:0����������#@
reshape5
StatefulPartitionedCall:0����������tensorflow/serving/predict:��
��
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
layer-10
layer_with_weights-7
layer-11
layer_with_weights-8
layer-12
layer-13
layer_with_weights-9
layer-14
layer-15
layer_with_weights-10
layer-16
layer-17
layer_with_weights-11
layer-18
layer-19
layer-20
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
+�&call_and_return_all_conditional_losses
�__call__
�_default_save_signature"��
_tf_keras_sequential�{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential", "layers": [{"class_name": "Conv2D", "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1000, 35, 1]}}}, {"class_name": "MaxPooling2D", "config": {"name": "max1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 2]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 2]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "ma3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 2]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "ma4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 2]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv5", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "ma5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 2]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense2", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "sigmoid", "trainable": true, "dtype": "float32", "activation": "sigmoid"}}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1000, 1]}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 1000, 35, 1]}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1000, 35, 1]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "Conv2D", "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1000, 35, 1]}}}, {"class_name": "MaxPooling2D", "config": {"name": "max1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 2]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 2]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "ma3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 2]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "ma4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 2]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv5", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "ma5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 2]}, "data_format": "channels_last"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense2", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "sigmoid", "trainable": true, "dtype": "float32", "activation": "sigmoid"}}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1000, 1]}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 1000, 35, 1]}}}, "training_config": {"loss": "binary_crossentropy", "metrics": ["mae"], "weighted_metrics": null, "loss_weights": null, "sample_weight_mode": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�


kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1000, 35, 1]}}
�
"	variables
#trainable_variables
$regularization_losses
%	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "max1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "max1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�	
&axis
	'gamma
(beta
)moving_mean
*moving_variance
+	variables
,trainable_variables
-regularization_losses
.	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1000, 17, 128]}}
�


/kernel
0bias
1	variables
2trainable_variables
3regularization_losses
4	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1000, 17, 128]}}
�
5	variables
6trainable_variables
7regularization_losses
8	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "max2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "max2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�	
9axis
	:gamma
;beta
<moving_mean
=moving_variance
>	variables
?trainable_variables
@regularization_losses
A	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1000, 8, 128]}}
�


Bkernel
Cbias
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1000, 8, 128]}}
�
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "ma3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "ma3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�	
Laxis
	Mgamma
Nbeta
Omoving_mean
Pmoving_variance
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1000, 4, 128]}}
�


Ukernel
Vbias
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1000, 4, 128]}}
�
[	variables
\trainable_variables
]regularization_losses
^	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "ma4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "ma4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�	
_axis
	`gamma
abeta
bmoving_mean
cmoving_variance
d	variables
etrainable_variables
fregularization_losses
g	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1000, 2, 128]}}
�


hkernel
ibias
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv5", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1000, 2, 128]}}
�
n	variables
otrainable_variables
pregularization_losses
q	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "ma5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "ma5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�	
raxis
	sgamma
tbeta
umoving_mean
vmoving_variance
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1000, 1, 128]}}
�
{	variables
|trainable_variables
}regularization_losses
~	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
�

kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1000, 1, 128]}}
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense2", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1000, 1, 64]}}
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Activation", "name": "sigmoid", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "sigmoid", "trainable": true, "dtype": "float32", "activation": "sigmoid"}}
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Reshape", "name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1000, 1]}}}
�
	�iter
�beta_1
�beta_2

�decay
�learning_ratem�m�'m�(m�/m�0m�:m�;m�Bm�Cm�Mm�Nm�Um�Vm�`m�am�hm�im�sm�tm�m�	�m�	�m�	�m�v�v�'v�(v�/v�0v�:v�;v�Bv�Cv�Mv�Nv�Uv�Vv�`v�av�hv�iv�sv�tv�v�	�v�	�v�	�v�"
	optimizer
�
0
1
'2
(3
)4
*5
/6
07
:8
;9
<10
=11
B12
C13
M14
N15
O16
P17
U18
V19
`20
a21
b22
c23
h24
i25
s26
t27
u28
v29
30
�31
�32
�33"
trackable_list_wrapper
�
0
1
'2
(3
/4
05
:6
;7
B8
C9
M10
N11
U12
V13
`14
a15
h16
i17
s18
t19
20
�21
�22
�23"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
	variables
�layer_metrics
trainable_variables
�metrics
�layers
regularization_losses
 �layer_regularization_losses
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
':%�2conv1/kernel
:�2
conv1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
	variables
�layer_metrics
trainable_variables
�metrics
�layers
 regularization_losses
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
"	variables
�layer_metrics
#trainable_variables
�metrics
�layers
$regularization_losses
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
(:&�2batch_normalization/gamma
':%�2batch_normalization/beta
0:.� (2batch_normalization/moving_mean
4:2� (2#batch_normalization/moving_variance
<
'0
(1
)2
*3"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
+	variables
�layer_metrics
,trainable_variables
�metrics
�layers
-regularization_losses
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
(:&��2conv2/kernel
:�2
conv2/bias
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
1	variables
�layer_metrics
2trainable_variables
�metrics
�layers
3regularization_losses
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
5	variables
�layer_metrics
6trainable_variables
�metrics
�layers
7regularization_losses
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(�2batch_normalization_1/gamma
):'�2batch_normalization_1/beta
2:0� (2!batch_normalization_1/moving_mean
6:4� (2%batch_normalization_1/moving_variance
<
:0
;1
<2
=3"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
>	variables
�layer_metrics
?trainable_variables
�metrics
�layers
@regularization_losses
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
(:&��2conv3/kernel
:�2
conv3/bias
.
B0
C1"
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
D	variables
�layer_metrics
Etrainable_variables
�metrics
�layers
Fregularization_losses
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
H	variables
�layer_metrics
Itrainable_variables
�metrics
�layers
Jregularization_losses
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(�2batch_normalization_2/gamma
):'�2batch_normalization_2/beta
2:0� (2!batch_normalization_2/moving_mean
6:4� (2%batch_normalization_2/moving_variance
<
M0
N1
O2
P3"
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
Q	variables
�layer_metrics
Rtrainable_variables
�metrics
�layers
Sregularization_losses
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
(:&��2conv4/kernel
:�2
conv4/bias
.
U0
V1"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
W	variables
�layer_metrics
Xtrainable_variables
�metrics
�layers
Yregularization_losses
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
[	variables
�layer_metrics
\trainable_variables
�metrics
�layers
]regularization_losses
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(�2batch_normalization_3/gamma
):'�2batch_normalization_3/beta
2:0� (2!batch_normalization_3/moving_mean
6:4� (2%batch_normalization_3/moving_variance
<
`0
a1
b2
c3"
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
d	variables
�layer_metrics
etrainable_variables
�metrics
�layers
fregularization_losses
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
(:&��2conv5/kernel
:�2
conv5/bias
.
h0
i1"
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
j	variables
�layer_metrics
ktrainable_variables
�metrics
�layers
lregularization_losses
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
n	variables
�layer_metrics
otrainable_variables
�metrics
�layers
pregularization_losses
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(�2batch_normalization_4/gamma
):'�2batch_normalization_4/beta
2:0� (2!batch_normalization_4/moving_mean
6:4� (2%batch_normalization_4/moving_variance
<
s0
t1
u2
v3"
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
w	variables
�layer_metrics
xtrainable_variables
�metrics
�layers
yregularization_losses
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
{	variables
�layer_metrics
|trainable_variables
�metrics
�layers
}regularization_losses
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 :	�@2dense1/kernel
:@2dense1/bias
/
0
�1"
trackable_list_wrapper
/
0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�	variables
�layer_metrics
�trainable_variables
�metrics
�layers
�regularization_losses
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�	variables
�layer_metrics
�trainable_variables
�metrics
�layers
�regularization_losses
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:@2dense2/kernel
:2dense2/bias
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�	variables
�layer_metrics
�trainable_variables
�metrics
�layers
�regularization_losses
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�	variables
�layer_metrics
�trainable_variables
�metrics
�layers
�regularization_losses
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�	variables
�layer_metrics
�trainable_variables
�metrics
�layers
�regularization_losses
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
f
)0
*1
<2
=3
O4
P5
b6
c7
u8
v9"
trackable_list_wrapper
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
u0
v1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�

�total

�count
�	variables
�	keras_api"�
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
�

�total

�count
�
_fn_kwargs
�	variables
�	keras_api"�
_tf_keras_metric�{"class_name": "MeanMetricWrapper", "name": "mae", "dtype": "float32", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}}
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
,:*�2Adam/conv1/kernel/m
:�2Adam/conv1/bias/m
-:+�2 Adam/batch_normalization/gamma/m
,:*�2Adam/batch_normalization/beta/m
-:+��2Adam/conv2/kernel/m
:�2Adam/conv2/bias/m
/:-�2"Adam/batch_normalization_1/gamma/m
.:,�2!Adam/batch_normalization_1/beta/m
-:+��2Adam/conv3/kernel/m
:�2Adam/conv3/bias/m
/:-�2"Adam/batch_normalization_2/gamma/m
.:,�2!Adam/batch_normalization_2/beta/m
-:+��2Adam/conv4/kernel/m
:�2Adam/conv4/bias/m
/:-�2"Adam/batch_normalization_3/gamma/m
.:,�2!Adam/batch_normalization_3/beta/m
-:+��2Adam/conv5/kernel/m
:�2Adam/conv5/bias/m
/:-�2"Adam/batch_normalization_4/gamma/m
.:,�2!Adam/batch_normalization_4/beta/m
%:#	�@2Adam/dense1/kernel/m
:@2Adam/dense1/bias/m
$:"@2Adam/dense2/kernel/m
:2Adam/dense2/bias/m
,:*�2Adam/conv1/kernel/v
:�2Adam/conv1/bias/v
-:+�2 Adam/batch_normalization/gamma/v
,:*�2Adam/batch_normalization/beta/v
-:+��2Adam/conv2/kernel/v
:�2Adam/conv2/bias/v
/:-�2"Adam/batch_normalization_1/gamma/v
.:,�2!Adam/batch_normalization_1/beta/v
-:+��2Adam/conv3/kernel/v
:�2Adam/conv3/bias/v
/:-�2"Adam/batch_normalization_2/gamma/v
.:,�2!Adam/batch_normalization_2/beta/v
-:+��2Adam/conv4/kernel/v
:�2Adam/conv4/bias/v
/:-�2"Adam/batch_normalization_3/gamma/v
.:,�2!Adam/batch_normalization_3/beta/v
-:+��2Adam/conv5/kernel/v
:�2Adam/conv5/bias/v
/:-�2"Adam/batch_normalization_4/gamma/v
.:,�2!Adam/batch_normalization_4/beta/v
%:#	�@2Adam/dense1/kernel/v
:@2Adam/dense1/bias/v
$:"@2Adam/dense2/kernel/v
:2Adam/dense2/bias/v
�2�
D__inference_sequential_layer_call_and_return_conditional_losses_6814
D__inference_sequential_layer_call_and_return_conditional_losses_7548
D__inference_sequential_layer_call_and_return_conditional_losses_7727
D__inference_sequential_layer_call_and_return_conditional_losses_6721�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
)__inference_sequential_layer_call_fn_7147
)__inference_sequential_layer_call_fn_6981
)__inference_sequential_layer_call_fn_7873
)__inference_sequential_layer_call_fn_7800�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
__inference__wrapped_model_5279�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *2�/
-�*

the_inputs����������#
�2�
?__inference_conv1_layer_call_and_return_conditional_losses_5291�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������
�2�
$__inference_conv1_layer_call_fn_5301�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������
�2�
>__inference_max1_layer_call_and_return_conditional_losses_5307�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
#__inference_max1_layer_call_fn_5313�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
M__inference_batch_normalization_layer_call_and_return_conditional_losses_8009
M__inference_batch_normalization_layer_call_and_return_conditional_losses_7934
M__inference_batch_normalization_layer_call_and_return_conditional_losses_7916
M__inference_batch_normalization_layer_call_and_return_conditional_losses_7991�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
2__inference_batch_normalization_layer_call_fn_8035
2__inference_batch_normalization_layer_call_fn_8022
2__inference_batch_normalization_layer_call_fn_7960
2__inference_batch_normalization_layer_call_fn_7947�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
?__inference_conv2_layer_call_and_return_conditional_losses_5451�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *8�5
3�0,����������������������������
�2�
$__inference_conv2_layer_call_fn_5461�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *8�5
3�0,����������������������������
�2�
>__inference_max2_layer_call_and_return_conditional_losses_5467�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
#__inference_max2_layer_call_fn_5473�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8153
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8171
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8096
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8078�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
4__inference_batch_normalization_1_layer_call_fn_8197
4__inference_batch_normalization_1_layer_call_fn_8122
4__inference_batch_normalization_1_layer_call_fn_8184
4__inference_batch_normalization_1_layer_call_fn_8109�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
?__inference_conv3_layer_call_and_return_conditional_losses_5611�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *8�5
3�0,����������������������������
�2�
$__inference_conv3_layer_call_fn_5621�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *8�5
3�0,����������������������������
�2�
=__inference_ma3_layer_call_and_return_conditional_losses_5627�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
"__inference_ma3_layer_call_fn_5633�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_8315
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_8333
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_8258
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_8240�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
4__inference_batch_normalization_2_layer_call_fn_8284
4__inference_batch_normalization_2_layer_call_fn_8346
4__inference_batch_normalization_2_layer_call_fn_8271
4__inference_batch_normalization_2_layer_call_fn_8359�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
?__inference_conv4_layer_call_and_return_conditional_losses_5771�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *8�5
3�0,����������������������������
�2�
$__inference_conv4_layer_call_fn_5781�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *8�5
3�0,����������������������������
�2�
=__inference_ma4_layer_call_and_return_conditional_losses_5787�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
"__inference_ma4_layer_call_fn_5793�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_8402
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_8477
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_8420
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_8495�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
4__inference_batch_normalization_3_layer_call_fn_8508
4__inference_batch_normalization_3_layer_call_fn_8433
4__inference_batch_normalization_3_layer_call_fn_8521
4__inference_batch_normalization_3_layer_call_fn_8446�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
?__inference_conv5_layer_call_and_return_conditional_losses_5931�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *8�5
3�0,����������������������������
�2�
$__inference_conv5_layer_call_fn_5941�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *8�5
3�0,����������������������������
�2�
=__inference_ma5_layer_call_and_return_conditional_losses_5947�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
"__inference_ma5_layer_call_fn_5953�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_8582
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_8657
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_8564
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_8639�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
4__inference_batch_normalization_4_layer_call_fn_8595
4__inference_batch_normalization_4_layer_call_fn_8608
4__inference_batch_normalization_4_layer_call_fn_8683
4__inference_batch_normalization_4_layer_call_fn_8670�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
A__inference_dropout_layer_call_and_return_conditional_losses_8700
A__inference_dropout_layer_call_and_return_conditional_losses_8695�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
&__inference_dropout_layer_call_fn_8710
&__inference_dropout_layer_call_fn_8705�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
@__inference_dense1_layer_call_and_return_conditional_losses_8741�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
%__inference_dense1_layer_call_fn_8750�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_dropout_1_layer_call_and_return_conditional_losses_8762
C__inference_dropout_1_layer_call_and_return_conditional_losses_8767�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
(__inference_dropout_1_layer_call_fn_8777
(__inference_dropout_1_layer_call_fn_8772�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
@__inference_dense2_layer_call_and_return_conditional_losses_8807�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
%__inference_dense2_layer_call_fn_8816�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_sigmoid_layer_call_and_return_conditional_losses_8821�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
&__inference_sigmoid_layer_call_fn_8826�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_reshape_layer_call_and_return_conditional_losses_8839�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
&__inference_reshape_layer_call_fn_8844�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
4B2
"__inference_signature_wrapper_7290
the_inputs�
__inference__wrapped_model_5279�%'()*/0:;<=BCMNOPUV`abchistuv���<�9
2�/
-�*

the_inputs����������#
� "6�3
1
reshape&�#
reshape�����������
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8078�:;<=N�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8096�:;<=N�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8153v:;<==�:
3�0
*�'
inputs�����������
p
� "/�,
%�"
0�����������
� �
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8171v:;<==�:
3�0
*�'
inputs�����������
p 
� "/�,
%�"
0�����������
� �
4__inference_batch_normalization_1_layer_call_fn_8109�:;<=N�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
4__inference_batch_normalization_1_layer_call_fn_8122�:;<=N�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
4__inference_batch_normalization_1_layer_call_fn_8184i:;<==�:
3�0
*�'
inputs�����������
p
� ""�������������
4__inference_batch_normalization_1_layer_call_fn_8197i:;<==�:
3�0
*�'
inputs�����������
p 
� ""�������������
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_8240vMNOP=�:
3�0
*�'
inputs�����������
p
� "/�,
%�"
0�����������
� �
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_8258vMNOP=�:
3�0
*�'
inputs�����������
p 
� "/�,
%�"
0�����������
� �
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_8315�MNOPN�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_8333�MNOPN�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
4__inference_batch_normalization_2_layer_call_fn_8271iMNOP=�:
3�0
*�'
inputs�����������
p
� ""�������������
4__inference_batch_normalization_2_layer_call_fn_8284iMNOP=�:
3�0
*�'
inputs�����������
p 
� ""�������������
4__inference_batch_normalization_2_layer_call_fn_8346�MNOPN�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
4__inference_batch_normalization_2_layer_call_fn_8359�MNOPN�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_8402�`abcN�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_8420�`abcN�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_8477v`abc=�:
3�0
*�'
inputs�����������
p
� "/�,
%�"
0�����������
� �
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_8495v`abc=�:
3�0
*�'
inputs�����������
p 
� "/�,
%�"
0�����������
� �
4__inference_batch_normalization_3_layer_call_fn_8433�`abcN�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
4__inference_batch_normalization_3_layer_call_fn_8446�`abcN�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
4__inference_batch_normalization_3_layer_call_fn_8508i`abc=�:
3�0
*�'
inputs�����������
p
� ""�������������
4__inference_batch_normalization_3_layer_call_fn_8521i`abc=�:
3�0
*�'
inputs�����������
p 
� ""�������������
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_8564�stuvN�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_8582�stuvN�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_8639vstuv=�:
3�0
*�'
inputs�����������
p
� "/�,
%�"
0�����������
� �
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_8657vstuv=�:
3�0
*�'
inputs�����������
p 
� "/�,
%�"
0�����������
� �
4__inference_batch_normalization_4_layer_call_fn_8595�stuvN�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
4__inference_batch_normalization_4_layer_call_fn_8608�stuvN�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
4__inference_batch_normalization_4_layer_call_fn_8670istuv=�:
3�0
*�'
inputs�����������
p
� ""�������������
4__inference_batch_normalization_4_layer_call_fn_8683istuv=�:
3�0
*�'
inputs�����������
p 
� ""�������������
M__inference_batch_normalization_layer_call_and_return_conditional_losses_7916v'()*=�:
3�0
*�'
inputs�����������
p
� "/�,
%�"
0�����������
� �
M__inference_batch_normalization_layer_call_and_return_conditional_losses_7934v'()*=�:
3�0
*�'
inputs�����������
p 
� "/�,
%�"
0�����������
� �
M__inference_batch_normalization_layer_call_and_return_conditional_losses_7991�'()*N�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
M__inference_batch_normalization_layer_call_and_return_conditional_losses_8009�'()*N�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
2__inference_batch_normalization_layer_call_fn_7947i'()*=�:
3�0
*�'
inputs�����������
p
� ""�������������
2__inference_batch_normalization_layer_call_fn_7960i'()*=�:
3�0
*�'
inputs�����������
p 
� ""�������������
2__inference_batch_normalization_layer_call_fn_8022�'()*N�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
2__inference_batch_normalization_layer_call_fn_8035�'()*N�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
?__inference_conv1_layer_call_and_return_conditional_losses_5291�I�F
?�<
:�7
inputs+���������������������������
� "@�=
6�3
0,����������������������������
� �
$__inference_conv1_layer_call_fn_5301�I�F
?�<
:�7
inputs+���������������������������
� "3�0,�����������������������������
?__inference_conv2_layer_call_and_return_conditional_losses_5451�/0J�G
@�=
;�8
inputs,����������������������������
� "@�=
6�3
0,����������������������������
� �
$__inference_conv2_layer_call_fn_5461�/0J�G
@�=
;�8
inputs,����������������������������
� "3�0,�����������������������������
?__inference_conv3_layer_call_and_return_conditional_losses_5611�BCJ�G
@�=
;�8
inputs,����������������������������
� "@�=
6�3
0,����������������������������
� �
$__inference_conv3_layer_call_fn_5621�BCJ�G
@�=
;�8
inputs,����������������������������
� "3�0,�����������������������������
?__inference_conv4_layer_call_and_return_conditional_losses_5771�UVJ�G
@�=
;�8
inputs,����������������������������
� "@�=
6�3
0,����������������������������
� �
$__inference_conv4_layer_call_fn_5781�UVJ�G
@�=
;�8
inputs,����������������������������
� "3�0,�����������������������������
?__inference_conv5_layer_call_and_return_conditional_losses_5931�hiJ�G
@�=
;�8
inputs,����������������������������
� "@�=
6�3
0,����������������������������
� �
$__inference_conv5_layer_call_fn_5941�hiJ�G
@�=
;�8
inputs,����������������������������
� "3�0,�����������������������������
@__inference_dense1_layer_call_and_return_conditional_losses_8741p�9�6
/�,
*�'
inputs�����������
� ".�+
$�!
0����������@
� �
%__inference_dense1_layer_call_fn_8750c�9�6
/�,
*�'
inputs�����������
� "!�����������@�
@__inference_dense2_layer_call_and_return_conditional_losses_8807p��8�5
.�+
)�&
inputs����������@
� ".�+
$�!
0����������
� �
%__inference_dense2_layer_call_fn_8816c��8�5
.�+
)�&
inputs����������@
� "!������������
C__inference_dropout_1_layer_call_and_return_conditional_losses_8762n<�9
2�/
)�&
inputs����������@
p
� ".�+
$�!
0����������@
� �
C__inference_dropout_1_layer_call_and_return_conditional_losses_8767n<�9
2�/
)�&
inputs����������@
p 
� ".�+
$�!
0����������@
� �
(__inference_dropout_1_layer_call_fn_8772a<�9
2�/
)�&
inputs����������@
p
� "!�����������@�
(__inference_dropout_1_layer_call_fn_8777a<�9
2�/
)�&
inputs����������@
p 
� "!�����������@�
A__inference_dropout_layer_call_and_return_conditional_losses_8695p=�:
3�0
*�'
inputs�����������
p
� "/�,
%�"
0�����������
� �
A__inference_dropout_layer_call_and_return_conditional_losses_8700p=�:
3�0
*�'
inputs�����������
p 
� "/�,
%�"
0�����������
� �
&__inference_dropout_layer_call_fn_8705c=�:
3�0
*�'
inputs�����������
p
� ""�������������
&__inference_dropout_layer_call_fn_8710c=�:
3�0
*�'
inputs�����������
p 
� ""�������������
=__inference_ma3_layer_call_and_return_conditional_losses_5627�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
"__inference_ma3_layer_call_fn_5633�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
=__inference_ma4_layer_call_and_return_conditional_losses_5787�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
"__inference_ma4_layer_call_fn_5793�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
=__inference_ma5_layer_call_and_return_conditional_losses_5947�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
"__inference_ma5_layer_call_fn_5953�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
>__inference_max1_layer_call_and_return_conditional_losses_5307�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
#__inference_max1_layer_call_fn_5313�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
>__inference_max2_layer_call_and_return_conditional_losses_5467�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
#__inference_max2_layer_call_fn_5473�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
A__inference_reshape_layer_call_and_return_conditional_losses_8839f8�5
.�+
)�&
inputs����������
� "*�'
 �
0����������
� �
&__inference_reshape_layer_call_fn_8844Y8�5
.�+
)�&
inputs����������
� "������������
D__inference_sequential_layer_call_and_return_conditional_losses_6721�%'()*/0:;<=BCMNOPUV`abchistuv���D�A
:�7
-�*

the_inputs����������#
p

 
� "*�'
 �
0����������
� �
D__inference_sequential_layer_call_and_return_conditional_losses_6814�%'()*/0:;<=BCMNOPUV`abchistuv���D�A
:�7
-�*

the_inputs����������#
p 

 
� "*�'
 �
0����������
� �
D__inference_sequential_layer_call_and_return_conditional_losses_7548�%'()*/0:;<=BCMNOPUV`abchistuv���@�=
6�3
)�&
inputs����������#
p

 
� "*�'
 �
0����������
� �
D__inference_sequential_layer_call_and_return_conditional_losses_7727�%'()*/0:;<=BCMNOPUV`abchistuv���@�=
6�3
)�&
inputs����������#
p 

 
� "*�'
 �
0����������
� �
)__inference_sequential_layer_call_fn_6981�%'()*/0:;<=BCMNOPUV`abchistuv���D�A
:�7
-�*

the_inputs����������#
p

 
� "������������
)__inference_sequential_layer_call_fn_7147�%'()*/0:;<=BCMNOPUV`abchistuv���D�A
:�7
-�*

the_inputs����������#
p 

 
� "������������
)__inference_sequential_layer_call_fn_7800�%'()*/0:;<=BCMNOPUV`abchistuv���@�=
6�3
)�&
inputs����������#
p

 
� "������������
)__inference_sequential_layer_call_fn_7873�%'()*/0:;<=BCMNOPUV`abchistuv���@�=
6�3
)�&
inputs����������#
p 

 
� "������������
A__inference_sigmoid_layer_call_and_return_conditional_losses_8821j8�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� �
&__inference_sigmoid_layer_call_fn_8826]8�5
.�+
)�&
inputs����������
� "!������������
"__inference_signature_wrapper_7290�%'()*/0:;<=BCMNOPUV`abchistuv���J�G
� 
@�=
;

the_inputs-�*

the_inputs����������#"6�3
1
reshape&�#
reshape����������
Ì!
½
B
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
:
FloorMod
x"T
y"T
z"T"
Ttype:
	2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
:
Maximum
x"T
y"T
z"T"
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
:
Minimum
x"T
y"T
z"T"
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
¥
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
-
Sqrt
x"T
y"T"
Ttype:

2
¾
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
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.4.12v2.4.0-49-g85c8b2a817f8¯º
|
dense_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namedense_18/kernel
u
#dense_18/kernel/Read/ReadVariableOpReadVariableOpdense_18/kernel* 
_output_shapes
:
*
dtype0
s
dense_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_18/bias
l
!dense_18/bias/Read/ReadVariableOpReadVariableOpdense_18/bias*
_output_shapes	
:*
dtype0
|
dense_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namedense_19/kernel
u
#dense_19/kernel/Read/ReadVariableOpReadVariableOpdense_19/kernel* 
_output_shapes
:
*
dtype0
s
dense_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_19/bias
l
!dense_19/bias/Read/ReadVariableOpReadVariableOpdense_19/bias*
_output_shapes	
:*
dtype0
{
dense_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	* 
shared_namedense_21/kernel
t
#dense_21/kernel/Read/ReadVariableOpReadVariableOpdense_21/kernel*
_output_shapes
:	*
dtype0
s
dense_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_21/bias
l
!dense_21/bias/Read/ReadVariableOpReadVariableOpdense_21/bias*
_output_shapes	
:*
dtype0
|
dense_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namedense_20/kernel
u
#dense_20/kernel/Read/ReadVariableOpReadVariableOpdense_20/kernel* 
_output_shapes
:
*
dtype0
s
dense_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_20/bias
l
!dense_20/bias/Read/ReadVariableOpReadVariableOpdense_20/bias*
_output_shapes	
:*
dtype0
|
dense_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namedense_22/kernel
u
#dense_22/kernel/Read/ReadVariableOpReadVariableOpdense_22/kernel* 
_output_shapes
:
*
dtype0
s
dense_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_22/bias
l
!dense_22/bias/Read/ReadVariableOpReadVariableOpdense_22/bias*
_output_shapes	
:*
dtype0
|
dense_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namedense_23/kernel
u
#dense_23/kernel/Read/ReadVariableOpReadVariableOpdense_23/kernel* 
_output_shapes
:
*
dtype0
s
dense_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_23/bias
l
!dense_23/bias/Read/ReadVariableOpReadVariableOpdense_23/bias*
_output_shapes	
:*
dtype0
|
dense_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namedense_24/kernel
u
#dense_24/kernel/Read/ReadVariableOpReadVariableOpdense_24/kernel* 
_output_shapes
:
*
dtype0
s
dense_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_24/bias
l
!dense_24/bias/Read/ReadVariableOpReadVariableOpdense_24/bias*
_output_shapes	
:*
dtype0
|
dense_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namedense_25/kernel
u
#dense_25/kernel/Read/ReadVariableOpReadVariableOpdense_25/kernel* 
_output_shapes
:
*
dtype0
s
dense_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_25/bias
l
!dense_25/bias/Read/ReadVariableOpReadVariableOpdense_25/bias*
_output_shapes	
:*
dtype0
{
dense_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	* 
shared_namedense_26/kernel
t
#dense_26/kernel/Read/ReadVariableOpReadVariableOpdense_26/kernel*
_output_shapes
:	*
dtype0
r
dense_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_26/bias
k
!dense_26/bias/Read/ReadVariableOpReadVariableOpdense_26/bias*
_output_shapes
:*
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

embedding_14/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	4*(
shared_nameembedding_14/embeddings

+embedding_14/embeddings/Read/ReadVariableOpReadVariableOpembedding_14/embeddings*
_output_shapes
:	4*
dtype0

embedding_12/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*(
shared_nameembedding_12/embeddings

+embedding_12/embeddings/Read/ReadVariableOpReadVariableOpembedding_12/embeddings*
_output_shapes
:	*
dtype0

embedding_13/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*(
shared_nameembedding_13/embeddings

+embedding_13/embeddings/Read/ReadVariableOpReadVariableOpembedding_13/embeddings*
_output_shapes
:	*
dtype0

embedding_17/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	4*(
shared_nameembedding_17/embeddings

+embedding_17/embeddings/Read/ReadVariableOpReadVariableOpembedding_17/embeddings*
_output_shapes
:	4*
dtype0

embedding_15/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*(
shared_nameembedding_15/embeddings

+embedding_15/embeddings/Read/ReadVariableOpReadVariableOpembedding_15/embeddings*
_output_shapes
:	*
dtype0

embedding_16/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*(
shared_nameembedding_16/embeddings

+embedding_16/embeddings/Read/ReadVariableOpReadVariableOpembedding_16/embeddings*
_output_shapes
:	*
dtype0

 normalize_2/normalization_2/meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" normalize_2/normalization_2/mean

4normalize_2/normalization_2/mean/Read/ReadVariableOpReadVariableOp normalize_2/normalization_2/mean*
_output_shapes	
:*
dtype0
¡
$normalize_2/normalization_2/varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$normalize_2/normalization_2/variance

8normalize_2/normalization_2/variance/Read/ReadVariableOpReadVariableOp$normalize_2/normalization_2/variance*
_output_shapes	
:*
dtype0

!normalize_2/normalization_2/countVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *2
shared_name#!normalize_2/normalization_2/count

5normalize_2/normalization_2/count/Read/ReadVariableOpReadVariableOp!normalize_2/normalization_2/count*
_output_shapes
: *
dtype0	
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

Adam/dense_18/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/dense_18/kernel/m

*Adam/dense_18/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_18/kernel/m* 
_output_shapes
:
*
dtype0

Adam/dense_18/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_18/bias/m
z
(Adam/dense_18/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_18/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_19/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/dense_19/kernel/m

*Adam/dense_19/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_19/kernel/m* 
_output_shapes
:
*
dtype0

Adam/dense_19/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_19/bias/m
z
(Adam/dense_19/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_19/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_21/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameAdam/dense_21/kernel/m

*Adam/dense_21/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_21/kernel/m*
_output_shapes
:	*
dtype0

Adam/dense_21/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_21/bias/m
z
(Adam/dense_21/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_21/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_20/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/dense_20/kernel/m

*Adam/dense_20/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_20/kernel/m* 
_output_shapes
:
*
dtype0

Adam/dense_20/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_20/bias/m
z
(Adam/dense_20/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_20/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_22/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/dense_22/kernel/m

*Adam/dense_22/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_22/kernel/m* 
_output_shapes
:
*
dtype0

Adam/dense_22/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_22/bias/m
z
(Adam/dense_22/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_22/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_23/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/dense_23/kernel/m

*Adam/dense_23/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_23/kernel/m* 
_output_shapes
:
*
dtype0

Adam/dense_23/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_23/bias/m
z
(Adam/dense_23/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_23/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_24/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/dense_24/kernel/m

*Adam/dense_24/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_24/kernel/m* 
_output_shapes
:
*
dtype0

Adam/dense_24/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_24/bias/m
z
(Adam/dense_24/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_24/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_25/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/dense_25/kernel/m

*Adam/dense_25/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_25/kernel/m* 
_output_shapes
:
*
dtype0

Adam/dense_25/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_25/bias/m
z
(Adam/dense_25/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_25/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_26/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameAdam/dense_26/kernel/m

*Adam/dense_26/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_26/kernel/m*
_output_shapes
:	*
dtype0

Adam/dense_26/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_26/bias/m
y
(Adam/dense_26/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_26/bias/m*
_output_shapes
:*
dtype0

Adam/embedding_14/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	4*/
shared_name Adam/embedding_14/embeddings/m

2Adam/embedding_14/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_14/embeddings/m*
_output_shapes
:	4*
dtype0

Adam/embedding_12/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*/
shared_name Adam/embedding_12/embeddings/m

2Adam/embedding_12/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_12/embeddings/m*
_output_shapes
:	*
dtype0

Adam/embedding_13/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*/
shared_name Adam/embedding_13/embeddings/m

2Adam/embedding_13/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_13/embeddings/m*
_output_shapes
:	*
dtype0

Adam/embedding_17/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	4*/
shared_name Adam/embedding_17/embeddings/m

2Adam/embedding_17/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_17/embeddings/m*
_output_shapes
:	4*
dtype0

Adam/embedding_15/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*/
shared_name Adam/embedding_15/embeddings/m

2Adam/embedding_15/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_15/embeddings/m*
_output_shapes
:	*
dtype0

Adam/embedding_16/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*/
shared_name Adam/embedding_16/embeddings/m

2Adam/embedding_16/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_16/embeddings/m*
_output_shapes
:	*
dtype0

Adam/dense_18/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/dense_18/kernel/v

*Adam/dense_18/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_18/kernel/v* 
_output_shapes
:
*
dtype0

Adam/dense_18/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_18/bias/v
z
(Adam/dense_18/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_18/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_19/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/dense_19/kernel/v

*Adam/dense_19/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_19/kernel/v* 
_output_shapes
:
*
dtype0

Adam/dense_19/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_19/bias/v
z
(Adam/dense_19/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_19/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_21/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameAdam/dense_21/kernel/v

*Adam/dense_21/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_21/kernel/v*
_output_shapes
:	*
dtype0

Adam/dense_21/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_21/bias/v
z
(Adam/dense_21/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_21/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_20/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/dense_20/kernel/v

*Adam/dense_20/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_20/kernel/v* 
_output_shapes
:
*
dtype0

Adam/dense_20/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_20/bias/v
z
(Adam/dense_20/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_20/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_22/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/dense_22/kernel/v

*Adam/dense_22/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_22/kernel/v* 
_output_shapes
:
*
dtype0

Adam/dense_22/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_22/bias/v
z
(Adam/dense_22/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_22/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_23/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/dense_23/kernel/v

*Adam/dense_23/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_23/kernel/v* 
_output_shapes
:
*
dtype0

Adam/dense_23/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_23/bias/v
z
(Adam/dense_23/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_23/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_24/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/dense_24/kernel/v

*Adam/dense_24/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_24/kernel/v* 
_output_shapes
:
*
dtype0

Adam/dense_24/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_24/bias/v
z
(Adam/dense_24/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_24/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_25/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/dense_25/kernel/v

*Adam/dense_25/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_25/kernel/v* 
_output_shapes
:
*
dtype0

Adam/dense_25/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_25/bias/v
z
(Adam/dense_25/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_25/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_26/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameAdam/dense_26/kernel/v

*Adam/dense_26/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_26/kernel/v*
_output_shapes
:	*
dtype0

Adam/dense_26/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_26/bias/v
y
(Adam/dense_26/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_26/bias/v*
_output_shapes
:*
dtype0

Adam/embedding_14/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	4*/
shared_name Adam/embedding_14/embeddings/v

2Adam/embedding_14/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_14/embeddings/v*
_output_shapes
:	4*
dtype0

Adam/embedding_12/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*/
shared_name Adam/embedding_12/embeddings/v

2Adam/embedding_12/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_12/embeddings/v*
_output_shapes
:	*
dtype0

Adam/embedding_13/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*/
shared_name Adam/embedding_13/embeddings/v

2Adam/embedding_13/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_13/embeddings/v*
_output_shapes
:	*
dtype0

Adam/embedding_17/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	4*/
shared_name Adam/embedding_17/embeddings/v

2Adam/embedding_17/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_17/embeddings/v*
_output_shapes
:	4*
dtype0

Adam/embedding_15/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*/
shared_name Adam/embedding_15/embeddings/v

2Adam/embedding_15/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_15/embeddings/v*
_output_shapes
:	*
dtype0

Adam/embedding_16/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*/
shared_name Adam/embedding_16/embeddings/v

2Adam/embedding_16/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_16/embeddings/v*
_output_shapes
:	*
dtype0
J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *    
L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 * $tI
L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *    

NoOpNoOp
å
Const_5Const"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB B

layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer-5
layer-6
layer-7
	layer-8

layer_with_weights-2

layer-9
layer-10
layer_with_weights-3
layer-11
layer_with_weights-4
layer-12
layer_with_weights-5
layer-13
layer_with_weights-6
layer-14
layer-15
layer_with_weights-7
layer-16
layer-17
layer_with_weights-8
layer-18
layer-19
layer-20
layer_with_weights-9
layer-21
layer-22
layer-23
layer_with_weights-10
layer-24
layer_with_weights-11
layer-25
	optimizer
loss
	variables
trainable_variables
regularization_losses
 	keras_api
!
signatures
 
 
 
è
"layer-0
#layer-1
$layer-2
%layer-3
&layer_with_weights-0
&layer-4
'layer_with_weights-1
'layer-5
(layer-6
)layer-7
*layer-8
+layer_with_weights-2
+layer-9
,layer-10
-layer-11
.layer-12
/layer-13
0layer-14
1	variables
2trainable_variables
3regularization_losses
4	keras_api
è
5layer-0
6layer-1
7layer-2
8layer-3
9layer_with_weights-0
9layer-4
:layer_with_weights-1
:layer-5
;layer-6
<layer-7
=layer-8
>layer_with_weights-2
>layer-9
?layer-10
@layer-11
Alayer-12
Blayer-13
Clayer-14
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api

H	keras_api

I	keras_api

J	keras_api

K	keras_api
h

Lkernel
Mbias
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api

R	keras_api
h

Skernel
Tbias
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
h

Ykernel
Zbias
[	variables
\trainable_variables
]regularization_losses
^	keras_api
h

_kernel
`bias
a	variables
btrainable_variables
cregularization_losses
d	keras_api
h

ekernel
fbias
g	variables
htrainable_variables
iregularization_losses
j	keras_api

k	keras_api
h

lkernel
mbias
n	variables
otrainable_variables
pregularization_losses
q	keras_api

r	keras_api
h

skernel
tbias
u	variables
vtrainable_variables
wregularization_losses
x	keras_api

y	keras_api

z	keras_api
i

{kernel
|bias
}	variables
~trainable_variables
regularization_losses
	keras_api

	keras_api

	keras_api
f
	normalize
	variables
trainable_variables
regularization_losses
	keras_api
n
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
µ
	iter
beta_1
beta_2

decay
learning_rateLmÁMmÂSmÃTmÄYmÅZmÆ_mÇ`mÈemÉfmÊlmËmmÌsmÍtmÎ{mÏ|mÐ	mÑ	mÒ	mÓ	mÔ	mÕ	mÖ	m×	mØLvÙMvÚSvÛTvÜYvÝZvÞ_vß`vàeváfvâlvãmväsvåtvæ{vç|vè	vé	vê	vë	vì	ví	vî	vï	vð
 
Ù
0
1
2
3
4
5
L6
M7
S8
T9
Y10
Z11
_12
`13
e14
f15
l16
m17
s18
t19
{20
|21
22
23
24
25
26
¾
0
1
2
3
4
5
L6
M7
S8
T9
Y10
Z11
_12
`13
e14
f15
l16
m17
s18
t19
{20
|21
22
23
 
²
 layer_regularization_losses
layers
	variables
non_trainable_variables
trainable_variables
regularization_losses
layer_metrics
 metrics
 
 
V
¡	variables
¢trainable_variables
£regularization_losses
¤	keras_api

¥	keras_api

¦	keras_api
g

embeddings
§	variables
¨trainable_variables
©regularization_losses
ª	keras_api
g

embeddings
«	variables
¬trainable_variables
­regularization_losses
®	keras_api

¯	keras_api

°	keras_api

±	keras_api
g

embeddings
²	variables
³trainable_variables
´regularization_losses
µ	keras_api

¶	keras_api

·	keras_api

¸	keras_api

¹	keras_api

º	keras_api

0
1
2

0
1
2
 
²
 »layer_regularization_losses
¼layers
1	variables
½non_trainable_variables
2trainable_variables
3regularization_losses
¾layer_metrics
¿metrics
 
V
À	variables
Átrainable_variables
Âregularization_losses
Ã	keras_api

Ä	keras_api

Å	keras_api
g

embeddings
Æ	variables
Çtrainable_variables
Èregularization_losses
É	keras_api
g

embeddings
Ê	variables
Ëtrainable_variables
Ìregularization_losses
Í	keras_api

Î	keras_api

Ï	keras_api

Ð	keras_api
g

embeddings
Ñ	variables
Òtrainable_variables
Óregularization_losses
Ô	keras_api

Õ	keras_api

Ö	keras_api

×	keras_api

Ø	keras_api

Ù	keras_api

0
1
2

0
1
2
 
²
 Úlayer_regularization_losses
Ûlayers
D	variables
Ünon_trainable_variables
Etrainable_variables
Fregularization_losses
Ýlayer_metrics
Þmetrics
 
 
 
 
[Y
VARIABLE_VALUEdense_18/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_18/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

L0
M1

L0
M1
 
²
 ßlayer_regularization_losses
àlayers
N	variables
ánon_trainable_variables
Otrainable_variables
Pregularization_losses
âlayer_metrics
ãmetrics
 
[Y
VARIABLE_VALUEdense_19/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_19/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

S0
T1

S0
T1
 
²
 älayer_regularization_losses
ålayers
U	variables
ænon_trainable_variables
Vtrainable_variables
Wregularization_losses
çlayer_metrics
èmetrics
[Y
VARIABLE_VALUEdense_21/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_21/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

Y0
Z1

Y0
Z1
 
²
 élayer_regularization_losses
êlayers
[	variables
ënon_trainable_variables
\trainable_variables
]regularization_losses
ìlayer_metrics
ímetrics
[Y
VARIABLE_VALUEdense_20/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_20/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

_0
`1

_0
`1
 
²
 îlayer_regularization_losses
ïlayers
a	variables
ðnon_trainable_variables
btrainable_variables
cregularization_losses
ñlayer_metrics
òmetrics
[Y
VARIABLE_VALUEdense_22/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_22/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

e0
f1

e0
f1
 
²
 ólayer_regularization_losses
ôlayers
g	variables
õnon_trainable_variables
htrainable_variables
iregularization_losses
ölayer_metrics
÷metrics
 
[Y
VARIABLE_VALUEdense_23/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_23/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

l0
m1

l0
m1
 
²
 ølayer_regularization_losses
ùlayers
n	variables
únon_trainable_variables
otrainable_variables
pregularization_losses
ûlayer_metrics
ümetrics
 
[Y
VARIABLE_VALUEdense_24/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_24/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

s0
t1

s0
t1
 
²
 ýlayer_regularization_losses
þlayers
u	variables
ÿnon_trainable_variables
vtrainable_variables
wregularization_losses
layer_metrics
metrics
 
 
[Y
VARIABLE_VALUEdense_25/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_25/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

{0
|1

{0
|1
 
²
 layer_regularization_losses
layers
}	variables
non_trainable_variables
~trainable_variables
regularization_losses
layer_metrics
metrics
 
 
c
state_variables
_broadcast_shape
	mean
variance

count
	keras_api

0
1
2
 
 
µ
 layer_regularization_losses
layers
	variables
non_trainable_variables
trainable_variables
regularization_losses
layer_metrics
metrics
\Z
VARIABLE_VALUEdense_26/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_26/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
µ
 layer_regularization_losses
layers
	variables
non_trainable_variables
trainable_variables
regularization_losses
layer_metrics
metrics
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
SQ
VARIABLE_VALUEembedding_14/embeddings&variables/0/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEembedding_12/embeddings&variables/1/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEembedding_13/embeddings&variables/2/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEembedding_17/embeddings&variables/3/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEembedding_15/embeddings&variables/4/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEembedding_16/embeddings&variables/5/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE normalize_2/normalization_2/mean'variables/22/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE$normalize_2/normalization_2/variance'variables/23/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!normalize_2/normalization_2/count'variables/24/.ATTRIBUTES/VARIABLE_VALUE
 
Æ
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
21
22
23
24
25

0
1
2
 

0
 
 
 
µ
 layer_regularization_losses
layers
¡	variables
non_trainable_variables
¢trainable_variables
£regularization_losses
layer_metrics
metrics
 
 

0

0
 
µ
 layer_regularization_losses
layers
§	variables
non_trainable_variables
¨trainable_variables
©regularization_losses
layer_metrics
metrics

0

0
 
µ
 layer_regularization_losses
 layers
«	variables
¡non_trainable_variables
¬trainable_variables
­regularization_losses
¢layer_metrics
£metrics
 
 
 

0

0
 
µ
 ¤layer_regularization_losses
¥layers
²	variables
¦non_trainable_variables
³trainable_variables
´regularization_losses
§layer_metrics
¨metrics
 
 
 
 
 
 
n
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10
-11
.12
/13
014
 
 
 
 
 
 
µ
 ©layer_regularization_losses
ªlayers
À	variables
«non_trainable_variables
Átrainable_variables
Âregularization_losses
¬layer_metrics
­metrics
 
 

0

0
 
µ
 ®layer_regularization_losses
¯layers
Æ	variables
°non_trainable_variables
Çtrainable_variables
Èregularization_losses
±layer_metrics
²metrics

0

0
 
µ
 ³layer_regularization_losses
´layers
Ê	variables
µnon_trainable_variables
Ëtrainable_variables
Ìregularization_losses
¶layer_metrics
·metrics
 
 
 

0

0
 
µ
 ¸layer_regularization_losses
¹layers
Ñ	variables
ºnon_trainable_variables
Òtrainable_variables
Óregularization_losses
»layer_metrics
¼metrics
 
 
 
 
 
 
n
50
61
72
83
94
:5
;6
<7
=8
>9
?10
@11
A12
B13
C14
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
 
 
 
 
 
 
 
 
 
&
	mean
variance

count
 
 
 

0

0
1
2
 
 
 
 
 
 
 
8

½total

¾count
¿	variables
À	keras_api
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
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

½0
¾1

¿	variables
~|
VARIABLE_VALUEAdam/dense_18/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_18/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_19/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_19/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_21/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_21/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_20/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_20/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_22/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_22/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_23/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_23/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_24/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_24/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_25/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_25/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_26/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_26/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/embedding_14/embeddings/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/embedding_12/embeddings/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/embedding_13/embeddings/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/embedding_17/embeddings/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/embedding_15/embeddings/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/embedding_16/embeddings/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_18/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_18/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_19/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_19/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_21/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_21/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_20/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_20/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_22/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_22/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_23/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_23/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_24/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_24/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_25/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_25/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_26/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_26/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/embedding_14/embeddings/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/embedding_12/embeddings/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/embedding_13/embeddings/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/embedding_17/embeddings/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/embedding_15/embeddings/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/embedding_16/embeddings/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
w
serving_default_betsPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ

y
serving_default_cards0Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
y
serving_default_cards1Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
Æ
StatefulPartitionedCallStatefulPartitionedCallserving_default_betsserving_default_cards0serving_default_cards1ConstConst_1embedding_14/embeddingsembedding_12/embeddingsembedding_13/embeddingsConst_2embedding_17/embeddingsembedding_15/embeddingsembedding_16/embeddingsConst_3Const_4dense_18/kerneldense_18/biasdense_21/kerneldense_21/biasdense_19/kerneldense_19/biasdense_20/kerneldense_20/biasdense_22/kerneldense_22/biasdense_23/kerneldense_23/biasdense_24/kerneldense_24/biasdense_25/kerneldense_25/bias normalize_2/normalization_2/mean$normalize_2/normalization_2/variancedense_26/kerneldense_26/bias*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*<
_read_only_resource_inputs
	
 !*-
config_proto

CPU

GPU 2J 8 *0
f+R)
'__inference_signature_wrapper_200072082
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_18/kernel/Read/ReadVariableOp!dense_18/bias/Read/ReadVariableOp#dense_19/kernel/Read/ReadVariableOp!dense_19/bias/Read/ReadVariableOp#dense_21/kernel/Read/ReadVariableOp!dense_21/bias/Read/ReadVariableOp#dense_20/kernel/Read/ReadVariableOp!dense_20/bias/Read/ReadVariableOp#dense_22/kernel/Read/ReadVariableOp!dense_22/bias/Read/ReadVariableOp#dense_23/kernel/Read/ReadVariableOp!dense_23/bias/Read/ReadVariableOp#dense_24/kernel/Read/ReadVariableOp!dense_24/bias/Read/ReadVariableOp#dense_25/kernel/Read/ReadVariableOp!dense_25/bias/Read/ReadVariableOp#dense_26/kernel/Read/ReadVariableOp!dense_26/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp+embedding_14/embeddings/Read/ReadVariableOp+embedding_12/embeddings/Read/ReadVariableOp+embedding_13/embeddings/Read/ReadVariableOp+embedding_17/embeddings/Read/ReadVariableOp+embedding_15/embeddings/Read/ReadVariableOp+embedding_16/embeddings/Read/ReadVariableOp4normalize_2/normalization_2/mean/Read/ReadVariableOp8normalize_2/normalization_2/variance/Read/ReadVariableOp5normalize_2/normalization_2/count/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_18/kernel/m/Read/ReadVariableOp(Adam/dense_18/bias/m/Read/ReadVariableOp*Adam/dense_19/kernel/m/Read/ReadVariableOp(Adam/dense_19/bias/m/Read/ReadVariableOp*Adam/dense_21/kernel/m/Read/ReadVariableOp(Adam/dense_21/bias/m/Read/ReadVariableOp*Adam/dense_20/kernel/m/Read/ReadVariableOp(Adam/dense_20/bias/m/Read/ReadVariableOp*Adam/dense_22/kernel/m/Read/ReadVariableOp(Adam/dense_22/bias/m/Read/ReadVariableOp*Adam/dense_23/kernel/m/Read/ReadVariableOp(Adam/dense_23/bias/m/Read/ReadVariableOp*Adam/dense_24/kernel/m/Read/ReadVariableOp(Adam/dense_24/bias/m/Read/ReadVariableOp*Adam/dense_25/kernel/m/Read/ReadVariableOp(Adam/dense_25/bias/m/Read/ReadVariableOp*Adam/dense_26/kernel/m/Read/ReadVariableOp(Adam/dense_26/bias/m/Read/ReadVariableOp2Adam/embedding_14/embeddings/m/Read/ReadVariableOp2Adam/embedding_12/embeddings/m/Read/ReadVariableOp2Adam/embedding_13/embeddings/m/Read/ReadVariableOp2Adam/embedding_17/embeddings/m/Read/ReadVariableOp2Adam/embedding_15/embeddings/m/Read/ReadVariableOp2Adam/embedding_16/embeddings/m/Read/ReadVariableOp*Adam/dense_18/kernel/v/Read/ReadVariableOp(Adam/dense_18/bias/v/Read/ReadVariableOp*Adam/dense_19/kernel/v/Read/ReadVariableOp(Adam/dense_19/bias/v/Read/ReadVariableOp*Adam/dense_21/kernel/v/Read/ReadVariableOp(Adam/dense_21/bias/v/Read/ReadVariableOp*Adam/dense_20/kernel/v/Read/ReadVariableOp(Adam/dense_20/bias/v/Read/ReadVariableOp*Adam/dense_22/kernel/v/Read/ReadVariableOp(Adam/dense_22/bias/v/Read/ReadVariableOp*Adam/dense_23/kernel/v/Read/ReadVariableOp(Adam/dense_23/bias/v/Read/ReadVariableOp*Adam/dense_24/kernel/v/Read/ReadVariableOp(Adam/dense_24/bias/v/Read/ReadVariableOp*Adam/dense_25/kernel/v/Read/ReadVariableOp(Adam/dense_25/bias/v/Read/ReadVariableOp*Adam/dense_26/kernel/v/Read/ReadVariableOp(Adam/dense_26/bias/v/Read/ReadVariableOp2Adam/embedding_14/embeddings/v/Read/ReadVariableOp2Adam/embedding_12/embeddings/v/Read/ReadVariableOp2Adam/embedding_13/embeddings/v/Read/ReadVariableOp2Adam/embedding_17/embeddings/v/Read/ReadVariableOp2Adam/embedding_15/embeddings/v/Read/ReadVariableOp2Adam/embedding_16/embeddings/v/Read/ReadVariableOpConst_5*_
TinX
V2T		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_save_200073380

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_18/kerneldense_18/biasdense_19/kerneldense_19/biasdense_21/kerneldense_21/biasdense_20/kerneldense_20/biasdense_22/kerneldense_22/biasdense_23/kerneldense_23/biasdense_24/kerneldense_24/biasdense_25/kerneldense_25/biasdense_26/kerneldense_26/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateembedding_14/embeddingsembedding_12/embeddingsembedding_13/embeddingsembedding_17/embeddingsembedding_15/embeddingsembedding_16/embeddings normalize_2/normalization_2/mean$normalize_2/normalization_2/variance!normalize_2/normalization_2/counttotalcountAdam/dense_18/kernel/mAdam/dense_18/bias/mAdam/dense_19/kernel/mAdam/dense_19/bias/mAdam/dense_21/kernel/mAdam/dense_21/bias/mAdam/dense_20/kernel/mAdam/dense_20/bias/mAdam/dense_22/kernel/mAdam/dense_22/bias/mAdam/dense_23/kernel/mAdam/dense_23/bias/mAdam/dense_24/kernel/mAdam/dense_24/bias/mAdam/dense_25/kernel/mAdam/dense_25/bias/mAdam/dense_26/kernel/mAdam/dense_26/bias/mAdam/embedding_14/embeddings/mAdam/embedding_12/embeddings/mAdam/embedding_13/embeddings/mAdam/embedding_17/embeddings/mAdam/embedding_15/embeddings/mAdam/embedding_16/embeddings/mAdam/dense_18/kernel/vAdam/dense_18/bias/vAdam/dense_19/kernel/vAdam/dense_19/bias/vAdam/dense_21/kernel/vAdam/dense_21/bias/vAdam/dense_20/kernel/vAdam/dense_20/bias/vAdam/dense_22/kernel/vAdam/dense_22/bias/vAdam/dense_23/kernel/vAdam/dense_23/bias/vAdam/dense_24/kernel/vAdam/dense_24/bias/vAdam/dense_25/kernel/vAdam/dense_25/bias/vAdam/dense_26/kernel/vAdam/dense_26/bias/vAdam/embedding_14/embeddings/vAdam/embedding_12/embeddings/vAdam/embedding_13/embeddings/vAdam/embedding_17/embeddings/vAdam/embedding_15/embeddings/vAdam/embedding_16/embeddings/v*^
TinW
U2S*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference__traced_restore_200073636½
Î-
Ú
F__inference_model_5_layer_call_and_return_conditional_losses_200071167

inputs*
&tf_math_greater_equal_7_greaterequal_y
embedding_17_200071149
embedding_15_200071152
embedding_16_200071157
identity¢$embedding_15/StatefulPartitionedCall¢$embedding_16/StatefulPartitionedCall¢$embedding_17/StatefulPartitionedCallÚ
flatten_5/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_flatten_5_layer_call_and_return_conditional_losses_2000710072
flatten_5/PartitionedCall
*tf.clip_by_value_7/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2,
*tf.clip_by_value_7/clip_by_value/Minimum/yê
(tf.clip_by_value_7/clip_by_value/MinimumMinimum"flatten_5/PartitionedCall:output:03tf.clip_by_value_7/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(tf.clip_by_value_7/clip_by_value/Minimum
"tf.clip_by_value_7/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_7/clip_by_value/yÜ
 tf.clip_by_value_7/clip_by_valueMaximum,tf.clip_by_value_7/clip_by_value/Minimum:z:0+tf.clip_by_value_7/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 tf.clip_by_value_7/clip_by_value
#tf.compat.v1.floor_div_5/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2%
#tf.compat.v1.floor_div_5/FloorDiv/yØ
!tf.compat.v1.floor_div_5/FloorDivFloorDiv$tf.clip_by_value_7/clip_by_value:z:0,tf.compat.v1.floor_div_5/FloorDiv/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!tf.compat.v1.floor_div_5/FloorDivÚ
$tf.math.greater_equal_7/GreaterEqualGreaterEqual"flatten_5/PartitionedCall:output:0&tf_math_greater_equal_7_greaterequal_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$tf.math.greater_equal_7/GreaterEqual
tf.math.floormod_5/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2
tf.math.floormod_5/FloorMod/yÆ
tf.math.floormod_5/FloorModFloorMod$tf.clip_by_value_7/clip_by_value:z:0&tf.math.floormod_5/FloorMod/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.floormod_5/FloorModº
$embedding_17/StatefulPartitionedCallStatefulPartitionedCall$tf.clip_by_value_7/clip_by_value:z:0embedding_17_200071149*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_embedding_17_layer_call_and_return_conditional_losses_2000710352&
$embedding_17/StatefulPartitionedCall»
$embedding_15/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.floor_div_5/FloorDiv:z:0embedding_15_200071152*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_embedding_15_layer_call_and_return_conditional_losses_2000710572&
$embedding_15/StatefulPartitionedCall
tf.cast_7/CastCast(tf.math.greater_equal_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.cast_7/CastÜ
tf.__operators__.add_14/AddV2AddV2-embedding_17/StatefulPartitionedCall:output:0-embedding_15/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_14/AddV2µ
$embedding_16/StatefulPartitionedCallStatefulPartitionedCalltf.math.floormod_5/FloorMod:z:0embedding_16_200071157*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_embedding_16_layer_call_and_return_conditional_losses_2000710812&
$embedding_16/StatefulPartitionedCallÐ
tf.__operators__.add_15/AddV2AddV2!tf.__operators__.add_14/AddV2:z:0-embedding_16/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_15/AddV2
tf.expand_dims_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2!
tf.expand_dims_5/ExpandDims/dim¼
tf.expand_dims_5/ExpandDims
ExpandDimstf.cast_7/Cast:y:0(tf.expand_dims_5/ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.expand_dims_5/ExpandDims·
tf.math.multiply_5/MulMul!tf.__operators__.add_15/AddV2:z:0$tf.expand_dims_5/ExpandDims:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.multiply_5/Mul
*tf.math.reduce_sum_5/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_5/Sum/reduction_indices¿
tf.math.reduce_sum_5/SumSumtf.math.multiply_5/Mul:z:03tf.math.reduce_sum_5/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.reduce_sum_5/Sumë
IdentityIdentity!tf.math.reduce_sum_5/Sum:output:0%^embedding_15/StatefulPartitionedCall%^embedding_16/StatefulPartitionedCall%^embedding_17/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: :::2L
$embedding_15/StatefulPartitionedCall$embedding_15/StatefulPartitionedCall2L
$embedding_16/StatefulPartitionedCall$embedding_16/StatefulPartitionedCall2L
$embedding_17/StatefulPartitionedCall$embedding_17/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: 
Ëø
¦
M__inference_custom_model_2_layer_call_and_return_conditional_losses_200072422

inputs_0_0

inputs_0_1
inputs_1*
&tf_math_greater_equal_8_greaterequal_y2
.model_4_tf_math_greater_equal_6_greaterequal_y3
/model_4_embedding_14_embedding_lookup_2000722723
/model_4_embedding_12_embedding_lookup_2000722783
/model_4_embedding_13_embedding_lookup_2000722862
.model_5_tf_math_greater_equal_7_greaterequal_y3
/model_5_embedding_17_embedding_lookup_2000723103
/model_5_embedding_15_embedding_lookup_2000723163
/model_5_embedding_16_embedding_lookup_200072324.
*tf_clip_by_value_8_clip_by_value_minimum_y&
"tf_clip_by_value_8_clip_by_value_y+
'dense_18_matmul_readvariableop_resource,
(dense_18_biasadd_readvariableop_resource+
'dense_21_matmul_readvariableop_resource,
(dense_21_biasadd_readvariableop_resource+
'dense_19_matmul_readvariableop_resource,
(dense_19_biasadd_readvariableop_resource+
'dense_20_matmul_readvariableop_resource,
(dense_20_biasadd_readvariableop_resource+
'dense_22_matmul_readvariableop_resource,
(dense_22_biasadd_readvariableop_resource+
'dense_23_matmul_readvariableop_resource,
(dense_23_biasadd_readvariableop_resource+
'dense_24_matmul_readvariableop_resource,
(dense_24_biasadd_readvariableop_resource+
'dense_25_matmul_readvariableop_resource,
(dense_25_biasadd_readvariableop_resource?
;normalize_2_normalization_2_reshape_readvariableop_resourceA
=normalize_2_normalization_2_reshape_1_readvariableop_resource+
'dense_26_matmul_readvariableop_resource,
(dense_26_biasadd_readvariableop_resource
identity¢dense_18/BiasAdd/ReadVariableOp¢dense_18/MatMul/ReadVariableOp¢dense_19/BiasAdd/ReadVariableOp¢dense_19/MatMul/ReadVariableOp¢dense_20/BiasAdd/ReadVariableOp¢dense_20/MatMul/ReadVariableOp¢dense_21/BiasAdd/ReadVariableOp¢dense_21/MatMul/ReadVariableOp¢dense_22/BiasAdd/ReadVariableOp¢dense_22/MatMul/ReadVariableOp¢dense_23/BiasAdd/ReadVariableOp¢dense_23/MatMul/ReadVariableOp¢dense_24/BiasAdd/ReadVariableOp¢dense_24/MatMul/ReadVariableOp¢dense_25/BiasAdd/ReadVariableOp¢dense_25/MatMul/ReadVariableOp¢dense_26/BiasAdd/ReadVariableOp¢dense_26/MatMul/ReadVariableOp¢%model_4/embedding_12/embedding_lookup¢%model_4/embedding_13/embedding_lookup¢%model_4/embedding_14/embedding_lookup¢%model_5/embedding_15/embedding_lookup¢%model_5/embedding_16/embedding_lookup¢%model_5/embedding_17/embedding_lookup¢2normalize_2/normalization_2/Reshape/ReadVariableOp¢4normalize_2/normalization_2/Reshape_1/ReadVariableOpÀ
$tf.math.greater_equal_8/GreaterEqualGreaterEqualinputs_1&tf_math_greater_equal_8_greaterequal_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2&
$tf.math.greater_equal_8/GreaterEqual
model_4/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
model_4/flatten_4/Const¡
model_4/flatten_4/ReshapeReshape
inputs_0_0 model_4/flatten_4/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_4/flatten_4/Reshape­
2model_4/tf.clip_by_value_6/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI24
2model_4/tf.clip_by_value_6/clip_by_value/Minimum/y
0model_4/tf.clip_by_value_6/clip_by_value/MinimumMinimum"model_4/flatten_4/Reshape:output:0;model_4/tf.clip_by_value_6/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0model_4/tf.clip_by_value_6/clip_by_value/Minimum
*model_4/tf.clip_by_value_6/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*model_4/tf.clip_by_value_6/clip_by_value/yü
(model_4/tf.clip_by_value_6/clip_by_valueMaximum4model_4/tf.clip_by_value_6/clip_by_value/Minimum:z:03model_4/tf.clip_by_value_6/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(model_4/tf.clip_by_value_6/clip_by_value
+model_4/tf.compat.v1.floor_div_4/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2-
+model_4/tf.compat.v1.floor_div_4/FloorDiv/yø
)model_4/tf.compat.v1.floor_div_4/FloorDivFloorDiv,model_4/tf.clip_by_value_6/clip_by_value:z:04model_4/tf.compat.v1.floor_div_4/FloorDiv/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)model_4/tf.compat.v1.floor_div_4/FloorDivò
,model_4/tf.math.greater_equal_6/GreaterEqualGreaterEqual"model_4/flatten_4/Reshape:output:0.model_4_tf_math_greater_equal_6_greaterequal_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,model_4/tf.math.greater_equal_6/GreaterEqual
%model_4/tf.math.floormod_4/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2'
%model_4/tf.math.floormod_4/FloorMod/yæ
#model_4/tf.math.floormod_4/FloorModFloorMod,model_4/tf.clip_by_value_6/clip_by_value:z:0.model_4/tf.math.floormod_4/FloorMod/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#model_4/tf.math.floormod_4/FloorMod­
model_4/embedding_14/CastCast,model_4/tf.clip_by_value_6/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_4/embedding_14/Castí
%model_4/embedding_14/embedding_lookupResourceGather/model_4_embedding_14_embedding_lookup_200072272model_4/embedding_14/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@model_4/embedding_14/embedding_lookup/200072272*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02'
%model_4/embedding_14/embedding_lookupÅ
.model_4/embedding_14/embedding_lookup/IdentityIdentity.model_4/embedding_14/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@model_4/embedding_14/embedding_lookup/200072272*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.model_4/embedding_14/embedding_lookup/Identityà
0model_4/embedding_14/embedding_lookup/Identity_1Identity7model_4/embedding_14/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0model_4/embedding_14/embedding_lookup/Identity_1®
model_4/embedding_12/CastCast-model_4/tf.compat.v1.floor_div_4/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_4/embedding_12/Castí
%model_4/embedding_12/embedding_lookupResourceGather/model_4_embedding_12_embedding_lookup_200072278model_4/embedding_12/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@model_4/embedding_12/embedding_lookup/200072278*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02'
%model_4/embedding_12/embedding_lookupÅ
.model_4/embedding_12/embedding_lookup/IdentityIdentity.model_4/embedding_12/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@model_4/embedding_12/embedding_lookup/200072278*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.model_4/embedding_12/embedding_lookup/Identityà
0model_4/embedding_12/embedding_lookup/Identity_1Identity7model_4/embedding_12/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0model_4/embedding_12/embedding_lookup/Identity_1«
model_4/tf.cast_6/CastCast0model_4/tf.math.greater_equal_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_4/tf.cast_6/Cast
%model_4/tf.__operators__.add_12/AddV2AddV29model_4/embedding_14/embedding_lookup/Identity_1:output:09model_4/embedding_12/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%model_4/tf.__operators__.add_12/AddV2¨
model_4/embedding_13/CastCast'model_4/tf.math.floormod_4/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_4/embedding_13/Castí
%model_4/embedding_13/embedding_lookupResourceGather/model_4_embedding_13_embedding_lookup_200072286model_4/embedding_13/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@model_4/embedding_13/embedding_lookup/200072286*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02'
%model_4/embedding_13/embedding_lookupÅ
.model_4/embedding_13/embedding_lookup/IdentityIdentity.model_4/embedding_13/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@model_4/embedding_13/embedding_lookup/200072286*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.model_4/embedding_13/embedding_lookup/Identityà
0model_4/embedding_13/embedding_lookup/Identity_1Identity7model_4/embedding_13/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0model_4/embedding_13/embedding_lookup/Identity_1ô
%model_4/tf.__operators__.add_13/AddV2AddV2)model_4/tf.__operators__.add_12/AddV2:z:09model_4/embedding_13/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%model_4/tf.__operators__.add_13/AddV2
'model_4/tf.expand_dims_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2)
'model_4/tf.expand_dims_4/ExpandDims/dimÜ
#model_4/tf.expand_dims_4/ExpandDims
ExpandDimsmodel_4/tf.cast_6/Cast:y:00model_4/tf.expand_dims_4/ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#model_4/tf.expand_dims_4/ExpandDims×
model_4/tf.math.multiply_4/MulMul)model_4/tf.__operators__.add_13/AddV2:z:0,model_4/tf.expand_dims_4/ExpandDims:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
model_4/tf.math.multiply_4/Mulª
2model_4/tf.math.reduce_sum_4/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :24
2model_4/tf.math.reduce_sum_4/Sum/reduction_indicesß
 model_4/tf.math.reduce_sum_4/SumSum"model_4/tf.math.multiply_4/Mul:z:0;model_4/tf.math.reduce_sum_4/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 model_4/tf.math.reduce_sum_4/Sum
model_5/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
model_5/flatten_5/Const¡
model_5/flatten_5/ReshapeReshape
inputs_0_1 model_5/flatten_5/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_5/flatten_5/Reshape­
2model_5/tf.clip_by_value_7/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI24
2model_5/tf.clip_by_value_7/clip_by_value/Minimum/y
0model_5/tf.clip_by_value_7/clip_by_value/MinimumMinimum"model_5/flatten_5/Reshape:output:0;model_5/tf.clip_by_value_7/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0model_5/tf.clip_by_value_7/clip_by_value/Minimum
*model_5/tf.clip_by_value_7/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*model_5/tf.clip_by_value_7/clip_by_value/yü
(model_5/tf.clip_by_value_7/clip_by_valueMaximum4model_5/tf.clip_by_value_7/clip_by_value/Minimum:z:03model_5/tf.clip_by_value_7/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(model_5/tf.clip_by_value_7/clip_by_value
+model_5/tf.compat.v1.floor_div_5/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2-
+model_5/tf.compat.v1.floor_div_5/FloorDiv/yø
)model_5/tf.compat.v1.floor_div_5/FloorDivFloorDiv,model_5/tf.clip_by_value_7/clip_by_value:z:04model_5/tf.compat.v1.floor_div_5/FloorDiv/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)model_5/tf.compat.v1.floor_div_5/FloorDivò
,model_5/tf.math.greater_equal_7/GreaterEqualGreaterEqual"model_5/flatten_5/Reshape:output:0.model_5_tf_math_greater_equal_7_greaterequal_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,model_5/tf.math.greater_equal_7/GreaterEqual
%model_5/tf.math.floormod_5/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2'
%model_5/tf.math.floormod_5/FloorMod/yæ
#model_5/tf.math.floormod_5/FloorModFloorMod,model_5/tf.clip_by_value_7/clip_by_value:z:0.model_5/tf.math.floormod_5/FloorMod/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#model_5/tf.math.floormod_5/FloorMod­
model_5/embedding_17/CastCast,model_5/tf.clip_by_value_7/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_5/embedding_17/Castí
%model_5/embedding_17/embedding_lookupResourceGather/model_5_embedding_17_embedding_lookup_200072310model_5/embedding_17/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@model_5/embedding_17/embedding_lookup/200072310*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02'
%model_5/embedding_17/embedding_lookupÅ
.model_5/embedding_17/embedding_lookup/IdentityIdentity.model_5/embedding_17/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@model_5/embedding_17/embedding_lookup/200072310*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.model_5/embedding_17/embedding_lookup/Identityà
0model_5/embedding_17/embedding_lookup/Identity_1Identity7model_5/embedding_17/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0model_5/embedding_17/embedding_lookup/Identity_1®
model_5/embedding_15/CastCast-model_5/tf.compat.v1.floor_div_5/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_5/embedding_15/Castí
%model_5/embedding_15/embedding_lookupResourceGather/model_5_embedding_15_embedding_lookup_200072316model_5/embedding_15/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@model_5/embedding_15/embedding_lookup/200072316*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02'
%model_5/embedding_15/embedding_lookupÅ
.model_5/embedding_15/embedding_lookup/IdentityIdentity.model_5/embedding_15/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@model_5/embedding_15/embedding_lookup/200072316*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.model_5/embedding_15/embedding_lookup/Identityà
0model_5/embedding_15/embedding_lookup/Identity_1Identity7model_5/embedding_15/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0model_5/embedding_15/embedding_lookup/Identity_1«
model_5/tf.cast_7/CastCast0model_5/tf.math.greater_equal_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_5/tf.cast_7/Cast
%model_5/tf.__operators__.add_14/AddV2AddV29model_5/embedding_17/embedding_lookup/Identity_1:output:09model_5/embedding_15/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%model_5/tf.__operators__.add_14/AddV2¨
model_5/embedding_16/CastCast'model_5/tf.math.floormod_5/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_5/embedding_16/Castí
%model_5/embedding_16/embedding_lookupResourceGather/model_5_embedding_16_embedding_lookup_200072324model_5/embedding_16/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@model_5/embedding_16/embedding_lookup/200072324*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02'
%model_5/embedding_16/embedding_lookupÅ
.model_5/embedding_16/embedding_lookup/IdentityIdentity.model_5/embedding_16/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@model_5/embedding_16/embedding_lookup/200072324*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.model_5/embedding_16/embedding_lookup/Identityà
0model_5/embedding_16/embedding_lookup/Identity_1Identity7model_5/embedding_16/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0model_5/embedding_16/embedding_lookup/Identity_1ô
%model_5/tf.__operators__.add_15/AddV2AddV2)model_5/tf.__operators__.add_14/AddV2:z:09model_5/embedding_16/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%model_5/tf.__operators__.add_15/AddV2
'model_5/tf.expand_dims_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2)
'model_5/tf.expand_dims_5/ExpandDims/dimÜ
#model_5/tf.expand_dims_5/ExpandDims
ExpandDimsmodel_5/tf.cast_7/Cast:y:00model_5/tf.expand_dims_5/ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#model_5/tf.expand_dims_5/ExpandDims×
model_5/tf.math.multiply_5/MulMul)model_5/tf.__operators__.add_15/AddV2:z:0,model_5/tf.expand_dims_5/ExpandDims:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
model_5/tf.math.multiply_5/Mulª
2model_5/tf.math.reduce_sum_5/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :24
2model_5/tf.math.reduce_sum_5/Sum/reduction_indicesß
 model_5/tf.math.reduce_sum_5/SumSum"model_5/tf.math.multiply_5/Mul:z:0;model_5/tf.math.reduce_sum_5/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 model_5/tf.math.reduce_sum_5/SumÇ
(tf.clip_by_value_8/clip_by_value/MinimumMinimuminputs_1*tf_clip_by_value_8_clip_by_value_minimum_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2*
(tf.clip_by_value_8/clip_by_value/MinimumÓ
 tf.clip_by_value_8/clip_by_valueMaximum,tf.clip_by_value_8/clip_by_value/Minimum:z:0"tf_clip_by_value_8_clip_by_value_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2"
 tf.clip_by_value_8/clip_by_value
tf.cast_8/CastCast(tf.math.greater_equal_8/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
tf.cast_8/Castt
tf.concat_6/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_6/concat/axisè
tf.concat_6/concatConcatV2)model_4/tf.math.reduce_sum_4/Sum:output:0)model_5/tf.math.reduce_sum_5/Sum:output:0 tf.concat_6/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.concat_6/concat}
tf.concat_7/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
tf.concat_7/concat/axisË
tf.concat_7/concatConcatV2$tf.clip_by_value_8/clip_by_value:z:0tf.cast_8/Cast:y:0 tf.concat_7/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.concat_7/concatª
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_18/MatMul/ReadVariableOp¤
dense_18/MatMulMatMultf.concat_6/concat:output:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_18/MatMul¨
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_18/BiasAdd/ReadVariableOp¦
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_18/BiasAddt
dense_18/ReluReludense_18/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_18/Relu©
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02 
dense_21/MatMul/ReadVariableOp¤
dense_21/MatMulMatMultf.concat_7/concat:output:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_21/MatMul¨
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_21/BiasAdd/ReadVariableOp¦
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_21/BiasAddª
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_19/MatMul/ReadVariableOp¤
dense_19/MatMulMatMuldense_18/Relu:activations:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_19/MatMul¨
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_19/BiasAdd/ReadVariableOp¦
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_19/BiasAddt
dense_19/ReluReludense_19/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_19/Reluª
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_20/MatMul/ReadVariableOp¤
dense_20/MatMulMatMuldense_19/Relu:activations:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_20/MatMul¨
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_20/BiasAdd/ReadVariableOp¦
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_20/BiasAddt
dense_20/ReluReludense_20/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_20/Reluª
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_22/MatMul/ReadVariableOp¢
dense_22/MatMulMatMuldense_21/BiasAdd:output:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_22/MatMul¨
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_22/BiasAdd/ReadVariableOp¦
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_22/BiasAdd}
tf.concat_8/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
tf.concat_8/concat/axisÊ
tf.concat_8/concatConcatV2dense_20/Relu:activations:0dense_22/BiasAdd:output:0 tf.concat_8/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.concat_8/concatª
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_23/MatMul/ReadVariableOp¤
dense_23/MatMulMatMultf.concat_8/concat:output:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_23/MatMul¨
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_23/BiasAdd/ReadVariableOp¦
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_23/BiasAdd|
tf.nn.relu_6/ReluReludense_23/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.nn.relu_6/Reluª
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_24/MatMul/ReadVariableOp¨
dense_24/MatMulMatMultf.nn.relu_6/Relu:activations:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_24/MatMul¨
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_24/BiasAdd/ReadVariableOp¦
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_24/BiasAdd¶
tf.__operators__.add_16/AddV2AddV2dense_24/BiasAdd:output:0tf.nn.relu_6/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_16/AddV2
tf.nn.relu_7/ReluRelu!tf.__operators__.add_16/AddV2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.nn.relu_7/Reluª
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_25/MatMul/ReadVariableOp¨
dense_25/MatMulMatMultf.nn.relu_7/Relu:activations:0&dense_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_25/MatMul¨
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_25/BiasAdd/ReadVariableOp¦
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_25/BiasAdd¶
tf.__operators__.add_17/AddV2AddV2dense_25/BiasAdd:output:0tf.nn.relu_7/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_17/AddV2
tf.nn.relu_8/ReluRelu!tf.__operators__.add_17/AddV2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.nn.relu_8/Reluá
2normalize_2/normalization_2/Reshape/ReadVariableOpReadVariableOp;normalize_2_normalization_2_reshape_readvariableop_resource*
_output_shapes	
:*
dtype024
2normalize_2/normalization_2/Reshape/ReadVariableOp§
)normalize_2/normalization_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2+
)normalize_2/normalization_2/Reshape/shapeï
#normalize_2/normalization_2/ReshapeReshape:normalize_2/normalization_2/Reshape/ReadVariableOp:value:02normalize_2/normalization_2/Reshape/shape:output:0*
T0*
_output_shapes
:	2%
#normalize_2/normalization_2/Reshapeç
4normalize_2/normalization_2/Reshape_1/ReadVariableOpReadVariableOp=normalize_2_normalization_2_reshape_1_readvariableop_resource*
_output_shapes	
:*
dtype026
4normalize_2/normalization_2/Reshape_1/ReadVariableOp«
+normalize_2/normalization_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2-
+normalize_2/normalization_2/Reshape_1/shape÷
%normalize_2/normalization_2/Reshape_1Reshape<normalize_2/normalization_2/Reshape_1/ReadVariableOp:value:04normalize_2/normalization_2/Reshape_1/shape:output:0*
T0*
_output_shapes
:	2'
%normalize_2/normalization_2/Reshape_1Ë
normalize_2/normalization_2/subSubtf.nn.relu_8/Relu:activations:0,normalize_2/normalization_2/Reshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
normalize_2/normalization_2/sub¦
 normalize_2/normalization_2/SqrtSqrt.normalize_2/normalization_2/Reshape_1:output:0*
T0*
_output_shapes
:	2"
 normalize_2/normalization_2/Sqrt
%normalize_2/normalization_2/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö32'
%normalize_2/normalization_2/Maximum/yÕ
#normalize_2/normalization_2/MaximumMaximum$normalize_2/normalization_2/Sqrt:y:0.normalize_2/normalization_2/Maximum/y:output:0*
T0*
_output_shapes
:	2%
#normalize_2/normalization_2/MaximumÖ
#normalize_2/normalization_2/truedivRealDiv#normalize_2/normalization_2/sub:z:0'normalize_2/normalization_2/Maximum:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#normalize_2/normalization_2/truediv©
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02 
dense_26/MatMul/ReadVariableOp¯
dense_26/MatMulMatMul'normalize_2/normalization_2/truediv:z:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_26/MatMul§
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_26/BiasAdd/ReadVariableOp¥
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_26/BiasAdd¤
IdentityIdentitydense_26/BiasAdd:output:0 ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp ^dense_26/BiasAdd/ReadVariableOp^dense_26/MatMul/ReadVariableOp&^model_4/embedding_12/embedding_lookup&^model_4/embedding_13/embedding_lookup&^model_4/embedding_14/embedding_lookup&^model_5/embedding_15/embedding_lookup&^model_5/embedding_16/embedding_lookup&^model_5/embedding_17/embedding_lookup3^normalize_2/normalization_2/Reshape/ReadVariableOp5^normalize_2/normalization_2/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*À
_input_shapes®
«:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
: : :::: :::: : ::::::::::::::::::::2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2@
dense_25/MatMul/ReadVariableOpdense_25/MatMul/ReadVariableOp2B
dense_26/BiasAdd/ReadVariableOpdense_26/BiasAdd/ReadVariableOp2@
dense_26/MatMul/ReadVariableOpdense_26/MatMul/ReadVariableOp2N
%model_4/embedding_12/embedding_lookup%model_4/embedding_12/embedding_lookup2N
%model_4/embedding_13/embedding_lookup%model_4/embedding_13/embedding_lookup2N
%model_4/embedding_14/embedding_lookup%model_4/embedding_14/embedding_lookup2N
%model_5/embedding_15/embedding_lookup%model_5/embedding_15/embedding_lookup2N
%model_5/embedding_16/embedding_lookup%model_5/embedding_16/embedding_lookup2N
%model_5/embedding_17/embedding_lookup%model_5/embedding_17/embedding_lookup2h
2normalize_2/normalization_2/Reshape/ReadVariableOp2normalize_2/normalization_2/Reshape/ReadVariableOp2l
4normalize_2/normalization_2/Reshape_1/ReadVariableOp4normalize_2/normalization_2/Reshape_1/ReadVariableOp:S O
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/0/0:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/0/1:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
¼º
È
$__inference__wrapped_model_200070771

cards0

cards1
bets9
5custom_model_2_tf_math_greater_equal_8_greaterequal_yA
=custom_model_2_model_4_tf_math_greater_equal_6_greaterequal_yB
>custom_model_2_model_4_embedding_14_embedding_lookup_200070621B
>custom_model_2_model_4_embedding_12_embedding_lookup_200070627B
>custom_model_2_model_4_embedding_13_embedding_lookup_200070635A
=custom_model_2_model_5_tf_math_greater_equal_7_greaterequal_yB
>custom_model_2_model_5_embedding_17_embedding_lookup_200070659B
>custom_model_2_model_5_embedding_15_embedding_lookup_200070665B
>custom_model_2_model_5_embedding_16_embedding_lookup_200070673=
9custom_model_2_tf_clip_by_value_8_clip_by_value_minimum_y5
1custom_model_2_tf_clip_by_value_8_clip_by_value_y:
6custom_model_2_dense_18_matmul_readvariableop_resource;
7custom_model_2_dense_18_biasadd_readvariableop_resource:
6custom_model_2_dense_21_matmul_readvariableop_resource;
7custom_model_2_dense_21_biasadd_readvariableop_resource:
6custom_model_2_dense_19_matmul_readvariableop_resource;
7custom_model_2_dense_19_biasadd_readvariableop_resource:
6custom_model_2_dense_20_matmul_readvariableop_resource;
7custom_model_2_dense_20_biasadd_readvariableop_resource:
6custom_model_2_dense_22_matmul_readvariableop_resource;
7custom_model_2_dense_22_biasadd_readvariableop_resource:
6custom_model_2_dense_23_matmul_readvariableop_resource;
7custom_model_2_dense_23_biasadd_readvariableop_resource:
6custom_model_2_dense_24_matmul_readvariableop_resource;
7custom_model_2_dense_24_biasadd_readvariableop_resource:
6custom_model_2_dense_25_matmul_readvariableop_resource;
7custom_model_2_dense_25_biasadd_readvariableop_resourceN
Jcustom_model_2_normalize_2_normalization_2_reshape_readvariableop_resourceP
Lcustom_model_2_normalize_2_normalization_2_reshape_1_readvariableop_resource:
6custom_model_2_dense_26_matmul_readvariableop_resource;
7custom_model_2_dense_26_biasadd_readvariableop_resource
identity¢.custom_model_2/dense_18/BiasAdd/ReadVariableOp¢-custom_model_2/dense_18/MatMul/ReadVariableOp¢.custom_model_2/dense_19/BiasAdd/ReadVariableOp¢-custom_model_2/dense_19/MatMul/ReadVariableOp¢.custom_model_2/dense_20/BiasAdd/ReadVariableOp¢-custom_model_2/dense_20/MatMul/ReadVariableOp¢.custom_model_2/dense_21/BiasAdd/ReadVariableOp¢-custom_model_2/dense_21/MatMul/ReadVariableOp¢.custom_model_2/dense_22/BiasAdd/ReadVariableOp¢-custom_model_2/dense_22/MatMul/ReadVariableOp¢.custom_model_2/dense_23/BiasAdd/ReadVariableOp¢-custom_model_2/dense_23/MatMul/ReadVariableOp¢.custom_model_2/dense_24/BiasAdd/ReadVariableOp¢-custom_model_2/dense_24/MatMul/ReadVariableOp¢.custom_model_2/dense_25/BiasAdd/ReadVariableOp¢-custom_model_2/dense_25/MatMul/ReadVariableOp¢.custom_model_2/dense_26/BiasAdd/ReadVariableOp¢-custom_model_2/dense_26/MatMul/ReadVariableOp¢4custom_model_2/model_4/embedding_12/embedding_lookup¢4custom_model_2/model_4/embedding_13/embedding_lookup¢4custom_model_2/model_4/embedding_14/embedding_lookup¢4custom_model_2/model_5/embedding_15/embedding_lookup¢4custom_model_2/model_5/embedding_16/embedding_lookup¢4custom_model_2/model_5/embedding_17/embedding_lookup¢Acustom_model_2/normalize_2/normalization_2/Reshape/ReadVariableOp¢Ccustom_model_2/normalize_2/normalization_2/Reshape_1/ReadVariableOpé
3custom_model_2/tf.math.greater_equal_8/GreaterEqualGreaterEqualbets5custom_model_2_tf_math_greater_equal_8_greaterequal_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
25
3custom_model_2/tf.math.greater_equal_8/GreaterEqual¡
&custom_model_2/model_4/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2(
&custom_model_2/model_4/flatten_4/ConstÊ
(custom_model_2/model_4/flatten_4/ReshapeReshapecards0/custom_model_2/model_4/flatten_4/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(custom_model_2/model_4/flatten_4/ReshapeË
Acustom_model_2/model_4/tf.clip_by_value_6/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2C
Acustom_model_2/model_4/tf.clip_by_value_6/clip_by_value/Minimum/y¾
?custom_model_2/model_4/tf.clip_by_value_6/clip_by_value/MinimumMinimum1custom_model_2/model_4/flatten_4/Reshape:output:0Jcustom_model_2/model_4/tf.clip_by_value_6/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2A
?custom_model_2/model_4/tf.clip_by_value_6/clip_by_value/Minimum»
9custom_model_2/model_4/tf.clip_by_value_6/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2;
9custom_model_2/model_4/tf.clip_by_value_6/clip_by_value/y¸
7custom_model_2/model_4/tf.clip_by_value_6/clip_by_valueMaximumCcustom_model_2/model_4/tf.clip_by_value_6/clip_by_value/Minimum:z:0Bcustom_model_2/model_4/tf.clip_by_value_6/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ29
7custom_model_2/model_4/tf.clip_by_value_6/clip_by_value½
:custom_model_2/model_4/tf.compat.v1.floor_div_4/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2<
:custom_model_2/model_4/tf.compat.v1.floor_div_4/FloorDiv/y´
8custom_model_2/model_4/tf.compat.v1.floor_div_4/FloorDivFloorDiv;custom_model_2/model_4/tf.clip_by_value_6/clip_by_value:z:0Ccustom_model_2/model_4/tf.compat.v1.floor_div_4/FloorDiv/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:
8custom_model_2/model_4/tf.compat.v1.floor_div_4/FloorDiv®
;custom_model_2/model_4/tf.math.greater_equal_6/GreaterEqualGreaterEqual1custom_model_2/model_4/flatten_4/Reshape:output:0=custom_model_2_model_4_tf_math_greater_equal_6_greaterequal_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2=
;custom_model_2/model_4/tf.math.greater_equal_6/GreaterEqual±
4custom_model_2/model_4/tf.math.floormod_4/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @26
4custom_model_2/model_4/tf.math.floormod_4/FloorMod/y¢
2custom_model_2/model_4/tf.math.floormod_4/FloorModFloorMod;custom_model_2/model_4/tf.clip_by_value_6/clip_by_value:z:0=custom_model_2/model_4/tf.math.floormod_4/FloorMod/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2custom_model_2/model_4/tf.math.floormod_4/FloorModÚ
(custom_model_2/model_4/embedding_14/CastCast;custom_model_2/model_4/tf.clip_by_value_6/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(custom_model_2/model_4/embedding_14/Cast¸
4custom_model_2/model_4/embedding_14/embedding_lookupResourceGather>custom_model_2_model_4_embedding_14_embedding_lookup_200070621,custom_model_2/model_4/embedding_14/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*Q
_classG
ECloc:@custom_model_2/model_4/embedding_14/embedding_lookup/200070621*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype026
4custom_model_2/model_4/embedding_14/embedding_lookup
=custom_model_2/model_4/embedding_14/embedding_lookup/IdentityIdentity=custom_model_2/model_4/embedding_14/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*Q
_classG
ECloc:@custom_model_2/model_4/embedding_14/embedding_lookup/200070621*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2?
=custom_model_2/model_4/embedding_14/embedding_lookup/Identity
?custom_model_2/model_4/embedding_14/embedding_lookup/Identity_1IdentityFcustom_model_2/model_4/embedding_14/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2A
?custom_model_2/model_4/embedding_14/embedding_lookup/Identity_1Û
(custom_model_2/model_4/embedding_12/CastCast<custom_model_2/model_4/tf.compat.v1.floor_div_4/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(custom_model_2/model_4/embedding_12/Cast¸
4custom_model_2/model_4/embedding_12/embedding_lookupResourceGather>custom_model_2_model_4_embedding_12_embedding_lookup_200070627,custom_model_2/model_4/embedding_12/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*Q
_classG
ECloc:@custom_model_2/model_4/embedding_12/embedding_lookup/200070627*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype026
4custom_model_2/model_4/embedding_12/embedding_lookup
=custom_model_2/model_4/embedding_12/embedding_lookup/IdentityIdentity=custom_model_2/model_4/embedding_12/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*Q
_classG
ECloc:@custom_model_2/model_4/embedding_12/embedding_lookup/200070627*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2?
=custom_model_2/model_4/embedding_12/embedding_lookup/Identity
?custom_model_2/model_4/embedding_12/embedding_lookup/Identity_1IdentityFcustom_model_2/model_4/embedding_12/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2A
?custom_model_2/model_4/embedding_12/embedding_lookup/Identity_1Ø
%custom_model_2/model_4/tf.cast_6/CastCast?custom_model_2/model_4/tf.math.greater_equal_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%custom_model_2/model_4/tf.cast_6/CastÀ
4custom_model_2/model_4/tf.__operators__.add_12/AddV2AddV2Hcustom_model_2/model_4/embedding_14/embedding_lookup/Identity_1:output:0Hcustom_model_2/model_4/embedding_12/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ26
4custom_model_2/model_4/tf.__operators__.add_12/AddV2Õ
(custom_model_2/model_4/embedding_13/CastCast6custom_model_2/model_4/tf.math.floormod_4/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(custom_model_2/model_4/embedding_13/Cast¸
4custom_model_2/model_4/embedding_13/embedding_lookupResourceGather>custom_model_2_model_4_embedding_13_embedding_lookup_200070635,custom_model_2/model_4/embedding_13/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*Q
_classG
ECloc:@custom_model_2/model_4/embedding_13/embedding_lookup/200070635*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype026
4custom_model_2/model_4/embedding_13/embedding_lookup
=custom_model_2/model_4/embedding_13/embedding_lookup/IdentityIdentity=custom_model_2/model_4/embedding_13/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*Q
_classG
ECloc:@custom_model_2/model_4/embedding_13/embedding_lookup/200070635*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2?
=custom_model_2/model_4/embedding_13/embedding_lookup/Identity
?custom_model_2/model_4/embedding_13/embedding_lookup/Identity_1IdentityFcustom_model_2/model_4/embedding_13/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2A
?custom_model_2/model_4/embedding_13/embedding_lookup/Identity_1°
4custom_model_2/model_4/tf.__operators__.add_13/AddV2AddV28custom_model_2/model_4/tf.__operators__.add_12/AddV2:z:0Hcustom_model_2/model_4/embedding_13/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ26
4custom_model_2/model_4/tf.__operators__.add_13/AddV2»
6custom_model_2/model_4/tf.expand_dims_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ28
6custom_model_2/model_4/tf.expand_dims_4/ExpandDims/dim
2custom_model_2/model_4/tf.expand_dims_4/ExpandDims
ExpandDims)custom_model_2/model_4/tf.cast_6/Cast:y:0?custom_model_2/model_4/tf.expand_dims_4/ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2custom_model_2/model_4/tf.expand_dims_4/ExpandDims
-custom_model_2/model_4/tf.math.multiply_4/MulMul8custom_model_2/model_4/tf.__operators__.add_13/AddV2:z:0;custom_model_2/model_4/tf.expand_dims_4/ExpandDims:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-custom_model_2/model_4/tf.math.multiply_4/MulÈ
Acustom_model_2/model_4/tf.math.reduce_sum_4/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2C
Acustom_model_2/model_4/tf.math.reduce_sum_4/Sum/reduction_indices
/custom_model_2/model_4/tf.math.reduce_sum_4/SumSum1custom_model_2/model_4/tf.math.multiply_4/Mul:z:0Jcustom_model_2/model_4/tf.math.reduce_sum_4/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/custom_model_2/model_4/tf.math.reduce_sum_4/Sum¡
&custom_model_2/model_5/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2(
&custom_model_2/model_5/flatten_5/ConstÊ
(custom_model_2/model_5/flatten_5/ReshapeReshapecards1/custom_model_2/model_5/flatten_5/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(custom_model_2/model_5/flatten_5/ReshapeË
Acustom_model_2/model_5/tf.clip_by_value_7/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2C
Acustom_model_2/model_5/tf.clip_by_value_7/clip_by_value/Minimum/y¾
?custom_model_2/model_5/tf.clip_by_value_7/clip_by_value/MinimumMinimum1custom_model_2/model_5/flatten_5/Reshape:output:0Jcustom_model_2/model_5/tf.clip_by_value_7/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2A
?custom_model_2/model_5/tf.clip_by_value_7/clip_by_value/Minimum»
9custom_model_2/model_5/tf.clip_by_value_7/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2;
9custom_model_2/model_5/tf.clip_by_value_7/clip_by_value/y¸
7custom_model_2/model_5/tf.clip_by_value_7/clip_by_valueMaximumCcustom_model_2/model_5/tf.clip_by_value_7/clip_by_value/Minimum:z:0Bcustom_model_2/model_5/tf.clip_by_value_7/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ29
7custom_model_2/model_5/tf.clip_by_value_7/clip_by_value½
:custom_model_2/model_5/tf.compat.v1.floor_div_5/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2<
:custom_model_2/model_5/tf.compat.v1.floor_div_5/FloorDiv/y´
8custom_model_2/model_5/tf.compat.v1.floor_div_5/FloorDivFloorDiv;custom_model_2/model_5/tf.clip_by_value_7/clip_by_value:z:0Ccustom_model_2/model_5/tf.compat.v1.floor_div_5/FloorDiv/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:
8custom_model_2/model_5/tf.compat.v1.floor_div_5/FloorDiv®
;custom_model_2/model_5/tf.math.greater_equal_7/GreaterEqualGreaterEqual1custom_model_2/model_5/flatten_5/Reshape:output:0=custom_model_2_model_5_tf_math_greater_equal_7_greaterequal_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2=
;custom_model_2/model_5/tf.math.greater_equal_7/GreaterEqual±
4custom_model_2/model_5/tf.math.floormod_5/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @26
4custom_model_2/model_5/tf.math.floormod_5/FloorMod/y¢
2custom_model_2/model_5/tf.math.floormod_5/FloorModFloorMod;custom_model_2/model_5/tf.clip_by_value_7/clip_by_value:z:0=custom_model_2/model_5/tf.math.floormod_5/FloorMod/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2custom_model_2/model_5/tf.math.floormod_5/FloorModÚ
(custom_model_2/model_5/embedding_17/CastCast;custom_model_2/model_5/tf.clip_by_value_7/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(custom_model_2/model_5/embedding_17/Cast¸
4custom_model_2/model_5/embedding_17/embedding_lookupResourceGather>custom_model_2_model_5_embedding_17_embedding_lookup_200070659,custom_model_2/model_5/embedding_17/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*Q
_classG
ECloc:@custom_model_2/model_5/embedding_17/embedding_lookup/200070659*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype026
4custom_model_2/model_5/embedding_17/embedding_lookup
=custom_model_2/model_5/embedding_17/embedding_lookup/IdentityIdentity=custom_model_2/model_5/embedding_17/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*Q
_classG
ECloc:@custom_model_2/model_5/embedding_17/embedding_lookup/200070659*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2?
=custom_model_2/model_5/embedding_17/embedding_lookup/Identity
?custom_model_2/model_5/embedding_17/embedding_lookup/Identity_1IdentityFcustom_model_2/model_5/embedding_17/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2A
?custom_model_2/model_5/embedding_17/embedding_lookup/Identity_1Û
(custom_model_2/model_5/embedding_15/CastCast<custom_model_2/model_5/tf.compat.v1.floor_div_5/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(custom_model_2/model_5/embedding_15/Cast¸
4custom_model_2/model_5/embedding_15/embedding_lookupResourceGather>custom_model_2_model_5_embedding_15_embedding_lookup_200070665,custom_model_2/model_5/embedding_15/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*Q
_classG
ECloc:@custom_model_2/model_5/embedding_15/embedding_lookup/200070665*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype026
4custom_model_2/model_5/embedding_15/embedding_lookup
=custom_model_2/model_5/embedding_15/embedding_lookup/IdentityIdentity=custom_model_2/model_5/embedding_15/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*Q
_classG
ECloc:@custom_model_2/model_5/embedding_15/embedding_lookup/200070665*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2?
=custom_model_2/model_5/embedding_15/embedding_lookup/Identity
?custom_model_2/model_5/embedding_15/embedding_lookup/Identity_1IdentityFcustom_model_2/model_5/embedding_15/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2A
?custom_model_2/model_5/embedding_15/embedding_lookup/Identity_1Ø
%custom_model_2/model_5/tf.cast_7/CastCast?custom_model_2/model_5/tf.math.greater_equal_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%custom_model_2/model_5/tf.cast_7/CastÀ
4custom_model_2/model_5/tf.__operators__.add_14/AddV2AddV2Hcustom_model_2/model_5/embedding_17/embedding_lookup/Identity_1:output:0Hcustom_model_2/model_5/embedding_15/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ26
4custom_model_2/model_5/tf.__operators__.add_14/AddV2Õ
(custom_model_2/model_5/embedding_16/CastCast6custom_model_2/model_5/tf.math.floormod_5/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(custom_model_2/model_5/embedding_16/Cast¸
4custom_model_2/model_5/embedding_16/embedding_lookupResourceGather>custom_model_2_model_5_embedding_16_embedding_lookup_200070673,custom_model_2/model_5/embedding_16/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*Q
_classG
ECloc:@custom_model_2/model_5/embedding_16/embedding_lookup/200070673*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype026
4custom_model_2/model_5/embedding_16/embedding_lookup
=custom_model_2/model_5/embedding_16/embedding_lookup/IdentityIdentity=custom_model_2/model_5/embedding_16/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*Q
_classG
ECloc:@custom_model_2/model_5/embedding_16/embedding_lookup/200070673*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2?
=custom_model_2/model_5/embedding_16/embedding_lookup/Identity
?custom_model_2/model_5/embedding_16/embedding_lookup/Identity_1IdentityFcustom_model_2/model_5/embedding_16/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2A
?custom_model_2/model_5/embedding_16/embedding_lookup/Identity_1°
4custom_model_2/model_5/tf.__operators__.add_15/AddV2AddV28custom_model_2/model_5/tf.__operators__.add_14/AddV2:z:0Hcustom_model_2/model_5/embedding_16/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ26
4custom_model_2/model_5/tf.__operators__.add_15/AddV2»
6custom_model_2/model_5/tf.expand_dims_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ28
6custom_model_2/model_5/tf.expand_dims_5/ExpandDims/dim
2custom_model_2/model_5/tf.expand_dims_5/ExpandDims
ExpandDims)custom_model_2/model_5/tf.cast_7/Cast:y:0?custom_model_2/model_5/tf.expand_dims_5/ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2custom_model_2/model_5/tf.expand_dims_5/ExpandDims
-custom_model_2/model_5/tf.math.multiply_5/MulMul8custom_model_2/model_5/tf.__operators__.add_15/AddV2:z:0;custom_model_2/model_5/tf.expand_dims_5/ExpandDims:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-custom_model_2/model_5/tf.math.multiply_5/MulÈ
Acustom_model_2/model_5/tf.math.reduce_sum_5/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2C
Acustom_model_2/model_5/tf.math.reduce_sum_5/Sum/reduction_indices
/custom_model_2/model_5/tf.math.reduce_sum_5/SumSum1custom_model_2/model_5/tf.math.multiply_5/Mul:z:0Jcustom_model_2/model_5/tf.math.reduce_sum_5/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/custom_model_2/model_5/tf.math.reduce_sum_5/Sumð
7custom_model_2/tf.clip_by_value_8/clip_by_value/MinimumMinimumbets9custom_model_2_tf_clip_by_value_8_clip_by_value_minimum_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
29
7custom_model_2/tf.clip_by_value_8/clip_by_value/Minimum
/custom_model_2/tf.clip_by_value_8/clip_by_valueMaximum;custom_model_2/tf.clip_by_value_8/clip_by_value/Minimum:z:01custom_model_2_tf_clip_by_value_8_clip_by_value_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
21
/custom_model_2/tf.clip_by_value_8/clip_by_valueÀ
custom_model_2/tf.cast_8/CastCast7custom_model_2/tf.math.greater_equal_8/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
custom_model_2/tf.cast_8/Cast
&custom_model_2/tf.concat_6/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2(
&custom_model_2/tf.concat_6/concat/axis³
!custom_model_2/tf.concat_6/concatConcatV28custom_model_2/model_4/tf.math.reduce_sum_4/Sum:output:08custom_model_2/model_5/tf.math.reduce_sum_5/Sum:output:0/custom_model_2/tf.concat_6/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!custom_model_2/tf.concat_6/concat
&custom_model_2/tf.concat_7/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&custom_model_2/tf.concat_7/concat/axis
!custom_model_2/tf.concat_7/concatConcatV23custom_model_2/tf.clip_by_value_8/clip_by_value:z:0!custom_model_2/tf.cast_8/Cast:y:0/custom_model_2/tf.concat_7/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!custom_model_2/tf.concat_7/concat×
-custom_model_2/dense_18/MatMul/ReadVariableOpReadVariableOp6custom_model_2_dense_18_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02/
-custom_model_2/dense_18/MatMul/ReadVariableOpà
custom_model_2/dense_18/MatMulMatMul*custom_model_2/tf.concat_6/concat:output:05custom_model_2/dense_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
custom_model_2/dense_18/MatMulÕ
.custom_model_2/dense_18/BiasAdd/ReadVariableOpReadVariableOp7custom_model_2_dense_18_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.custom_model_2/dense_18/BiasAdd/ReadVariableOpâ
custom_model_2/dense_18/BiasAddBiasAdd(custom_model_2/dense_18/MatMul:product:06custom_model_2/dense_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
custom_model_2/dense_18/BiasAdd¡
custom_model_2/dense_18/ReluRelu(custom_model_2/dense_18/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
custom_model_2/dense_18/ReluÖ
-custom_model_2/dense_21/MatMul/ReadVariableOpReadVariableOp6custom_model_2_dense_21_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02/
-custom_model_2/dense_21/MatMul/ReadVariableOpà
custom_model_2/dense_21/MatMulMatMul*custom_model_2/tf.concat_7/concat:output:05custom_model_2/dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
custom_model_2/dense_21/MatMulÕ
.custom_model_2/dense_21/BiasAdd/ReadVariableOpReadVariableOp7custom_model_2_dense_21_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.custom_model_2/dense_21/BiasAdd/ReadVariableOpâ
custom_model_2/dense_21/BiasAddBiasAdd(custom_model_2/dense_21/MatMul:product:06custom_model_2/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
custom_model_2/dense_21/BiasAdd×
-custom_model_2/dense_19/MatMul/ReadVariableOpReadVariableOp6custom_model_2_dense_19_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02/
-custom_model_2/dense_19/MatMul/ReadVariableOpà
custom_model_2/dense_19/MatMulMatMul*custom_model_2/dense_18/Relu:activations:05custom_model_2/dense_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
custom_model_2/dense_19/MatMulÕ
.custom_model_2/dense_19/BiasAdd/ReadVariableOpReadVariableOp7custom_model_2_dense_19_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.custom_model_2/dense_19/BiasAdd/ReadVariableOpâ
custom_model_2/dense_19/BiasAddBiasAdd(custom_model_2/dense_19/MatMul:product:06custom_model_2/dense_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
custom_model_2/dense_19/BiasAdd¡
custom_model_2/dense_19/ReluRelu(custom_model_2/dense_19/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
custom_model_2/dense_19/Relu×
-custom_model_2/dense_20/MatMul/ReadVariableOpReadVariableOp6custom_model_2_dense_20_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02/
-custom_model_2/dense_20/MatMul/ReadVariableOpà
custom_model_2/dense_20/MatMulMatMul*custom_model_2/dense_19/Relu:activations:05custom_model_2/dense_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
custom_model_2/dense_20/MatMulÕ
.custom_model_2/dense_20/BiasAdd/ReadVariableOpReadVariableOp7custom_model_2_dense_20_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.custom_model_2/dense_20/BiasAdd/ReadVariableOpâ
custom_model_2/dense_20/BiasAddBiasAdd(custom_model_2/dense_20/MatMul:product:06custom_model_2/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
custom_model_2/dense_20/BiasAdd¡
custom_model_2/dense_20/ReluRelu(custom_model_2/dense_20/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
custom_model_2/dense_20/Relu×
-custom_model_2/dense_22/MatMul/ReadVariableOpReadVariableOp6custom_model_2_dense_22_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02/
-custom_model_2/dense_22/MatMul/ReadVariableOpÞ
custom_model_2/dense_22/MatMulMatMul(custom_model_2/dense_21/BiasAdd:output:05custom_model_2/dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
custom_model_2/dense_22/MatMulÕ
.custom_model_2/dense_22/BiasAdd/ReadVariableOpReadVariableOp7custom_model_2_dense_22_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.custom_model_2/dense_22/BiasAdd/ReadVariableOpâ
custom_model_2/dense_22/BiasAddBiasAdd(custom_model_2/dense_22/MatMul:product:06custom_model_2/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
custom_model_2/dense_22/BiasAdd
&custom_model_2/tf.concat_8/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&custom_model_2/tf.concat_8/concat/axis
!custom_model_2/tf.concat_8/concatConcatV2*custom_model_2/dense_20/Relu:activations:0(custom_model_2/dense_22/BiasAdd:output:0/custom_model_2/tf.concat_8/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!custom_model_2/tf.concat_8/concat×
-custom_model_2/dense_23/MatMul/ReadVariableOpReadVariableOp6custom_model_2_dense_23_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02/
-custom_model_2/dense_23/MatMul/ReadVariableOpà
custom_model_2/dense_23/MatMulMatMul*custom_model_2/tf.concat_8/concat:output:05custom_model_2/dense_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
custom_model_2/dense_23/MatMulÕ
.custom_model_2/dense_23/BiasAdd/ReadVariableOpReadVariableOp7custom_model_2_dense_23_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.custom_model_2/dense_23/BiasAdd/ReadVariableOpâ
custom_model_2/dense_23/BiasAddBiasAdd(custom_model_2/dense_23/MatMul:product:06custom_model_2/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
custom_model_2/dense_23/BiasAdd©
 custom_model_2/tf.nn.relu_6/ReluRelu(custom_model_2/dense_23/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 custom_model_2/tf.nn.relu_6/Relu×
-custom_model_2/dense_24/MatMul/ReadVariableOpReadVariableOp6custom_model_2_dense_24_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02/
-custom_model_2/dense_24/MatMul/ReadVariableOpä
custom_model_2/dense_24/MatMulMatMul.custom_model_2/tf.nn.relu_6/Relu:activations:05custom_model_2/dense_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
custom_model_2/dense_24/MatMulÕ
.custom_model_2/dense_24/BiasAdd/ReadVariableOpReadVariableOp7custom_model_2_dense_24_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.custom_model_2/dense_24/BiasAdd/ReadVariableOpâ
custom_model_2/dense_24/BiasAddBiasAdd(custom_model_2/dense_24/MatMul:product:06custom_model_2/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
custom_model_2/dense_24/BiasAddò
,custom_model_2/tf.__operators__.add_16/AddV2AddV2(custom_model_2/dense_24/BiasAdd:output:0.custom_model_2/tf.nn.relu_6/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,custom_model_2/tf.__operators__.add_16/AddV2±
 custom_model_2/tf.nn.relu_7/ReluRelu0custom_model_2/tf.__operators__.add_16/AddV2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 custom_model_2/tf.nn.relu_7/Relu×
-custom_model_2/dense_25/MatMul/ReadVariableOpReadVariableOp6custom_model_2_dense_25_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02/
-custom_model_2/dense_25/MatMul/ReadVariableOpä
custom_model_2/dense_25/MatMulMatMul.custom_model_2/tf.nn.relu_7/Relu:activations:05custom_model_2/dense_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
custom_model_2/dense_25/MatMulÕ
.custom_model_2/dense_25/BiasAdd/ReadVariableOpReadVariableOp7custom_model_2_dense_25_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.custom_model_2/dense_25/BiasAdd/ReadVariableOpâ
custom_model_2/dense_25/BiasAddBiasAdd(custom_model_2/dense_25/MatMul:product:06custom_model_2/dense_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
custom_model_2/dense_25/BiasAddò
,custom_model_2/tf.__operators__.add_17/AddV2AddV2(custom_model_2/dense_25/BiasAdd:output:0.custom_model_2/tf.nn.relu_7/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,custom_model_2/tf.__operators__.add_17/AddV2±
 custom_model_2/tf.nn.relu_8/ReluRelu0custom_model_2/tf.__operators__.add_17/AddV2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 custom_model_2/tf.nn.relu_8/Relu
Acustom_model_2/normalize_2/normalization_2/Reshape/ReadVariableOpReadVariableOpJcustom_model_2_normalize_2_normalization_2_reshape_readvariableop_resource*
_output_shapes	
:*
dtype02C
Acustom_model_2/normalize_2/normalization_2/Reshape/ReadVariableOpÅ
8custom_model_2/normalize_2/normalization_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2:
8custom_model_2/normalize_2/normalization_2/Reshape/shape«
2custom_model_2/normalize_2/normalization_2/ReshapeReshapeIcustom_model_2/normalize_2/normalization_2/Reshape/ReadVariableOp:value:0Acustom_model_2/normalize_2/normalization_2/Reshape/shape:output:0*
T0*
_output_shapes
:	24
2custom_model_2/normalize_2/normalization_2/Reshape
Ccustom_model_2/normalize_2/normalization_2/Reshape_1/ReadVariableOpReadVariableOpLcustom_model_2_normalize_2_normalization_2_reshape_1_readvariableop_resource*
_output_shapes	
:*
dtype02E
Ccustom_model_2/normalize_2/normalization_2/Reshape_1/ReadVariableOpÉ
:custom_model_2/normalize_2/normalization_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2<
:custom_model_2/normalize_2/normalization_2/Reshape_1/shape³
4custom_model_2/normalize_2/normalization_2/Reshape_1ReshapeKcustom_model_2/normalize_2/normalization_2/Reshape_1/ReadVariableOp:value:0Ccustom_model_2/normalize_2/normalization_2/Reshape_1/shape:output:0*
T0*
_output_shapes
:	26
4custom_model_2/normalize_2/normalization_2/Reshape_1
.custom_model_2/normalize_2/normalization_2/subSub.custom_model_2/tf.nn.relu_8/Relu:activations:0;custom_model_2/normalize_2/normalization_2/Reshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.custom_model_2/normalize_2/normalization_2/subÓ
/custom_model_2/normalize_2/normalization_2/SqrtSqrt=custom_model_2/normalize_2/normalization_2/Reshape_1:output:0*
T0*
_output_shapes
:	21
/custom_model_2/normalize_2/normalization_2/Sqrt±
4custom_model_2/normalize_2/normalization_2/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö326
4custom_model_2/normalize_2/normalization_2/Maximum/y
2custom_model_2/normalize_2/normalization_2/MaximumMaximum3custom_model_2/normalize_2/normalization_2/Sqrt:y:0=custom_model_2/normalize_2/normalization_2/Maximum/y:output:0*
T0*
_output_shapes
:	24
2custom_model_2/normalize_2/normalization_2/Maximum
2custom_model_2/normalize_2/normalization_2/truedivRealDiv2custom_model_2/normalize_2/normalization_2/sub:z:06custom_model_2/normalize_2/normalization_2/Maximum:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2custom_model_2/normalize_2/normalization_2/truedivÖ
-custom_model_2/dense_26/MatMul/ReadVariableOpReadVariableOp6custom_model_2_dense_26_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02/
-custom_model_2/dense_26/MatMul/ReadVariableOpë
custom_model_2/dense_26/MatMulMatMul6custom_model_2/normalize_2/normalization_2/truediv:z:05custom_model_2/dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
custom_model_2/dense_26/MatMulÔ
.custom_model_2/dense_26/BiasAdd/ReadVariableOpReadVariableOp7custom_model_2_dense_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.custom_model_2/dense_26/BiasAdd/ReadVariableOpá
custom_model_2/dense_26/BiasAddBiasAdd(custom_model_2/dense_26/MatMul:product:06custom_model_2/dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
custom_model_2/dense_26/BiasAdd¹
IdentityIdentity(custom_model_2/dense_26/BiasAdd:output:0/^custom_model_2/dense_18/BiasAdd/ReadVariableOp.^custom_model_2/dense_18/MatMul/ReadVariableOp/^custom_model_2/dense_19/BiasAdd/ReadVariableOp.^custom_model_2/dense_19/MatMul/ReadVariableOp/^custom_model_2/dense_20/BiasAdd/ReadVariableOp.^custom_model_2/dense_20/MatMul/ReadVariableOp/^custom_model_2/dense_21/BiasAdd/ReadVariableOp.^custom_model_2/dense_21/MatMul/ReadVariableOp/^custom_model_2/dense_22/BiasAdd/ReadVariableOp.^custom_model_2/dense_22/MatMul/ReadVariableOp/^custom_model_2/dense_23/BiasAdd/ReadVariableOp.^custom_model_2/dense_23/MatMul/ReadVariableOp/^custom_model_2/dense_24/BiasAdd/ReadVariableOp.^custom_model_2/dense_24/MatMul/ReadVariableOp/^custom_model_2/dense_25/BiasAdd/ReadVariableOp.^custom_model_2/dense_25/MatMul/ReadVariableOp/^custom_model_2/dense_26/BiasAdd/ReadVariableOp.^custom_model_2/dense_26/MatMul/ReadVariableOp5^custom_model_2/model_4/embedding_12/embedding_lookup5^custom_model_2/model_4/embedding_13/embedding_lookup5^custom_model_2/model_4/embedding_14/embedding_lookup5^custom_model_2/model_5/embedding_15/embedding_lookup5^custom_model_2/model_5/embedding_16/embedding_lookup5^custom_model_2/model_5/embedding_17/embedding_lookupB^custom_model_2/normalize_2/normalization_2/Reshape/ReadVariableOpD^custom_model_2/normalize_2/normalization_2/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*À
_input_shapes®
«:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
: : :::: :::: : ::::::::::::::::::::2`
.custom_model_2/dense_18/BiasAdd/ReadVariableOp.custom_model_2/dense_18/BiasAdd/ReadVariableOp2^
-custom_model_2/dense_18/MatMul/ReadVariableOp-custom_model_2/dense_18/MatMul/ReadVariableOp2`
.custom_model_2/dense_19/BiasAdd/ReadVariableOp.custom_model_2/dense_19/BiasAdd/ReadVariableOp2^
-custom_model_2/dense_19/MatMul/ReadVariableOp-custom_model_2/dense_19/MatMul/ReadVariableOp2`
.custom_model_2/dense_20/BiasAdd/ReadVariableOp.custom_model_2/dense_20/BiasAdd/ReadVariableOp2^
-custom_model_2/dense_20/MatMul/ReadVariableOp-custom_model_2/dense_20/MatMul/ReadVariableOp2`
.custom_model_2/dense_21/BiasAdd/ReadVariableOp.custom_model_2/dense_21/BiasAdd/ReadVariableOp2^
-custom_model_2/dense_21/MatMul/ReadVariableOp-custom_model_2/dense_21/MatMul/ReadVariableOp2`
.custom_model_2/dense_22/BiasAdd/ReadVariableOp.custom_model_2/dense_22/BiasAdd/ReadVariableOp2^
-custom_model_2/dense_22/MatMul/ReadVariableOp-custom_model_2/dense_22/MatMul/ReadVariableOp2`
.custom_model_2/dense_23/BiasAdd/ReadVariableOp.custom_model_2/dense_23/BiasAdd/ReadVariableOp2^
-custom_model_2/dense_23/MatMul/ReadVariableOp-custom_model_2/dense_23/MatMul/ReadVariableOp2`
.custom_model_2/dense_24/BiasAdd/ReadVariableOp.custom_model_2/dense_24/BiasAdd/ReadVariableOp2^
-custom_model_2/dense_24/MatMul/ReadVariableOp-custom_model_2/dense_24/MatMul/ReadVariableOp2`
.custom_model_2/dense_25/BiasAdd/ReadVariableOp.custom_model_2/dense_25/BiasAdd/ReadVariableOp2^
-custom_model_2/dense_25/MatMul/ReadVariableOp-custom_model_2/dense_25/MatMul/ReadVariableOp2`
.custom_model_2/dense_26/BiasAdd/ReadVariableOp.custom_model_2/dense_26/BiasAdd/ReadVariableOp2^
-custom_model_2/dense_26/MatMul/ReadVariableOp-custom_model_2/dense_26/MatMul/ReadVariableOp2l
4custom_model_2/model_4/embedding_12/embedding_lookup4custom_model_2/model_4/embedding_12/embedding_lookup2l
4custom_model_2/model_4/embedding_13/embedding_lookup4custom_model_2/model_4/embedding_13/embedding_lookup2l
4custom_model_2/model_4/embedding_14/embedding_lookup4custom_model_2/model_4/embedding_14/embedding_lookup2l
4custom_model_2/model_5/embedding_15/embedding_lookup4custom_model_2/model_5/embedding_15/embedding_lookup2l
4custom_model_2/model_5/embedding_16/embedding_lookup4custom_model_2/model_5/embedding_16/embedding_lookup2l
4custom_model_2/model_5/embedding_17/embedding_lookup4custom_model_2/model_5/embedding_17/embedding_lookup2
Acustom_model_2/normalize_2/normalization_2/Reshape/ReadVariableOpAcustom_model_2/normalize_2/normalization_2/Reshape/ReadVariableOp2
Ccustom_model_2/normalize_2/normalization_2/Reshape_1/ReadVariableOpCcustom_model_2/normalize_2/normalization_2/Reshape_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namecards0:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namecards1:MI
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


_user_specified_namebets:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ñ-
Û
F__inference_model_5_layer_call_and_return_conditional_losses_200071100
input_6*
&tf_math_greater_equal_7_greaterequal_y
embedding_17_200071044
embedding_15_200071066
embedding_16_200071090
identity¢$embedding_15/StatefulPartitionedCall¢$embedding_16/StatefulPartitionedCall¢$embedding_17/StatefulPartitionedCallÛ
flatten_5/PartitionedCallPartitionedCallinput_6*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_flatten_5_layer_call_and_return_conditional_losses_2000710072
flatten_5/PartitionedCall
*tf.clip_by_value_7/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2,
*tf.clip_by_value_7/clip_by_value/Minimum/yê
(tf.clip_by_value_7/clip_by_value/MinimumMinimum"flatten_5/PartitionedCall:output:03tf.clip_by_value_7/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(tf.clip_by_value_7/clip_by_value/Minimum
"tf.clip_by_value_7/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_7/clip_by_value/yÜ
 tf.clip_by_value_7/clip_by_valueMaximum,tf.clip_by_value_7/clip_by_value/Minimum:z:0+tf.clip_by_value_7/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 tf.clip_by_value_7/clip_by_value
#tf.compat.v1.floor_div_5/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2%
#tf.compat.v1.floor_div_5/FloorDiv/yØ
!tf.compat.v1.floor_div_5/FloorDivFloorDiv$tf.clip_by_value_7/clip_by_value:z:0,tf.compat.v1.floor_div_5/FloorDiv/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!tf.compat.v1.floor_div_5/FloorDivÚ
$tf.math.greater_equal_7/GreaterEqualGreaterEqual"flatten_5/PartitionedCall:output:0&tf_math_greater_equal_7_greaterequal_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$tf.math.greater_equal_7/GreaterEqual
tf.math.floormod_5/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2
tf.math.floormod_5/FloorMod/yÆ
tf.math.floormod_5/FloorModFloorMod$tf.clip_by_value_7/clip_by_value:z:0&tf.math.floormod_5/FloorMod/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.floormod_5/FloorModº
$embedding_17/StatefulPartitionedCallStatefulPartitionedCall$tf.clip_by_value_7/clip_by_value:z:0embedding_17_200071044*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_embedding_17_layer_call_and_return_conditional_losses_2000710352&
$embedding_17/StatefulPartitionedCall»
$embedding_15/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.floor_div_5/FloorDiv:z:0embedding_15_200071066*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_embedding_15_layer_call_and_return_conditional_losses_2000710572&
$embedding_15/StatefulPartitionedCall
tf.cast_7/CastCast(tf.math.greater_equal_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.cast_7/CastÜ
tf.__operators__.add_14/AddV2AddV2-embedding_17/StatefulPartitionedCall:output:0-embedding_15/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_14/AddV2µ
$embedding_16/StatefulPartitionedCallStatefulPartitionedCalltf.math.floormod_5/FloorMod:z:0embedding_16_200071090*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_embedding_16_layer_call_and_return_conditional_losses_2000710812&
$embedding_16/StatefulPartitionedCallÐ
tf.__operators__.add_15/AddV2AddV2!tf.__operators__.add_14/AddV2:z:0-embedding_16/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_15/AddV2
tf.expand_dims_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2!
tf.expand_dims_5/ExpandDims/dim¼
tf.expand_dims_5/ExpandDims
ExpandDimstf.cast_7/Cast:y:0(tf.expand_dims_5/ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.expand_dims_5/ExpandDims·
tf.math.multiply_5/MulMul!tf.__operators__.add_15/AddV2:z:0$tf.expand_dims_5/ExpandDims:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.multiply_5/Mul
*tf.math.reduce_sum_5/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_5/Sum/reduction_indices¿
tf.math.reduce_sum_5/SumSumtf.math.multiply_5/Mul:z:03tf.math.reduce_sum_5/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.reduce_sum_5/Sumë
IdentityIdentity!tf.math.reduce_sum_5/Sum:output:0%^embedding_15/StatefulPartitionedCall%^embedding_16/StatefulPartitionedCall%^embedding_17/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: :::2L
$embedding_15/StatefulPartitionedCall$embedding_15/StatefulPartitionedCall2L
$embedding_16/StatefulPartitionedCall$embedding_16/StatefulPartitionedCall2L
$embedding_17/StatefulPartitionedCall$embedding_17/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_6:

_output_shapes
: 
Ì
¤
J__inference_normalize_2_layer_call_and_return_conditional_losses_200071545
x3
/normalization_2_reshape_readvariableop_resource5
1normalization_2_reshape_1_readvariableop_resource
identity¢&normalization_2/Reshape/ReadVariableOp¢(normalization_2/Reshape_1/ReadVariableOp½
&normalization_2/Reshape/ReadVariableOpReadVariableOp/normalization_2_reshape_readvariableop_resource*
_output_shapes	
:*
dtype02(
&normalization_2/Reshape/ReadVariableOp
normalization_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_2/Reshape/shape¿
normalization_2/ReshapeReshape.normalization_2/Reshape/ReadVariableOp:value:0&normalization_2/Reshape/shape:output:0*
T0*
_output_shapes
:	2
normalization_2/ReshapeÃ
(normalization_2/Reshape_1/ReadVariableOpReadVariableOp1normalization_2_reshape_1_readvariableop_resource*
_output_shapes	
:*
dtype02*
(normalization_2/Reshape_1/ReadVariableOp
normalization_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_2/Reshape_1/shapeÇ
normalization_2/Reshape_1Reshape0normalization_2/Reshape_1/ReadVariableOp:value:0(normalization_2/Reshape_1/shape:output:0*
T0*
_output_shapes
:	2
normalization_2/Reshape_1
normalization_2/subSubx normalization_2/Reshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
normalization_2/sub
normalization_2/SqrtSqrt"normalization_2/Reshape_1:output:0*
T0*
_output_shapes
:	2
normalization_2/Sqrt{
normalization_2/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö32
normalization_2/Maximum/y¥
normalization_2/MaximumMaximumnormalization_2/Sqrt:y:0"normalization_2/Maximum/y:output:0*
T0*
_output_shapes
:	2
normalization_2/Maximum¦
normalization_2/truedivRealDivnormalization_2/sub:z:0normalization_2/Maximum:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
normalization_2/truedivÄ
IdentityIdentitynormalization_2/truediv:z:0'^normalization_2/Reshape/ReadVariableOp)^normalization_2/Reshape_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::2P
&normalization_2/Reshape/ReadVariableOp&normalization_2/Reshape/ReadVariableOp2T
(normalization_2/Reshape_1/ReadVariableOp(normalization_2/Reshape_1/ReadVariableOp:K G
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
ç

,__inference_dense_18_layer_call_fn_200072800

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_18_layer_call_and_return_conditional_losses_2000713212
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü
#
"__inference__traced_save_200073380
file_prefix.
*savev2_dense_18_kernel_read_readvariableop,
(savev2_dense_18_bias_read_readvariableop.
*savev2_dense_19_kernel_read_readvariableop,
(savev2_dense_19_bias_read_readvariableop.
*savev2_dense_21_kernel_read_readvariableop,
(savev2_dense_21_bias_read_readvariableop.
*savev2_dense_20_kernel_read_readvariableop,
(savev2_dense_20_bias_read_readvariableop.
*savev2_dense_22_kernel_read_readvariableop,
(savev2_dense_22_bias_read_readvariableop.
*savev2_dense_23_kernel_read_readvariableop,
(savev2_dense_23_bias_read_readvariableop.
*savev2_dense_24_kernel_read_readvariableop,
(savev2_dense_24_bias_read_readvariableop.
*savev2_dense_25_kernel_read_readvariableop,
(savev2_dense_25_bias_read_readvariableop.
*savev2_dense_26_kernel_read_readvariableop,
(savev2_dense_26_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop6
2savev2_embedding_14_embeddings_read_readvariableop6
2savev2_embedding_12_embeddings_read_readvariableop6
2savev2_embedding_13_embeddings_read_readvariableop6
2savev2_embedding_17_embeddings_read_readvariableop6
2savev2_embedding_15_embeddings_read_readvariableop6
2savev2_embedding_16_embeddings_read_readvariableop?
;savev2_normalize_2_normalization_2_mean_read_readvariableopC
?savev2_normalize_2_normalization_2_variance_read_readvariableop@
<savev2_normalize_2_normalization_2_count_read_readvariableop	$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_18_kernel_m_read_readvariableop3
/savev2_adam_dense_18_bias_m_read_readvariableop5
1savev2_adam_dense_19_kernel_m_read_readvariableop3
/savev2_adam_dense_19_bias_m_read_readvariableop5
1savev2_adam_dense_21_kernel_m_read_readvariableop3
/savev2_adam_dense_21_bias_m_read_readvariableop5
1savev2_adam_dense_20_kernel_m_read_readvariableop3
/savev2_adam_dense_20_bias_m_read_readvariableop5
1savev2_adam_dense_22_kernel_m_read_readvariableop3
/savev2_adam_dense_22_bias_m_read_readvariableop5
1savev2_adam_dense_23_kernel_m_read_readvariableop3
/savev2_adam_dense_23_bias_m_read_readvariableop5
1savev2_adam_dense_24_kernel_m_read_readvariableop3
/savev2_adam_dense_24_bias_m_read_readvariableop5
1savev2_adam_dense_25_kernel_m_read_readvariableop3
/savev2_adam_dense_25_bias_m_read_readvariableop5
1savev2_adam_dense_26_kernel_m_read_readvariableop3
/savev2_adam_dense_26_bias_m_read_readvariableop=
9savev2_adam_embedding_14_embeddings_m_read_readvariableop=
9savev2_adam_embedding_12_embeddings_m_read_readvariableop=
9savev2_adam_embedding_13_embeddings_m_read_readvariableop=
9savev2_adam_embedding_17_embeddings_m_read_readvariableop=
9savev2_adam_embedding_15_embeddings_m_read_readvariableop=
9savev2_adam_embedding_16_embeddings_m_read_readvariableop5
1savev2_adam_dense_18_kernel_v_read_readvariableop3
/savev2_adam_dense_18_bias_v_read_readvariableop5
1savev2_adam_dense_19_kernel_v_read_readvariableop3
/savev2_adam_dense_19_bias_v_read_readvariableop5
1savev2_adam_dense_21_kernel_v_read_readvariableop3
/savev2_adam_dense_21_bias_v_read_readvariableop5
1savev2_adam_dense_20_kernel_v_read_readvariableop3
/savev2_adam_dense_20_bias_v_read_readvariableop5
1savev2_adam_dense_22_kernel_v_read_readvariableop3
/savev2_adam_dense_22_bias_v_read_readvariableop5
1savev2_adam_dense_23_kernel_v_read_readvariableop3
/savev2_adam_dense_23_bias_v_read_readvariableop5
1savev2_adam_dense_24_kernel_v_read_readvariableop3
/savev2_adam_dense_24_bias_v_read_readvariableop5
1savev2_adam_dense_25_kernel_v_read_readvariableop3
/savev2_adam_dense_25_bias_v_read_readvariableop5
1savev2_adam_dense_26_kernel_v_read_readvariableop3
/savev2_adam_dense_26_bias_v_read_readvariableop=
9savev2_adam_embedding_14_embeddings_v_read_readvariableop=
9savev2_adam_embedding_12_embeddings_v_read_readvariableop=
9savev2_adam_embedding_13_embeddings_v_read_readvariableop=
9savev2_adam_embedding_17_embeddings_v_read_readvariableop=
9savev2_adam_embedding_15_embeddings_v_read_readvariableop=
9savev2_adam_embedding_16_embeddings_v_read_readvariableop
savev2_const_5

identity_1¢MergeV2Checkpoints
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
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
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
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename,
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*¯+
value¥+B¢+SB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names±
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*»
value±B®SB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesä!
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_18_kernel_read_readvariableop(savev2_dense_18_bias_read_readvariableop*savev2_dense_19_kernel_read_readvariableop(savev2_dense_19_bias_read_readvariableop*savev2_dense_21_kernel_read_readvariableop(savev2_dense_21_bias_read_readvariableop*savev2_dense_20_kernel_read_readvariableop(savev2_dense_20_bias_read_readvariableop*savev2_dense_22_kernel_read_readvariableop(savev2_dense_22_bias_read_readvariableop*savev2_dense_23_kernel_read_readvariableop(savev2_dense_23_bias_read_readvariableop*savev2_dense_24_kernel_read_readvariableop(savev2_dense_24_bias_read_readvariableop*savev2_dense_25_kernel_read_readvariableop(savev2_dense_25_bias_read_readvariableop*savev2_dense_26_kernel_read_readvariableop(savev2_dense_26_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop2savev2_embedding_14_embeddings_read_readvariableop2savev2_embedding_12_embeddings_read_readvariableop2savev2_embedding_13_embeddings_read_readvariableop2savev2_embedding_17_embeddings_read_readvariableop2savev2_embedding_15_embeddings_read_readvariableop2savev2_embedding_16_embeddings_read_readvariableop;savev2_normalize_2_normalization_2_mean_read_readvariableop?savev2_normalize_2_normalization_2_variance_read_readvariableop<savev2_normalize_2_normalization_2_count_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_18_kernel_m_read_readvariableop/savev2_adam_dense_18_bias_m_read_readvariableop1savev2_adam_dense_19_kernel_m_read_readvariableop/savev2_adam_dense_19_bias_m_read_readvariableop1savev2_adam_dense_21_kernel_m_read_readvariableop/savev2_adam_dense_21_bias_m_read_readvariableop1savev2_adam_dense_20_kernel_m_read_readvariableop/savev2_adam_dense_20_bias_m_read_readvariableop1savev2_adam_dense_22_kernel_m_read_readvariableop/savev2_adam_dense_22_bias_m_read_readvariableop1savev2_adam_dense_23_kernel_m_read_readvariableop/savev2_adam_dense_23_bias_m_read_readvariableop1savev2_adam_dense_24_kernel_m_read_readvariableop/savev2_adam_dense_24_bias_m_read_readvariableop1savev2_adam_dense_25_kernel_m_read_readvariableop/savev2_adam_dense_25_bias_m_read_readvariableop1savev2_adam_dense_26_kernel_m_read_readvariableop/savev2_adam_dense_26_bias_m_read_readvariableop9savev2_adam_embedding_14_embeddings_m_read_readvariableop9savev2_adam_embedding_12_embeddings_m_read_readvariableop9savev2_adam_embedding_13_embeddings_m_read_readvariableop9savev2_adam_embedding_17_embeddings_m_read_readvariableop9savev2_adam_embedding_15_embeddings_m_read_readvariableop9savev2_adam_embedding_16_embeddings_m_read_readvariableop1savev2_adam_dense_18_kernel_v_read_readvariableop/savev2_adam_dense_18_bias_v_read_readvariableop1savev2_adam_dense_19_kernel_v_read_readvariableop/savev2_adam_dense_19_bias_v_read_readvariableop1savev2_adam_dense_21_kernel_v_read_readvariableop/savev2_adam_dense_21_bias_v_read_readvariableop1savev2_adam_dense_20_kernel_v_read_readvariableop/savev2_adam_dense_20_bias_v_read_readvariableop1savev2_adam_dense_22_kernel_v_read_readvariableop/savev2_adam_dense_22_bias_v_read_readvariableop1savev2_adam_dense_23_kernel_v_read_readvariableop/savev2_adam_dense_23_bias_v_read_readvariableop1savev2_adam_dense_24_kernel_v_read_readvariableop/savev2_adam_dense_24_bias_v_read_readvariableop1savev2_adam_dense_25_kernel_v_read_readvariableop/savev2_adam_dense_25_bias_v_read_readvariableop1savev2_adam_dense_26_kernel_v_read_readvariableop/savev2_adam_dense_26_bias_v_read_readvariableop9savev2_adam_embedding_14_embeddings_v_read_readvariableop9savev2_adam_embedding_12_embeddings_v_read_readvariableop9savev2_adam_embedding_13_embeddings_v_read_readvariableop9savev2_adam_embedding_17_embeddings_v_read_readvariableop9savev2_adam_embedding_15_embeddings_v_read_readvariableop9savev2_adam_embedding_16_embeddings_v_read_readvariableopsavev2_const_5"/device:CPU:0*
_output_shapes
 *a
dtypesW
U2S		2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*õ
_input_shapesã
à: :
::
::	::
::
::
::
::
::	:: : : : : :	4:	:	:	4:	:	::: : : :
::
::	::
::
::
::
::
::	::	4:	:	:	4:	:	:
::
::	::
::
::
::
::
::	::	4:	:	:	4:	:	: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&	"
 
_output_shapes
:
:!


_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::
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
: :%!

_output_shapes
:	4:%!

_output_shapes
:	:%!

_output_shapes
:	:%!

_output_shapes
:	4:%!

_output_shapes
:	:%!

_output_shapes
:	:!

_output_shapes	
::!

_output_shapes	
:: 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :&#"
 
_output_shapes
:
:!$

_output_shapes	
::&%"
 
_output_shapes
:
:!&

_output_shapes	
::%'!

_output_shapes
:	:!(

_output_shapes	
::&)"
 
_output_shapes
:
:!*

_output_shapes	
::&+"
 
_output_shapes
:
:!,

_output_shapes	
::&-"
 
_output_shapes
:
:!.

_output_shapes	
::&/"
 
_output_shapes
:
:!0

_output_shapes	
::&1"
 
_output_shapes
:
:!2

_output_shapes	
::%3!

_output_shapes
:	: 4

_output_shapes
::%5!

_output_shapes
:	4:%6!

_output_shapes
:	:%7!

_output_shapes
:	:%8!

_output_shapes
:	4:%9!

_output_shapes
:	:%:!

_output_shapes
:	:&;"
 
_output_shapes
:
:!<

_output_shapes	
::&="
 
_output_shapes
:
:!>

_output_shapes	
::%?!

_output_shapes
:	:!@

_output_shapes	
::&A"
 
_output_shapes
:
:!B

_output_shapes	
::&C"
 
_output_shapes
:
:!D

_output_shapes	
::&E"
 
_output_shapes
:
:!F

_output_shapes	
::&G"
 
_output_shapes
:
:!H

_output_shapes	
::&I"
 
_output_shapes
:
:!J

_output_shapes	
::%K!

_output_shapes
:	: L

_output_shapes
::%M!

_output_shapes
:	4:%N!

_output_shapes
:	:%O!

_output_shapes
:	:%P!

_output_shapes
:	4:%Q!

_output_shapes
:	:%R!

_output_shapes
:	:S

_output_shapes
: 
	
à
G__inference_dense_26_layer_call_and_return_conditional_losses_200071571

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¯×
 ,
%__inference__traced_restore_200073636
file_prefix$
 assignvariableop_dense_18_kernel$
 assignvariableop_1_dense_18_bias&
"assignvariableop_2_dense_19_kernel$
 assignvariableop_3_dense_19_bias&
"assignvariableop_4_dense_21_kernel$
 assignvariableop_5_dense_21_bias&
"assignvariableop_6_dense_20_kernel$
 assignvariableop_7_dense_20_bias&
"assignvariableop_8_dense_22_kernel$
 assignvariableop_9_dense_22_bias'
#assignvariableop_10_dense_23_kernel%
!assignvariableop_11_dense_23_bias'
#assignvariableop_12_dense_24_kernel%
!assignvariableop_13_dense_24_bias'
#assignvariableop_14_dense_25_kernel%
!assignvariableop_15_dense_25_bias'
#assignvariableop_16_dense_26_kernel%
!assignvariableop_17_dense_26_bias!
assignvariableop_18_adam_iter#
assignvariableop_19_adam_beta_1#
assignvariableop_20_adam_beta_2"
assignvariableop_21_adam_decay*
&assignvariableop_22_adam_learning_rate/
+assignvariableop_23_embedding_14_embeddings/
+assignvariableop_24_embedding_12_embeddings/
+assignvariableop_25_embedding_13_embeddings/
+assignvariableop_26_embedding_17_embeddings/
+assignvariableop_27_embedding_15_embeddings/
+assignvariableop_28_embedding_16_embeddings8
4assignvariableop_29_normalize_2_normalization_2_mean<
8assignvariableop_30_normalize_2_normalization_2_variance9
5assignvariableop_31_normalize_2_normalization_2_count
assignvariableop_32_total
assignvariableop_33_count.
*assignvariableop_34_adam_dense_18_kernel_m,
(assignvariableop_35_adam_dense_18_bias_m.
*assignvariableop_36_adam_dense_19_kernel_m,
(assignvariableop_37_adam_dense_19_bias_m.
*assignvariableop_38_adam_dense_21_kernel_m,
(assignvariableop_39_adam_dense_21_bias_m.
*assignvariableop_40_adam_dense_20_kernel_m,
(assignvariableop_41_adam_dense_20_bias_m.
*assignvariableop_42_adam_dense_22_kernel_m,
(assignvariableop_43_adam_dense_22_bias_m.
*assignvariableop_44_adam_dense_23_kernel_m,
(assignvariableop_45_adam_dense_23_bias_m.
*assignvariableop_46_adam_dense_24_kernel_m,
(assignvariableop_47_adam_dense_24_bias_m.
*assignvariableop_48_adam_dense_25_kernel_m,
(assignvariableop_49_adam_dense_25_bias_m.
*assignvariableop_50_adam_dense_26_kernel_m,
(assignvariableop_51_adam_dense_26_bias_m6
2assignvariableop_52_adam_embedding_14_embeddings_m6
2assignvariableop_53_adam_embedding_12_embeddings_m6
2assignvariableop_54_adam_embedding_13_embeddings_m6
2assignvariableop_55_adam_embedding_17_embeddings_m6
2assignvariableop_56_adam_embedding_15_embeddings_m6
2assignvariableop_57_adam_embedding_16_embeddings_m.
*assignvariableop_58_adam_dense_18_kernel_v,
(assignvariableop_59_adam_dense_18_bias_v.
*assignvariableop_60_adam_dense_19_kernel_v,
(assignvariableop_61_adam_dense_19_bias_v.
*assignvariableop_62_adam_dense_21_kernel_v,
(assignvariableop_63_adam_dense_21_bias_v.
*assignvariableop_64_adam_dense_20_kernel_v,
(assignvariableop_65_adam_dense_20_bias_v.
*assignvariableop_66_adam_dense_22_kernel_v,
(assignvariableop_67_adam_dense_22_bias_v.
*assignvariableop_68_adam_dense_23_kernel_v,
(assignvariableop_69_adam_dense_23_bias_v.
*assignvariableop_70_adam_dense_24_kernel_v,
(assignvariableop_71_adam_dense_24_bias_v.
*assignvariableop_72_adam_dense_25_kernel_v,
(assignvariableop_73_adam_dense_25_bias_v.
*assignvariableop_74_adam_dense_26_kernel_v,
(assignvariableop_75_adam_dense_26_bias_v6
2assignvariableop_76_adam_embedding_14_embeddings_v6
2assignvariableop_77_adam_embedding_12_embeddings_v6
2assignvariableop_78_adam_embedding_13_embeddings_v6
2assignvariableop_79_adam_embedding_17_embeddings_v6
2assignvariableop_80_adam_embedding_15_embeddings_v6
2assignvariableop_81_adam_embedding_16_embeddings_v
identity_83¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_69¢AssignVariableOp_7¢AssignVariableOp_70¢AssignVariableOp_71¢AssignVariableOp_72¢AssignVariableOp_73¢AssignVariableOp_74¢AssignVariableOp_75¢AssignVariableOp_76¢AssignVariableOp_77¢AssignVariableOp_78¢AssignVariableOp_79¢AssignVariableOp_8¢AssignVariableOp_80¢AssignVariableOp_81¢AssignVariableOp_9£,
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*¯+
value¥+B¢+SB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names·
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*»
value±B®SB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesÍ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*â
_output_shapesÏ
Ì:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*a
dtypesW
U2S		2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp assignvariableop_dense_18_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¥
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_18_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2§
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_19_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¥
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_19_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4§
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_21_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¥
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_21_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6§
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_20_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¥
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_20_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8§
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_22_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¥
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_22_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10«
AssignVariableOp_10AssignVariableOp#assignvariableop_10_dense_23_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11©
AssignVariableOp_11AssignVariableOp!assignvariableop_11_dense_23_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12«
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_24_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13©
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_24_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14«
AssignVariableOp_14AssignVariableOp#assignvariableop_14_dense_25_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15©
AssignVariableOp_15AssignVariableOp!assignvariableop_15_dense_25_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16«
AssignVariableOp_16AssignVariableOp#assignvariableop_16_dense_26_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17©
AssignVariableOp_17AssignVariableOp!assignvariableop_17_dense_26_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_18¥
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_iterIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19§
AssignVariableOp_19AssignVariableOpassignvariableop_19_adam_beta_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20§
AssignVariableOp_20AssignVariableOpassignvariableop_20_adam_beta_2Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21¦
AssignVariableOp_21AssignVariableOpassignvariableop_21_adam_decayIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22®
AssignVariableOp_22AssignVariableOp&assignvariableop_22_adam_learning_rateIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23³
AssignVariableOp_23AssignVariableOp+assignvariableop_23_embedding_14_embeddingsIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24³
AssignVariableOp_24AssignVariableOp+assignvariableop_24_embedding_12_embeddingsIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25³
AssignVariableOp_25AssignVariableOp+assignvariableop_25_embedding_13_embeddingsIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26³
AssignVariableOp_26AssignVariableOp+assignvariableop_26_embedding_17_embeddingsIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27³
AssignVariableOp_27AssignVariableOp+assignvariableop_27_embedding_15_embeddingsIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28³
AssignVariableOp_28AssignVariableOp+assignvariableop_28_embedding_16_embeddingsIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29¼
AssignVariableOp_29AssignVariableOp4assignvariableop_29_normalize_2_normalization_2_meanIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30À
AssignVariableOp_30AssignVariableOp8assignvariableop_30_normalize_2_normalization_2_varianceIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_31½
AssignVariableOp_31AssignVariableOp5assignvariableop_31_normalize_2_normalization_2_countIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32¡
AssignVariableOp_32AssignVariableOpassignvariableop_32_totalIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33¡
AssignVariableOp_33AssignVariableOpassignvariableop_33_countIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34²
AssignVariableOp_34AssignVariableOp*assignvariableop_34_adam_dense_18_kernel_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35°
AssignVariableOp_35AssignVariableOp(assignvariableop_35_adam_dense_18_bias_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36²
AssignVariableOp_36AssignVariableOp*assignvariableop_36_adam_dense_19_kernel_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37°
AssignVariableOp_37AssignVariableOp(assignvariableop_37_adam_dense_19_bias_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38²
AssignVariableOp_38AssignVariableOp*assignvariableop_38_adam_dense_21_kernel_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39°
AssignVariableOp_39AssignVariableOp(assignvariableop_39_adam_dense_21_bias_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40²
AssignVariableOp_40AssignVariableOp*assignvariableop_40_adam_dense_20_kernel_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41°
AssignVariableOp_41AssignVariableOp(assignvariableop_41_adam_dense_20_bias_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42²
AssignVariableOp_42AssignVariableOp*assignvariableop_42_adam_dense_22_kernel_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43°
AssignVariableOp_43AssignVariableOp(assignvariableop_43_adam_dense_22_bias_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44²
AssignVariableOp_44AssignVariableOp*assignvariableop_44_adam_dense_23_kernel_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45°
AssignVariableOp_45AssignVariableOp(assignvariableop_45_adam_dense_23_bias_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46²
AssignVariableOp_46AssignVariableOp*assignvariableop_46_adam_dense_24_kernel_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47°
AssignVariableOp_47AssignVariableOp(assignvariableop_47_adam_dense_24_bias_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48²
AssignVariableOp_48AssignVariableOp*assignvariableop_48_adam_dense_25_kernel_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49°
AssignVariableOp_49AssignVariableOp(assignvariableop_49_adam_dense_25_bias_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50²
AssignVariableOp_50AssignVariableOp*assignvariableop_50_adam_dense_26_kernel_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51°
AssignVariableOp_51AssignVariableOp(assignvariableop_51_adam_dense_26_bias_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52º
AssignVariableOp_52AssignVariableOp2assignvariableop_52_adam_embedding_14_embeddings_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53º
AssignVariableOp_53AssignVariableOp2assignvariableop_53_adam_embedding_12_embeddings_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54º
AssignVariableOp_54AssignVariableOp2assignvariableop_54_adam_embedding_13_embeddings_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55º
AssignVariableOp_55AssignVariableOp2assignvariableop_55_adam_embedding_17_embeddings_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56º
AssignVariableOp_56AssignVariableOp2assignvariableop_56_adam_embedding_15_embeddings_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57º
AssignVariableOp_57AssignVariableOp2assignvariableop_57_adam_embedding_16_embeddings_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58²
AssignVariableOp_58AssignVariableOp*assignvariableop_58_adam_dense_18_kernel_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59°
AssignVariableOp_59AssignVariableOp(assignvariableop_59_adam_dense_18_bias_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60²
AssignVariableOp_60AssignVariableOp*assignvariableop_60_adam_dense_19_kernel_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61°
AssignVariableOp_61AssignVariableOp(assignvariableop_61_adam_dense_19_bias_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62²
AssignVariableOp_62AssignVariableOp*assignvariableop_62_adam_dense_21_kernel_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63°
AssignVariableOp_63AssignVariableOp(assignvariableop_63_adam_dense_21_bias_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64²
AssignVariableOp_64AssignVariableOp*assignvariableop_64_adam_dense_20_kernel_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65°
AssignVariableOp_65AssignVariableOp(assignvariableop_65_adam_dense_20_bias_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66²
AssignVariableOp_66AssignVariableOp*assignvariableop_66_adam_dense_22_kernel_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67°
AssignVariableOp_67AssignVariableOp(assignvariableop_67_adam_dense_22_bias_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68²
AssignVariableOp_68AssignVariableOp*assignvariableop_68_adam_dense_23_kernel_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69°
AssignVariableOp_69AssignVariableOp(assignvariableop_69_adam_dense_23_bias_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70²
AssignVariableOp_70AssignVariableOp*assignvariableop_70_adam_dense_24_kernel_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71°
AssignVariableOp_71AssignVariableOp(assignvariableop_71_adam_dense_24_bias_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72²
AssignVariableOp_72AssignVariableOp*assignvariableop_72_adam_dense_25_kernel_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73°
AssignVariableOp_73AssignVariableOp(assignvariableop_73_adam_dense_25_bias_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74²
AssignVariableOp_74AssignVariableOp*assignvariableop_74_adam_dense_26_kernel_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75°
AssignVariableOp_75AssignVariableOp(assignvariableop_75_adam_dense_26_bias_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76º
AssignVariableOp_76AssignVariableOp2assignvariableop_76_adam_embedding_14_embeddings_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77º
AssignVariableOp_77AssignVariableOp2assignvariableop_77_adam_embedding_12_embeddings_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78º
AssignVariableOp_78AssignVariableOp2assignvariableop_78_adam_embedding_13_embeddings_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79º
AssignVariableOp_79AssignVariableOp2assignvariableop_79_adam_embedding_17_embeddings_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80º
AssignVariableOp_80AssignVariableOp2assignvariableop_80_adam_embedding_15_embeddings_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81º
AssignVariableOp_81AssignVariableOp2assignvariableop_81_adam_embedding_16_embeddings_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_819
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpê
Identity_82Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_82Ý
Identity_83IdentityIdentity_82:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_83"#
identity_83Identity_83:output:0*ß
_input_shapesÍ
Ê: ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_81AssignVariableOp_812(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
å

,__inference_dense_26_layer_call_fn_200072980

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_26_layer_call_and_return_conditional_losses_2000715712
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
9
ø
F__inference_model_5_layer_call_and_return_conditional_losses_200072712

inputs*
&tf_math_greater_equal_7_greaterequal_y+
'embedding_17_embedding_lookup_200072686+
'embedding_15_embedding_lookup_200072692+
'embedding_16_embedding_lookup_200072700
identity¢embedding_15/embedding_lookup¢embedding_16/embedding_lookup¢embedding_17/embedding_lookups
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
flatten_5/Const
flatten_5/ReshapeReshapeinputsflatten_5/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flatten_5/Reshape
*tf.clip_by_value_7/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2,
*tf.clip_by_value_7/clip_by_value/Minimum/yâ
(tf.clip_by_value_7/clip_by_value/MinimumMinimumflatten_5/Reshape:output:03tf.clip_by_value_7/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(tf.clip_by_value_7/clip_by_value/Minimum
"tf.clip_by_value_7/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_7/clip_by_value/yÜ
 tf.clip_by_value_7/clip_by_valueMaximum,tf.clip_by_value_7/clip_by_value/Minimum:z:0+tf.clip_by_value_7/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 tf.clip_by_value_7/clip_by_value
#tf.compat.v1.floor_div_5/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2%
#tf.compat.v1.floor_div_5/FloorDiv/yØ
!tf.compat.v1.floor_div_5/FloorDivFloorDiv$tf.clip_by_value_7/clip_by_value:z:0,tf.compat.v1.floor_div_5/FloorDiv/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!tf.compat.v1.floor_div_5/FloorDivÒ
$tf.math.greater_equal_7/GreaterEqualGreaterEqualflatten_5/Reshape:output:0&tf_math_greater_equal_7_greaterequal_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$tf.math.greater_equal_7/GreaterEqual
tf.math.floormod_5/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2
tf.math.floormod_5/FloorMod/yÆ
tf.math.floormod_5/FloorModFloorMod$tf.clip_by_value_7/clip_by_value:z:0&tf.math.floormod_5/FloorMod/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.floormod_5/FloorMod
embedding_17/CastCast$tf.clip_by_value_7/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_17/CastÅ
embedding_17/embedding_lookupResourceGather'embedding_17_embedding_lookup_200072686embedding_17/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*:
_class0
.,loc:@embedding_17/embedding_lookup/200072686*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_17/embedding_lookup¥
&embedding_17/embedding_lookup/IdentityIdentity&embedding_17/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*:
_class0
.,loc:@embedding_17/embedding_lookup/200072686*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&embedding_17/embedding_lookup/IdentityÈ
(embedding_17/embedding_lookup/Identity_1Identity/embedding_17/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(embedding_17/embedding_lookup/Identity_1
embedding_15/CastCast%tf.compat.v1.floor_div_5/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_15/CastÅ
embedding_15/embedding_lookupResourceGather'embedding_15_embedding_lookup_200072692embedding_15/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*:
_class0
.,loc:@embedding_15/embedding_lookup/200072692*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_15/embedding_lookup¥
&embedding_15/embedding_lookup/IdentityIdentity&embedding_15/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*:
_class0
.,loc:@embedding_15/embedding_lookup/200072692*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&embedding_15/embedding_lookup/IdentityÈ
(embedding_15/embedding_lookup/Identity_1Identity/embedding_15/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(embedding_15/embedding_lookup/Identity_1
tf.cast_7/CastCast(tf.math.greater_equal_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.cast_7/Castä
tf.__operators__.add_14/AddV2AddV21embedding_17/embedding_lookup/Identity_1:output:01embedding_15/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_14/AddV2
embedding_16/CastCasttf.math.floormod_5/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_16/CastÅ
embedding_16/embedding_lookupResourceGather'embedding_16_embedding_lookup_200072700embedding_16/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*:
_class0
.,loc:@embedding_16/embedding_lookup/200072700*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_16/embedding_lookup¥
&embedding_16/embedding_lookup/IdentityIdentity&embedding_16/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*:
_class0
.,loc:@embedding_16/embedding_lookup/200072700*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&embedding_16/embedding_lookup/IdentityÈ
(embedding_16/embedding_lookup/Identity_1Identity/embedding_16/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(embedding_16/embedding_lookup/Identity_1Ô
tf.__operators__.add_15/AddV2AddV2!tf.__operators__.add_14/AddV2:z:01embedding_16/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_15/AddV2
tf.expand_dims_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2!
tf.expand_dims_5/ExpandDims/dim¼
tf.expand_dims_5/ExpandDims
ExpandDimstf.cast_7/Cast:y:0(tf.expand_dims_5/ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.expand_dims_5/ExpandDims·
tf.math.multiply_5/MulMul!tf.__operators__.add_15/AddV2:z:0$tf.expand_dims_5/ExpandDims:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.multiply_5/Mul
*tf.math.reduce_sum_5/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_5/Sum/reduction_indices¿
tf.math.reduce_sum_5/SumSumtf.math.multiply_5/Mul:z:03tf.math.reduce_sum_5/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.reduce_sum_5/SumÖ
IdentityIdentity!tf.math.reduce_sum_5/Sum:output:0^embedding_15/embedding_lookup^embedding_16/embedding_lookup^embedding_17/embedding_lookup*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: :::2>
embedding_15/embedding_lookupembedding_15/embedding_lookup2>
embedding_16/embedding_lookupembedding_16/embedding_lookup2>
embedding_17/embedding_lookupembedding_17/embedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: 
®
d
H__inference_flatten_5_layer_call_and_return_conditional_losses_200073048

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ç

,__inference_dense_23_layer_call_fn_200072897

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_23_layer_call_and_return_conditional_losses_2000714552
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾

+__inference_model_5_layer_call_fn_200072780

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_model_5_layer_call_and_return_conditional_losses_2000712122
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: :::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: 
Î-
Ú
F__inference_model_5_layer_call_and_return_conditional_losses_200071212

inputs*
&tf_math_greater_equal_7_greaterequal_y
embedding_17_200071194
embedding_15_200071197
embedding_16_200071202
identity¢$embedding_15/StatefulPartitionedCall¢$embedding_16/StatefulPartitionedCall¢$embedding_17/StatefulPartitionedCallÚ
flatten_5/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_flatten_5_layer_call_and_return_conditional_losses_2000710072
flatten_5/PartitionedCall
*tf.clip_by_value_7/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2,
*tf.clip_by_value_7/clip_by_value/Minimum/yê
(tf.clip_by_value_7/clip_by_value/MinimumMinimum"flatten_5/PartitionedCall:output:03tf.clip_by_value_7/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(tf.clip_by_value_7/clip_by_value/Minimum
"tf.clip_by_value_7/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_7/clip_by_value/yÜ
 tf.clip_by_value_7/clip_by_valueMaximum,tf.clip_by_value_7/clip_by_value/Minimum:z:0+tf.clip_by_value_7/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 tf.clip_by_value_7/clip_by_value
#tf.compat.v1.floor_div_5/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2%
#tf.compat.v1.floor_div_5/FloorDiv/yØ
!tf.compat.v1.floor_div_5/FloorDivFloorDiv$tf.clip_by_value_7/clip_by_value:z:0,tf.compat.v1.floor_div_5/FloorDiv/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!tf.compat.v1.floor_div_5/FloorDivÚ
$tf.math.greater_equal_7/GreaterEqualGreaterEqual"flatten_5/PartitionedCall:output:0&tf_math_greater_equal_7_greaterequal_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$tf.math.greater_equal_7/GreaterEqual
tf.math.floormod_5/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2
tf.math.floormod_5/FloorMod/yÆ
tf.math.floormod_5/FloorModFloorMod$tf.clip_by_value_7/clip_by_value:z:0&tf.math.floormod_5/FloorMod/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.floormod_5/FloorModº
$embedding_17/StatefulPartitionedCallStatefulPartitionedCall$tf.clip_by_value_7/clip_by_value:z:0embedding_17_200071194*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_embedding_17_layer_call_and_return_conditional_losses_2000710352&
$embedding_17/StatefulPartitionedCall»
$embedding_15/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.floor_div_5/FloorDiv:z:0embedding_15_200071197*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_embedding_15_layer_call_and_return_conditional_losses_2000710572&
$embedding_15/StatefulPartitionedCall
tf.cast_7/CastCast(tf.math.greater_equal_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.cast_7/CastÜ
tf.__operators__.add_14/AddV2AddV2-embedding_17/StatefulPartitionedCall:output:0-embedding_15/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_14/AddV2µ
$embedding_16/StatefulPartitionedCallStatefulPartitionedCalltf.math.floormod_5/FloorMod:z:0embedding_16_200071202*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_embedding_16_layer_call_and_return_conditional_losses_2000710812&
$embedding_16/StatefulPartitionedCallÐ
tf.__operators__.add_15/AddV2AddV2!tf.__operators__.add_14/AddV2:z:0-embedding_16/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_15/AddV2
tf.expand_dims_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2!
tf.expand_dims_5/ExpandDims/dim¼
tf.expand_dims_5/ExpandDims
ExpandDimstf.cast_7/Cast:y:0(tf.expand_dims_5/ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.expand_dims_5/ExpandDims·
tf.math.multiply_5/MulMul!tf.__operators__.add_15/AddV2:z:0$tf.expand_dims_5/ExpandDims:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.multiply_5/Mul
*tf.math.reduce_sum_5/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_5/Sum/reduction_indices¿
tf.math.reduce_sum_5/SumSumtf.math.multiply_5/Mul:z:03tf.math.reduce_sum_5/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.reduce_sum_5/Sumë
IdentityIdentity!tf.math.reduce_sum_5/Sum:output:0%^embedding_15/StatefulPartitionedCall%^embedding_16/StatefulPartitionedCall%^embedding_17/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: :::2L
$embedding_15/StatefulPartitionedCall$embedding_15/StatefulPartitionedCall2L
$embedding_16/StatefulPartitionedCall$embedding_16/StatefulPartitionedCall2L
$embedding_17/StatefulPartitionedCall$embedding_17/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: 
Ô
v
0__inference_embedding_14_layer_call_fn_200073008

inputs
unknown
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_embedding_14_layer_call_and_return_conditional_losses_2000708092
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ô
v
0__inference_embedding_16_layer_call_fn_200073104

inputs
unknown
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_embedding_16_layer_call_and_return_conditional_losses_2000710812
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
çX
Ï

M__inference_custom_model_2_layer_call_and_return_conditional_losses_200071938

inputs
inputs_1
inputs_2*
&tf_math_greater_equal_8_greaterequal_y
model_4_200071853
model_4_200071855
model_4_200071857
model_4_200071859
model_5_200071862
model_5_200071864
model_5_200071866
model_5_200071868.
*tf_clip_by_value_8_clip_by_value_minimum_y&
"tf_clip_by_value_8_clip_by_value_y
dense_18_200071880
dense_18_200071882
dense_21_200071885
dense_21_200071887
dense_19_200071890
dense_19_200071892
dense_20_200071895
dense_20_200071897
dense_22_200071900
dense_22_200071902
dense_23_200071907
dense_23_200071909
dense_24_200071913
dense_24_200071915
dense_25_200071920
dense_25_200071922
normalize_2_200071927
normalize_2_200071929
dense_26_200071932
dense_26_200071934
identity¢ dense_18/StatefulPartitionedCall¢ dense_19/StatefulPartitionedCall¢ dense_20/StatefulPartitionedCall¢ dense_21/StatefulPartitionedCall¢ dense_22/StatefulPartitionedCall¢ dense_23/StatefulPartitionedCall¢ dense_24/StatefulPartitionedCall¢ dense_25/StatefulPartitionedCall¢ dense_26/StatefulPartitionedCall¢model_4/StatefulPartitionedCall¢model_5/StatefulPartitionedCall¢#normalize_2/StatefulPartitionedCallÀ
$tf.math.greater_equal_8/GreaterEqualGreaterEqualinputs_2&tf_math_greater_equal_8_greaterequal_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2&
$tf.math.greater_equal_8/GreaterEqualÂ
model_4/StatefulPartitionedCallStatefulPartitionedCallinputsmodel_4_200071853model_4_200071855model_4_200071857model_4_200071859*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_model_4_layer_call_and_return_conditional_losses_2000709862!
model_4/StatefulPartitionedCallÄ
model_5/StatefulPartitionedCallStatefulPartitionedCallinputs_1model_5_200071862model_5_200071864model_5_200071866model_5_200071868*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_model_5_layer_call_and_return_conditional_losses_2000712122!
model_5/StatefulPartitionedCallÇ
(tf.clip_by_value_8/clip_by_value/MinimumMinimuminputs_2*tf_clip_by_value_8_clip_by_value_minimum_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2*
(tf.clip_by_value_8/clip_by_value/MinimumÓ
 tf.clip_by_value_8/clip_by_valueMaximum,tf.clip_by_value_8/clip_by_value/Minimum:z:0"tf_clip_by_value_8_clip_by_value_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2"
 tf.clip_by_value_8/clip_by_value
tf.cast_8/CastCast(tf.math.greater_equal_8/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
tf.cast_8/Castt
tf.concat_6/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_6/concat/axisæ
tf.concat_6/concatConcatV2(model_4/StatefulPartitionedCall:output:0(model_5/StatefulPartitionedCall:output:0 tf.concat_6/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.concat_6/concat}
tf.concat_7/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
tf.concat_7/concat/axisË
tf.concat_7/concatConcatV2$tf.clip_by_value_8/clip_by_value:z:0tf.cast_8/Cast:y:0 tf.concat_7/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.concat_7/concat³
 dense_18/StatefulPartitionedCallStatefulPartitionedCalltf.concat_6/concat:output:0dense_18_200071880dense_18_200071882*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_18_layer_call_and_return_conditional_losses_2000713212"
 dense_18/StatefulPartitionedCall³
 dense_21/StatefulPartitionedCallStatefulPartitionedCalltf.concat_7/concat:output:0dense_21_200071885dense_21_200071887*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_21_layer_call_and_return_conditional_losses_2000713472"
 dense_21/StatefulPartitionedCallÁ
 dense_19/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0dense_19_200071890dense_19_200071892*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_19_layer_call_and_return_conditional_losses_2000713742"
 dense_19/StatefulPartitionedCallÁ
 dense_20/StatefulPartitionedCallStatefulPartitionedCall)dense_19/StatefulPartitionedCall:output:0dense_20_200071895dense_20_200071897*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_20_layer_call_and_return_conditional_losses_2000714012"
 dense_20/StatefulPartitionedCallÁ
 dense_22/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0dense_22_200071900dense_22_200071902*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_22_layer_call_and_return_conditional_losses_2000714272"
 dense_22/StatefulPartitionedCall}
tf.concat_8/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
tf.concat_8/concat/axisè
tf.concat_8/concatConcatV2)dense_20/StatefulPartitionedCall:output:0)dense_22/StatefulPartitionedCall:output:0 tf.concat_8/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.concat_8/concat³
 dense_23/StatefulPartitionedCallStatefulPartitionedCalltf.concat_8/concat:output:0dense_23_200071907dense_23_200071909*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_23_layer_call_and_return_conditional_losses_2000714552"
 dense_23/StatefulPartitionedCall
tf.nn.relu_6/ReluRelu)dense_23/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.nn.relu_6/Relu·
 dense_24/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_6/Relu:activations:0dense_24_200071913dense_24_200071915*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_24_layer_call_and_return_conditional_losses_2000714822"
 dense_24/StatefulPartitionedCallÆ
tf.__operators__.add_16/AddV2AddV2)dense_24/StatefulPartitionedCall:output:0tf.nn.relu_6/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_16/AddV2
tf.nn.relu_7/ReluRelu!tf.__operators__.add_16/AddV2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.nn.relu_7/Relu·
 dense_25/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_7/Relu:activations:0dense_25_200071920dense_25_200071922*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_25_layer_call_and_return_conditional_losses_2000715102"
 dense_25/StatefulPartitionedCallÆ
tf.__operators__.add_17/AddV2AddV2)dense_25/StatefulPartitionedCall:output:0tf.nn.relu_7/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_17/AddV2
tf.nn.relu_8/ReluRelu!tf.__operators__.add_17/AddV2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.nn.relu_8/ReluÆ
#normalize_2/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_8/Relu:activations:0normalize_2_200071927normalize_2_200071929*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_normalize_2_layer_call_and_return_conditional_losses_2000715452%
#normalize_2/StatefulPartitionedCallÃ
 dense_26/StatefulPartitionedCallStatefulPartitionedCall,normalize_2/StatefulPartitionedCall:output:0dense_26_200071932dense_26_200071934*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_26_layer_call_and_return_conditional_losses_2000715712"
 dense_26/StatefulPartitionedCall¢
IdentityIdentity)dense_26/StatefulPartitionedCall:output:0!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall ^model_4/StatefulPartitionedCall ^model_5/StatefulPartitionedCall$^normalize_2/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*À
_input_shapes®
«:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
: : :::: :::: : ::::::::::::::::::::2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2B
model_4/StatefulPartitionedCallmodel_4/StatefulPartitionedCall2B
model_5/StatefulPartitionedCallmodel_5/StatefulPartitionedCall2J
#normalize_2/StatefulPartitionedCall#normalize_2/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ó	

K__inference_embedding_14_layer_call_and_return_conditional_losses_200070809

inputs
embedding_lookup_200070803
identity¢embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Cast
embedding_lookupResourceGatherembedding_lookup_200070803Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*-
_class#
!loc:@embedding_lookup/200070803*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_lookupñ
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@embedding_lookup/200070803*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity¡
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ó	

K__inference_embedding_12_layer_call_and_return_conditional_losses_200070831

inputs
embedding_lookup_200070825
identity¢embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Cast
embedding_lookupResourceGatherembedding_lookup_200070825Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*-
_class#
!loc:@embedding_lookup/200070825*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_lookupñ
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@embedding_lookup/200070825*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity¡
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
à
G__inference_dense_26_layer_call_and_return_conditional_losses_200072971

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
à
G__inference_dense_21_layer_call_and_return_conditional_losses_200071347

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ú	
à
G__inference_dense_19_layer_call_and_return_conditional_losses_200071374

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®
d
H__inference_flatten_4_layer_call_and_return_conditional_losses_200070781

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
9
ø
F__inference_model_4_layer_call_and_return_conditional_losses_200072602

inputs*
&tf_math_greater_equal_6_greaterequal_y+
'embedding_14_embedding_lookup_200072576+
'embedding_12_embedding_lookup_200072582+
'embedding_13_embedding_lookup_200072590
identity¢embedding_12/embedding_lookup¢embedding_13/embedding_lookup¢embedding_14/embedding_lookups
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
flatten_4/Const
flatten_4/ReshapeReshapeinputsflatten_4/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flatten_4/Reshape
*tf.clip_by_value_6/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2,
*tf.clip_by_value_6/clip_by_value/Minimum/yâ
(tf.clip_by_value_6/clip_by_value/MinimumMinimumflatten_4/Reshape:output:03tf.clip_by_value_6/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(tf.clip_by_value_6/clip_by_value/Minimum
"tf.clip_by_value_6/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_6/clip_by_value/yÜ
 tf.clip_by_value_6/clip_by_valueMaximum,tf.clip_by_value_6/clip_by_value/Minimum:z:0+tf.clip_by_value_6/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 tf.clip_by_value_6/clip_by_value
#tf.compat.v1.floor_div_4/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2%
#tf.compat.v1.floor_div_4/FloorDiv/yØ
!tf.compat.v1.floor_div_4/FloorDivFloorDiv$tf.clip_by_value_6/clip_by_value:z:0,tf.compat.v1.floor_div_4/FloorDiv/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!tf.compat.v1.floor_div_4/FloorDivÒ
$tf.math.greater_equal_6/GreaterEqualGreaterEqualflatten_4/Reshape:output:0&tf_math_greater_equal_6_greaterequal_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$tf.math.greater_equal_6/GreaterEqual
tf.math.floormod_4/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2
tf.math.floormod_4/FloorMod/yÆ
tf.math.floormod_4/FloorModFloorMod$tf.clip_by_value_6/clip_by_value:z:0&tf.math.floormod_4/FloorMod/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.floormod_4/FloorMod
embedding_14/CastCast$tf.clip_by_value_6/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_14/CastÅ
embedding_14/embedding_lookupResourceGather'embedding_14_embedding_lookup_200072576embedding_14/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*:
_class0
.,loc:@embedding_14/embedding_lookup/200072576*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_14/embedding_lookup¥
&embedding_14/embedding_lookup/IdentityIdentity&embedding_14/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*:
_class0
.,loc:@embedding_14/embedding_lookup/200072576*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&embedding_14/embedding_lookup/IdentityÈ
(embedding_14/embedding_lookup/Identity_1Identity/embedding_14/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(embedding_14/embedding_lookup/Identity_1
embedding_12/CastCast%tf.compat.v1.floor_div_4/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_12/CastÅ
embedding_12/embedding_lookupResourceGather'embedding_12_embedding_lookup_200072582embedding_12/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*:
_class0
.,loc:@embedding_12/embedding_lookup/200072582*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_12/embedding_lookup¥
&embedding_12/embedding_lookup/IdentityIdentity&embedding_12/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*:
_class0
.,loc:@embedding_12/embedding_lookup/200072582*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&embedding_12/embedding_lookup/IdentityÈ
(embedding_12/embedding_lookup/Identity_1Identity/embedding_12/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(embedding_12/embedding_lookup/Identity_1
tf.cast_6/CastCast(tf.math.greater_equal_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.cast_6/Castä
tf.__operators__.add_12/AddV2AddV21embedding_14/embedding_lookup/Identity_1:output:01embedding_12/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_12/AddV2
embedding_13/CastCasttf.math.floormod_4/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_13/CastÅ
embedding_13/embedding_lookupResourceGather'embedding_13_embedding_lookup_200072590embedding_13/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*:
_class0
.,loc:@embedding_13/embedding_lookup/200072590*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_13/embedding_lookup¥
&embedding_13/embedding_lookup/IdentityIdentity&embedding_13/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*:
_class0
.,loc:@embedding_13/embedding_lookup/200072590*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&embedding_13/embedding_lookup/IdentityÈ
(embedding_13/embedding_lookup/Identity_1Identity/embedding_13/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(embedding_13/embedding_lookup/Identity_1Ô
tf.__operators__.add_13/AddV2AddV2!tf.__operators__.add_12/AddV2:z:01embedding_13/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_13/AddV2
tf.expand_dims_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2!
tf.expand_dims_4/ExpandDims/dim¼
tf.expand_dims_4/ExpandDims
ExpandDimstf.cast_6/Cast:y:0(tf.expand_dims_4/ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.expand_dims_4/ExpandDims·
tf.math.multiply_4/MulMul!tf.__operators__.add_13/AddV2:z:0$tf.expand_dims_4/ExpandDims:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.multiply_4/Mul
*tf.math.reduce_sum_4/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_4/Sum/reduction_indices¿
tf.math.reduce_sum_4/SumSumtf.math.multiply_4/Mul:z:03tf.math.reduce_sum_4/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.reduce_sum_4/SumÖ
IdentityIdentity!tf.math.reduce_sum_4/Sum:output:0^embedding_12/embedding_lookup^embedding_13/embedding_lookup^embedding_14/embedding_lookup*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: :::2>
embedding_12/embedding_lookupembedding_12/embedding_lookup2>
embedding_13/embedding_lookupembedding_13/embedding_lookup2>
embedding_14/embedding_lookupembedding_14/embedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: 
ú	
à
G__inference_dense_20_layer_call_and_return_conditional_losses_200072850

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ó	

K__inference_embedding_15_layer_call_and_return_conditional_losses_200073080

inputs
embedding_lookup_200073074
identity¢embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Cast
embedding_lookupResourceGatherembedding_lookup_200073074Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*-
_class#
!loc:@embedding_lookup/200073074*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_lookupñ
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@embedding_lookup/200073074*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity¡
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾

+__inference_model_4_layer_call_fn_200072657

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_model_4_layer_call_and_return_conditional_losses_2000709412
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: :::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: 
ç

,__inference_dense_19_layer_call_fn_200072820

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_19_layer_call_and_return_conditional_losses_2000713742
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ó	

K__inference_embedding_17_layer_call_and_return_conditional_losses_200071035

inputs
embedding_lookup_200071029
identity¢embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Cast
embedding_lookupResourceGatherembedding_lookup_200071029Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*-
_class#
!loc:@embedding_lookup/200071029*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_lookupñ
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@embedding_lookup/200071029*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity¡
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
å

,__inference_dense_21_layer_call_fn_200072839

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_21_layer_call_and_return_conditional_losses_2000713472
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
çX
Ï

M__inference_custom_model_2_layer_call_and_return_conditional_losses_200071777

inputs
inputs_1
inputs_2*
&tf_math_greater_equal_8_greaterequal_y
model_4_200071692
model_4_200071694
model_4_200071696
model_4_200071698
model_5_200071701
model_5_200071703
model_5_200071705
model_5_200071707.
*tf_clip_by_value_8_clip_by_value_minimum_y&
"tf_clip_by_value_8_clip_by_value_y
dense_18_200071719
dense_18_200071721
dense_21_200071724
dense_21_200071726
dense_19_200071729
dense_19_200071731
dense_20_200071734
dense_20_200071736
dense_22_200071739
dense_22_200071741
dense_23_200071746
dense_23_200071748
dense_24_200071752
dense_24_200071754
dense_25_200071759
dense_25_200071761
normalize_2_200071766
normalize_2_200071768
dense_26_200071771
dense_26_200071773
identity¢ dense_18/StatefulPartitionedCall¢ dense_19/StatefulPartitionedCall¢ dense_20/StatefulPartitionedCall¢ dense_21/StatefulPartitionedCall¢ dense_22/StatefulPartitionedCall¢ dense_23/StatefulPartitionedCall¢ dense_24/StatefulPartitionedCall¢ dense_25/StatefulPartitionedCall¢ dense_26/StatefulPartitionedCall¢model_4/StatefulPartitionedCall¢model_5/StatefulPartitionedCall¢#normalize_2/StatefulPartitionedCallÀ
$tf.math.greater_equal_8/GreaterEqualGreaterEqualinputs_2&tf_math_greater_equal_8_greaterequal_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2&
$tf.math.greater_equal_8/GreaterEqualÂ
model_4/StatefulPartitionedCallStatefulPartitionedCallinputsmodel_4_200071692model_4_200071694model_4_200071696model_4_200071698*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_model_4_layer_call_and_return_conditional_losses_2000709412!
model_4/StatefulPartitionedCallÄ
model_5/StatefulPartitionedCallStatefulPartitionedCallinputs_1model_5_200071701model_5_200071703model_5_200071705model_5_200071707*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_model_5_layer_call_and_return_conditional_losses_2000711672!
model_5/StatefulPartitionedCallÇ
(tf.clip_by_value_8/clip_by_value/MinimumMinimuminputs_2*tf_clip_by_value_8_clip_by_value_minimum_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2*
(tf.clip_by_value_8/clip_by_value/MinimumÓ
 tf.clip_by_value_8/clip_by_valueMaximum,tf.clip_by_value_8/clip_by_value/Minimum:z:0"tf_clip_by_value_8_clip_by_value_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2"
 tf.clip_by_value_8/clip_by_value
tf.cast_8/CastCast(tf.math.greater_equal_8/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
tf.cast_8/Castt
tf.concat_6/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_6/concat/axisæ
tf.concat_6/concatConcatV2(model_4/StatefulPartitionedCall:output:0(model_5/StatefulPartitionedCall:output:0 tf.concat_6/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.concat_6/concat}
tf.concat_7/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
tf.concat_7/concat/axisË
tf.concat_7/concatConcatV2$tf.clip_by_value_8/clip_by_value:z:0tf.cast_8/Cast:y:0 tf.concat_7/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.concat_7/concat³
 dense_18/StatefulPartitionedCallStatefulPartitionedCalltf.concat_6/concat:output:0dense_18_200071719dense_18_200071721*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_18_layer_call_and_return_conditional_losses_2000713212"
 dense_18/StatefulPartitionedCall³
 dense_21/StatefulPartitionedCallStatefulPartitionedCalltf.concat_7/concat:output:0dense_21_200071724dense_21_200071726*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_21_layer_call_and_return_conditional_losses_2000713472"
 dense_21/StatefulPartitionedCallÁ
 dense_19/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0dense_19_200071729dense_19_200071731*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_19_layer_call_and_return_conditional_losses_2000713742"
 dense_19/StatefulPartitionedCallÁ
 dense_20/StatefulPartitionedCallStatefulPartitionedCall)dense_19/StatefulPartitionedCall:output:0dense_20_200071734dense_20_200071736*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_20_layer_call_and_return_conditional_losses_2000714012"
 dense_20/StatefulPartitionedCallÁ
 dense_22/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0dense_22_200071739dense_22_200071741*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_22_layer_call_and_return_conditional_losses_2000714272"
 dense_22/StatefulPartitionedCall}
tf.concat_8/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
tf.concat_8/concat/axisè
tf.concat_8/concatConcatV2)dense_20/StatefulPartitionedCall:output:0)dense_22/StatefulPartitionedCall:output:0 tf.concat_8/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.concat_8/concat³
 dense_23/StatefulPartitionedCallStatefulPartitionedCalltf.concat_8/concat:output:0dense_23_200071746dense_23_200071748*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_23_layer_call_and_return_conditional_losses_2000714552"
 dense_23/StatefulPartitionedCall
tf.nn.relu_6/ReluRelu)dense_23/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.nn.relu_6/Relu·
 dense_24/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_6/Relu:activations:0dense_24_200071752dense_24_200071754*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_24_layer_call_and_return_conditional_losses_2000714822"
 dense_24/StatefulPartitionedCallÆ
tf.__operators__.add_16/AddV2AddV2)dense_24/StatefulPartitionedCall:output:0tf.nn.relu_6/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_16/AddV2
tf.nn.relu_7/ReluRelu!tf.__operators__.add_16/AddV2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.nn.relu_7/Relu·
 dense_25/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_7/Relu:activations:0dense_25_200071759dense_25_200071761*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_25_layer_call_and_return_conditional_losses_2000715102"
 dense_25/StatefulPartitionedCallÆ
tf.__operators__.add_17/AddV2AddV2)dense_25/StatefulPartitionedCall:output:0tf.nn.relu_7/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_17/AddV2
tf.nn.relu_8/ReluRelu!tf.__operators__.add_17/AddV2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.nn.relu_8/ReluÆ
#normalize_2/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_8/Relu:activations:0normalize_2_200071766normalize_2_200071768*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_normalize_2_layer_call_and_return_conditional_losses_2000715452%
#normalize_2/StatefulPartitionedCallÃ
 dense_26/StatefulPartitionedCallStatefulPartitionedCall,normalize_2/StatefulPartitionedCall:output:0dense_26_200071771dense_26_200071773*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_26_layer_call_and_return_conditional_losses_2000715712"
 dense_26/StatefulPartitionedCall¢
IdentityIdentity)dense_26/StatefulPartitionedCall:output:0!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall ^model_4/StatefulPartitionedCall ^model_5/StatefulPartitionedCall$^normalize_2/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*À
_input_shapes®
«:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
: : :::: :::: : ::::::::::::::::::::2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2B
model_4/StatefulPartitionedCallmodel_4/StatefulPartitionedCall2B
model_5/StatefulPartitionedCallmodel_5/StatefulPartitionedCall2J
#normalize_2/StatefulPartitionedCall#normalize_2/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ñ
ð
2__inference_custom_model_2_layer_call_fn_200072560

inputs_0_0

inputs_0_1
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

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

unknown_29
identity¢StatefulPartitionedCall¡
StatefulPartitionedCallStatefulPartitionedCall
inputs_0_0
inputs_0_1inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_29*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*<
_read_only_resource_inputs
	
 !*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_custom_model_2_layer_call_and_return_conditional_losses_2000719382
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*À
_input_shapes®
«:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
: : :::: :::: : ::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/0/0:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/0/1:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ñ-
Û
F__inference_model_5_layer_call_and_return_conditional_losses_200071132
input_6*
&tf_math_greater_equal_7_greaterequal_y
embedding_17_200071114
embedding_15_200071117
embedding_16_200071122
identity¢$embedding_15/StatefulPartitionedCall¢$embedding_16/StatefulPartitionedCall¢$embedding_17/StatefulPartitionedCallÛ
flatten_5/PartitionedCallPartitionedCallinput_6*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_flatten_5_layer_call_and_return_conditional_losses_2000710072
flatten_5/PartitionedCall
*tf.clip_by_value_7/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2,
*tf.clip_by_value_7/clip_by_value/Minimum/yê
(tf.clip_by_value_7/clip_by_value/MinimumMinimum"flatten_5/PartitionedCall:output:03tf.clip_by_value_7/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(tf.clip_by_value_7/clip_by_value/Minimum
"tf.clip_by_value_7/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_7/clip_by_value/yÜ
 tf.clip_by_value_7/clip_by_valueMaximum,tf.clip_by_value_7/clip_by_value/Minimum:z:0+tf.clip_by_value_7/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 tf.clip_by_value_7/clip_by_value
#tf.compat.v1.floor_div_5/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2%
#tf.compat.v1.floor_div_5/FloorDiv/yØ
!tf.compat.v1.floor_div_5/FloorDivFloorDiv$tf.clip_by_value_7/clip_by_value:z:0,tf.compat.v1.floor_div_5/FloorDiv/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!tf.compat.v1.floor_div_5/FloorDivÚ
$tf.math.greater_equal_7/GreaterEqualGreaterEqual"flatten_5/PartitionedCall:output:0&tf_math_greater_equal_7_greaterequal_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$tf.math.greater_equal_7/GreaterEqual
tf.math.floormod_5/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2
tf.math.floormod_5/FloorMod/yÆ
tf.math.floormod_5/FloorModFloorMod$tf.clip_by_value_7/clip_by_value:z:0&tf.math.floormod_5/FloorMod/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.floormod_5/FloorModº
$embedding_17/StatefulPartitionedCallStatefulPartitionedCall$tf.clip_by_value_7/clip_by_value:z:0embedding_17_200071114*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_embedding_17_layer_call_and_return_conditional_losses_2000710352&
$embedding_17/StatefulPartitionedCall»
$embedding_15/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.floor_div_5/FloorDiv:z:0embedding_15_200071117*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_embedding_15_layer_call_and_return_conditional_losses_2000710572&
$embedding_15/StatefulPartitionedCall
tf.cast_7/CastCast(tf.math.greater_equal_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.cast_7/CastÜ
tf.__operators__.add_14/AddV2AddV2-embedding_17/StatefulPartitionedCall:output:0-embedding_15/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_14/AddV2µ
$embedding_16/StatefulPartitionedCallStatefulPartitionedCalltf.math.floormod_5/FloorMod:z:0embedding_16_200071122*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_embedding_16_layer_call_and_return_conditional_losses_2000710812&
$embedding_16/StatefulPartitionedCallÐ
tf.__operators__.add_15/AddV2AddV2!tf.__operators__.add_14/AddV2:z:0-embedding_16/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_15/AddV2
tf.expand_dims_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2!
tf.expand_dims_5/ExpandDims/dim¼
tf.expand_dims_5/ExpandDims
ExpandDimstf.cast_7/Cast:y:0(tf.expand_dims_5/ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.expand_dims_5/ExpandDims·
tf.math.multiply_5/MulMul!tf.__operators__.add_15/AddV2:z:0$tf.expand_dims_5/ExpandDims:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.multiply_5/Mul
*tf.math.reduce_sum_5/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_5/Sum/reduction_indices¿
tf.math.reduce_sum_5/SumSumtf.math.multiply_5/Mul:z:03tf.math.reduce_sum_5/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.reduce_sum_5/Sumë
IdentityIdentity!tf.math.reduce_sum_5/Sum:output:0%^embedding_15/StatefulPartitionedCall%^embedding_16/StatefulPartitionedCall%^embedding_17/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: :::2L
$embedding_15/StatefulPartitionedCall$embedding_15/StatefulPartitionedCall2L
$embedding_16/StatefulPartitionedCall$embedding_16/StatefulPartitionedCall2L
$embedding_17/StatefulPartitionedCall$embedding_17/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_6:

_output_shapes
: 
Á

+__inference_model_4_layer_call_fn_200070997
input_5
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_model_4_layer_call_and_return_conditional_losses_2000709862
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: :::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_5:

_output_shapes
: 
Î-
Ú
F__inference_model_4_layer_call_and_return_conditional_losses_200070941

inputs*
&tf_math_greater_equal_6_greaterequal_y
embedding_14_200070923
embedding_12_200070926
embedding_13_200070931
identity¢$embedding_12/StatefulPartitionedCall¢$embedding_13/StatefulPartitionedCall¢$embedding_14/StatefulPartitionedCallÚ
flatten_4/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_flatten_4_layer_call_and_return_conditional_losses_2000707812
flatten_4/PartitionedCall
*tf.clip_by_value_6/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2,
*tf.clip_by_value_6/clip_by_value/Minimum/yê
(tf.clip_by_value_6/clip_by_value/MinimumMinimum"flatten_4/PartitionedCall:output:03tf.clip_by_value_6/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(tf.clip_by_value_6/clip_by_value/Minimum
"tf.clip_by_value_6/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_6/clip_by_value/yÜ
 tf.clip_by_value_6/clip_by_valueMaximum,tf.clip_by_value_6/clip_by_value/Minimum:z:0+tf.clip_by_value_6/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 tf.clip_by_value_6/clip_by_value
#tf.compat.v1.floor_div_4/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2%
#tf.compat.v1.floor_div_4/FloorDiv/yØ
!tf.compat.v1.floor_div_4/FloorDivFloorDiv$tf.clip_by_value_6/clip_by_value:z:0,tf.compat.v1.floor_div_4/FloorDiv/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!tf.compat.v1.floor_div_4/FloorDivÚ
$tf.math.greater_equal_6/GreaterEqualGreaterEqual"flatten_4/PartitionedCall:output:0&tf_math_greater_equal_6_greaterequal_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$tf.math.greater_equal_6/GreaterEqual
tf.math.floormod_4/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2
tf.math.floormod_4/FloorMod/yÆ
tf.math.floormod_4/FloorModFloorMod$tf.clip_by_value_6/clip_by_value:z:0&tf.math.floormod_4/FloorMod/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.floormod_4/FloorModº
$embedding_14/StatefulPartitionedCallStatefulPartitionedCall$tf.clip_by_value_6/clip_by_value:z:0embedding_14_200070923*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_embedding_14_layer_call_and_return_conditional_losses_2000708092&
$embedding_14/StatefulPartitionedCall»
$embedding_12/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.floor_div_4/FloorDiv:z:0embedding_12_200070926*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_embedding_12_layer_call_and_return_conditional_losses_2000708312&
$embedding_12/StatefulPartitionedCall
tf.cast_6/CastCast(tf.math.greater_equal_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.cast_6/CastÜ
tf.__operators__.add_12/AddV2AddV2-embedding_14/StatefulPartitionedCall:output:0-embedding_12/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_12/AddV2µ
$embedding_13/StatefulPartitionedCallStatefulPartitionedCalltf.math.floormod_4/FloorMod:z:0embedding_13_200070931*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_embedding_13_layer_call_and_return_conditional_losses_2000708552&
$embedding_13/StatefulPartitionedCallÐ
tf.__operators__.add_13/AddV2AddV2!tf.__operators__.add_12/AddV2:z:0-embedding_13/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_13/AddV2
tf.expand_dims_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2!
tf.expand_dims_4/ExpandDims/dim¼
tf.expand_dims_4/ExpandDims
ExpandDimstf.cast_6/Cast:y:0(tf.expand_dims_4/ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.expand_dims_4/ExpandDims·
tf.math.multiply_4/MulMul!tf.__operators__.add_13/AddV2:z:0$tf.expand_dims_4/ExpandDims:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.multiply_4/Mul
*tf.math.reduce_sum_4/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_4/Sum/reduction_indices¿
tf.math.reduce_sum_4/SumSumtf.math.multiply_4/Mul:z:03tf.math.reduce_sum_4/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.reduce_sum_4/Sumë
IdentityIdentity!tf.math.reduce_sum_4/Sum:output:0%^embedding_12/StatefulPartitionedCall%^embedding_13/StatefulPartitionedCall%^embedding_14/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: :::2L
$embedding_12/StatefulPartitionedCall$embedding_12/StatefulPartitionedCall2L
$embedding_13/StatefulPartitionedCall$embedding_13/StatefulPartitionedCall2L
$embedding_14/StatefulPartitionedCall$embedding_14/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: 
	
à
G__inference_dense_22_layer_call_and_return_conditional_losses_200071427

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ëø
¦
M__inference_custom_model_2_layer_call_and_return_conditional_losses_200072252

inputs_0_0

inputs_0_1
inputs_1*
&tf_math_greater_equal_8_greaterequal_y2
.model_4_tf_math_greater_equal_6_greaterequal_y3
/model_4_embedding_14_embedding_lookup_2000721023
/model_4_embedding_12_embedding_lookup_2000721083
/model_4_embedding_13_embedding_lookup_2000721162
.model_5_tf_math_greater_equal_7_greaterequal_y3
/model_5_embedding_17_embedding_lookup_2000721403
/model_5_embedding_15_embedding_lookup_2000721463
/model_5_embedding_16_embedding_lookup_200072154.
*tf_clip_by_value_8_clip_by_value_minimum_y&
"tf_clip_by_value_8_clip_by_value_y+
'dense_18_matmul_readvariableop_resource,
(dense_18_biasadd_readvariableop_resource+
'dense_21_matmul_readvariableop_resource,
(dense_21_biasadd_readvariableop_resource+
'dense_19_matmul_readvariableop_resource,
(dense_19_biasadd_readvariableop_resource+
'dense_20_matmul_readvariableop_resource,
(dense_20_biasadd_readvariableop_resource+
'dense_22_matmul_readvariableop_resource,
(dense_22_biasadd_readvariableop_resource+
'dense_23_matmul_readvariableop_resource,
(dense_23_biasadd_readvariableop_resource+
'dense_24_matmul_readvariableop_resource,
(dense_24_biasadd_readvariableop_resource+
'dense_25_matmul_readvariableop_resource,
(dense_25_biasadd_readvariableop_resource?
;normalize_2_normalization_2_reshape_readvariableop_resourceA
=normalize_2_normalization_2_reshape_1_readvariableop_resource+
'dense_26_matmul_readvariableop_resource,
(dense_26_biasadd_readvariableop_resource
identity¢dense_18/BiasAdd/ReadVariableOp¢dense_18/MatMul/ReadVariableOp¢dense_19/BiasAdd/ReadVariableOp¢dense_19/MatMul/ReadVariableOp¢dense_20/BiasAdd/ReadVariableOp¢dense_20/MatMul/ReadVariableOp¢dense_21/BiasAdd/ReadVariableOp¢dense_21/MatMul/ReadVariableOp¢dense_22/BiasAdd/ReadVariableOp¢dense_22/MatMul/ReadVariableOp¢dense_23/BiasAdd/ReadVariableOp¢dense_23/MatMul/ReadVariableOp¢dense_24/BiasAdd/ReadVariableOp¢dense_24/MatMul/ReadVariableOp¢dense_25/BiasAdd/ReadVariableOp¢dense_25/MatMul/ReadVariableOp¢dense_26/BiasAdd/ReadVariableOp¢dense_26/MatMul/ReadVariableOp¢%model_4/embedding_12/embedding_lookup¢%model_4/embedding_13/embedding_lookup¢%model_4/embedding_14/embedding_lookup¢%model_5/embedding_15/embedding_lookup¢%model_5/embedding_16/embedding_lookup¢%model_5/embedding_17/embedding_lookup¢2normalize_2/normalization_2/Reshape/ReadVariableOp¢4normalize_2/normalization_2/Reshape_1/ReadVariableOpÀ
$tf.math.greater_equal_8/GreaterEqualGreaterEqualinputs_1&tf_math_greater_equal_8_greaterequal_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2&
$tf.math.greater_equal_8/GreaterEqual
model_4/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
model_4/flatten_4/Const¡
model_4/flatten_4/ReshapeReshape
inputs_0_0 model_4/flatten_4/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_4/flatten_4/Reshape­
2model_4/tf.clip_by_value_6/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI24
2model_4/tf.clip_by_value_6/clip_by_value/Minimum/y
0model_4/tf.clip_by_value_6/clip_by_value/MinimumMinimum"model_4/flatten_4/Reshape:output:0;model_4/tf.clip_by_value_6/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0model_4/tf.clip_by_value_6/clip_by_value/Minimum
*model_4/tf.clip_by_value_6/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*model_4/tf.clip_by_value_6/clip_by_value/yü
(model_4/tf.clip_by_value_6/clip_by_valueMaximum4model_4/tf.clip_by_value_6/clip_by_value/Minimum:z:03model_4/tf.clip_by_value_6/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(model_4/tf.clip_by_value_6/clip_by_value
+model_4/tf.compat.v1.floor_div_4/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2-
+model_4/tf.compat.v1.floor_div_4/FloorDiv/yø
)model_4/tf.compat.v1.floor_div_4/FloorDivFloorDiv,model_4/tf.clip_by_value_6/clip_by_value:z:04model_4/tf.compat.v1.floor_div_4/FloorDiv/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)model_4/tf.compat.v1.floor_div_4/FloorDivò
,model_4/tf.math.greater_equal_6/GreaterEqualGreaterEqual"model_4/flatten_4/Reshape:output:0.model_4_tf_math_greater_equal_6_greaterequal_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,model_4/tf.math.greater_equal_6/GreaterEqual
%model_4/tf.math.floormod_4/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2'
%model_4/tf.math.floormod_4/FloorMod/yæ
#model_4/tf.math.floormod_4/FloorModFloorMod,model_4/tf.clip_by_value_6/clip_by_value:z:0.model_4/tf.math.floormod_4/FloorMod/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#model_4/tf.math.floormod_4/FloorMod­
model_4/embedding_14/CastCast,model_4/tf.clip_by_value_6/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_4/embedding_14/Castí
%model_4/embedding_14/embedding_lookupResourceGather/model_4_embedding_14_embedding_lookup_200072102model_4/embedding_14/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@model_4/embedding_14/embedding_lookup/200072102*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02'
%model_4/embedding_14/embedding_lookupÅ
.model_4/embedding_14/embedding_lookup/IdentityIdentity.model_4/embedding_14/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@model_4/embedding_14/embedding_lookup/200072102*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.model_4/embedding_14/embedding_lookup/Identityà
0model_4/embedding_14/embedding_lookup/Identity_1Identity7model_4/embedding_14/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0model_4/embedding_14/embedding_lookup/Identity_1®
model_4/embedding_12/CastCast-model_4/tf.compat.v1.floor_div_4/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_4/embedding_12/Castí
%model_4/embedding_12/embedding_lookupResourceGather/model_4_embedding_12_embedding_lookup_200072108model_4/embedding_12/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@model_4/embedding_12/embedding_lookup/200072108*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02'
%model_4/embedding_12/embedding_lookupÅ
.model_4/embedding_12/embedding_lookup/IdentityIdentity.model_4/embedding_12/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@model_4/embedding_12/embedding_lookup/200072108*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.model_4/embedding_12/embedding_lookup/Identityà
0model_4/embedding_12/embedding_lookup/Identity_1Identity7model_4/embedding_12/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0model_4/embedding_12/embedding_lookup/Identity_1«
model_4/tf.cast_6/CastCast0model_4/tf.math.greater_equal_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_4/tf.cast_6/Cast
%model_4/tf.__operators__.add_12/AddV2AddV29model_4/embedding_14/embedding_lookup/Identity_1:output:09model_4/embedding_12/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%model_4/tf.__operators__.add_12/AddV2¨
model_4/embedding_13/CastCast'model_4/tf.math.floormod_4/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_4/embedding_13/Castí
%model_4/embedding_13/embedding_lookupResourceGather/model_4_embedding_13_embedding_lookup_200072116model_4/embedding_13/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@model_4/embedding_13/embedding_lookup/200072116*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02'
%model_4/embedding_13/embedding_lookupÅ
.model_4/embedding_13/embedding_lookup/IdentityIdentity.model_4/embedding_13/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@model_4/embedding_13/embedding_lookup/200072116*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.model_4/embedding_13/embedding_lookup/Identityà
0model_4/embedding_13/embedding_lookup/Identity_1Identity7model_4/embedding_13/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0model_4/embedding_13/embedding_lookup/Identity_1ô
%model_4/tf.__operators__.add_13/AddV2AddV2)model_4/tf.__operators__.add_12/AddV2:z:09model_4/embedding_13/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%model_4/tf.__operators__.add_13/AddV2
'model_4/tf.expand_dims_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2)
'model_4/tf.expand_dims_4/ExpandDims/dimÜ
#model_4/tf.expand_dims_4/ExpandDims
ExpandDimsmodel_4/tf.cast_6/Cast:y:00model_4/tf.expand_dims_4/ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#model_4/tf.expand_dims_4/ExpandDims×
model_4/tf.math.multiply_4/MulMul)model_4/tf.__operators__.add_13/AddV2:z:0,model_4/tf.expand_dims_4/ExpandDims:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
model_4/tf.math.multiply_4/Mulª
2model_4/tf.math.reduce_sum_4/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :24
2model_4/tf.math.reduce_sum_4/Sum/reduction_indicesß
 model_4/tf.math.reduce_sum_4/SumSum"model_4/tf.math.multiply_4/Mul:z:0;model_4/tf.math.reduce_sum_4/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 model_4/tf.math.reduce_sum_4/Sum
model_5/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
model_5/flatten_5/Const¡
model_5/flatten_5/ReshapeReshape
inputs_0_1 model_5/flatten_5/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_5/flatten_5/Reshape­
2model_5/tf.clip_by_value_7/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI24
2model_5/tf.clip_by_value_7/clip_by_value/Minimum/y
0model_5/tf.clip_by_value_7/clip_by_value/MinimumMinimum"model_5/flatten_5/Reshape:output:0;model_5/tf.clip_by_value_7/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0model_5/tf.clip_by_value_7/clip_by_value/Minimum
*model_5/tf.clip_by_value_7/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*model_5/tf.clip_by_value_7/clip_by_value/yü
(model_5/tf.clip_by_value_7/clip_by_valueMaximum4model_5/tf.clip_by_value_7/clip_by_value/Minimum:z:03model_5/tf.clip_by_value_7/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(model_5/tf.clip_by_value_7/clip_by_value
+model_5/tf.compat.v1.floor_div_5/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2-
+model_5/tf.compat.v1.floor_div_5/FloorDiv/yø
)model_5/tf.compat.v1.floor_div_5/FloorDivFloorDiv,model_5/tf.clip_by_value_7/clip_by_value:z:04model_5/tf.compat.v1.floor_div_5/FloorDiv/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)model_5/tf.compat.v1.floor_div_5/FloorDivò
,model_5/tf.math.greater_equal_7/GreaterEqualGreaterEqual"model_5/flatten_5/Reshape:output:0.model_5_tf_math_greater_equal_7_greaterequal_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,model_5/tf.math.greater_equal_7/GreaterEqual
%model_5/tf.math.floormod_5/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2'
%model_5/tf.math.floormod_5/FloorMod/yæ
#model_5/tf.math.floormod_5/FloorModFloorMod,model_5/tf.clip_by_value_7/clip_by_value:z:0.model_5/tf.math.floormod_5/FloorMod/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#model_5/tf.math.floormod_5/FloorMod­
model_5/embedding_17/CastCast,model_5/tf.clip_by_value_7/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_5/embedding_17/Castí
%model_5/embedding_17/embedding_lookupResourceGather/model_5_embedding_17_embedding_lookup_200072140model_5/embedding_17/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@model_5/embedding_17/embedding_lookup/200072140*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02'
%model_5/embedding_17/embedding_lookupÅ
.model_5/embedding_17/embedding_lookup/IdentityIdentity.model_5/embedding_17/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@model_5/embedding_17/embedding_lookup/200072140*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.model_5/embedding_17/embedding_lookup/Identityà
0model_5/embedding_17/embedding_lookup/Identity_1Identity7model_5/embedding_17/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0model_5/embedding_17/embedding_lookup/Identity_1®
model_5/embedding_15/CastCast-model_5/tf.compat.v1.floor_div_5/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_5/embedding_15/Castí
%model_5/embedding_15/embedding_lookupResourceGather/model_5_embedding_15_embedding_lookup_200072146model_5/embedding_15/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@model_5/embedding_15/embedding_lookup/200072146*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02'
%model_5/embedding_15/embedding_lookupÅ
.model_5/embedding_15/embedding_lookup/IdentityIdentity.model_5/embedding_15/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@model_5/embedding_15/embedding_lookup/200072146*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.model_5/embedding_15/embedding_lookup/Identityà
0model_5/embedding_15/embedding_lookup/Identity_1Identity7model_5/embedding_15/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0model_5/embedding_15/embedding_lookup/Identity_1«
model_5/tf.cast_7/CastCast0model_5/tf.math.greater_equal_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_5/tf.cast_7/Cast
%model_5/tf.__operators__.add_14/AddV2AddV29model_5/embedding_17/embedding_lookup/Identity_1:output:09model_5/embedding_15/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%model_5/tf.__operators__.add_14/AddV2¨
model_5/embedding_16/CastCast'model_5/tf.math.floormod_5/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_5/embedding_16/Castí
%model_5/embedding_16/embedding_lookupResourceGather/model_5_embedding_16_embedding_lookup_200072154model_5/embedding_16/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@model_5/embedding_16/embedding_lookup/200072154*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02'
%model_5/embedding_16/embedding_lookupÅ
.model_5/embedding_16/embedding_lookup/IdentityIdentity.model_5/embedding_16/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@model_5/embedding_16/embedding_lookup/200072154*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.model_5/embedding_16/embedding_lookup/Identityà
0model_5/embedding_16/embedding_lookup/Identity_1Identity7model_5/embedding_16/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0model_5/embedding_16/embedding_lookup/Identity_1ô
%model_5/tf.__operators__.add_15/AddV2AddV2)model_5/tf.__operators__.add_14/AddV2:z:09model_5/embedding_16/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%model_5/tf.__operators__.add_15/AddV2
'model_5/tf.expand_dims_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2)
'model_5/tf.expand_dims_5/ExpandDims/dimÜ
#model_5/tf.expand_dims_5/ExpandDims
ExpandDimsmodel_5/tf.cast_7/Cast:y:00model_5/tf.expand_dims_5/ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#model_5/tf.expand_dims_5/ExpandDims×
model_5/tf.math.multiply_5/MulMul)model_5/tf.__operators__.add_15/AddV2:z:0,model_5/tf.expand_dims_5/ExpandDims:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
model_5/tf.math.multiply_5/Mulª
2model_5/tf.math.reduce_sum_5/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :24
2model_5/tf.math.reduce_sum_5/Sum/reduction_indicesß
 model_5/tf.math.reduce_sum_5/SumSum"model_5/tf.math.multiply_5/Mul:z:0;model_5/tf.math.reduce_sum_5/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 model_5/tf.math.reduce_sum_5/SumÇ
(tf.clip_by_value_8/clip_by_value/MinimumMinimuminputs_1*tf_clip_by_value_8_clip_by_value_minimum_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2*
(tf.clip_by_value_8/clip_by_value/MinimumÓ
 tf.clip_by_value_8/clip_by_valueMaximum,tf.clip_by_value_8/clip_by_value/Minimum:z:0"tf_clip_by_value_8_clip_by_value_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2"
 tf.clip_by_value_8/clip_by_value
tf.cast_8/CastCast(tf.math.greater_equal_8/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
tf.cast_8/Castt
tf.concat_6/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_6/concat/axisè
tf.concat_6/concatConcatV2)model_4/tf.math.reduce_sum_4/Sum:output:0)model_5/tf.math.reduce_sum_5/Sum:output:0 tf.concat_6/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.concat_6/concat}
tf.concat_7/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
tf.concat_7/concat/axisË
tf.concat_7/concatConcatV2$tf.clip_by_value_8/clip_by_value:z:0tf.cast_8/Cast:y:0 tf.concat_7/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.concat_7/concatª
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_18/MatMul/ReadVariableOp¤
dense_18/MatMulMatMultf.concat_6/concat:output:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_18/MatMul¨
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_18/BiasAdd/ReadVariableOp¦
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_18/BiasAddt
dense_18/ReluReludense_18/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_18/Relu©
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02 
dense_21/MatMul/ReadVariableOp¤
dense_21/MatMulMatMultf.concat_7/concat:output:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_21/MatMul¨
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_21/BiasAdd/ReadVariableOp¦
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_21/BiasAddª
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_19/MatMul/ReadVariableOp¤
dense_19/MatMulMatMuldense_18/Relu:activations:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_19/MatMul¨
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_19/BiasAdd/ReadVariableOp¦
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_19/BiasAddt
dense_19/ReluReludense_19/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_19/Reluª
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_20/MatMul/ReadVariableOp¤
dense_20/MatMulMatMuldense_19/Relu:activations:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_20/MatMul¨
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_20/BiasAdd/ReadVariableOp¦
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_20/BiasAddt
dense_20/ReluReludense_20/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_20/Reluª
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_22/MatMul/ReadVariableOp¢
dense_22/MatMulMatMuldense_21/BiasAdd:output:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_22/MatMul¨
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_22/BiasAdd/ReadVariableOp¦
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_22/BiasAdd}
tf.concat_8/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
tf.concat_8/concat/axisÊ
tf.concat_8/concatConcatV2dense_20/Relu:activations:0dense_22/BiasAdd:output:0 tf.concat_8/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.concat_8/concatª
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_23/MatMul/ReadVariableOp¤
dense_23/MatMulMatMultf.concat_8/concat:output:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_23/MatMul¨
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_23/BiasAdd/ReadVariableOp¦
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_23/BiasAdd|
tf.nn.relu_6/ReluReludense_23/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.nn.relu_6/Reluª
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_24/MatMul/ReadVariableOp¨
dense_24/MatMulMatMultf.nn.relu_6/Relu:activations:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_24/MatMul¨
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_24/BiasAdd/ReadVariableOp¦
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_24/BiasAdd¶
tf.__operators__.add_16/AddV2AddV2dense_24/BiasAdd:output:0tf.nn.relu_6/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_16/AddV2
tf.nn.relu_7/ReluRelu!tf.__operators__.add_16/AddV2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.nn.relu_7/Reluª
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_25/MatMul/ReadVariableOp¨
dense_25/MatMulMatMultf.nn.relu_7/Relu:activations:0&dense_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_25/MatMul¨
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_25/BiasAdd/ReadVariableOp¦
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_25/BiasAdd¶
tf.__operators__.add_17/AddV2AddV2dense_25/BiasAdd:output:0tf.nn.relu_7/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_17/AddV2
tf.nn.relu_8/ReluRelu!tf.__operators__.add_17/AddV2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.nn.relu_8/Reluá
2normalize_2/normalization_2/Reshape/ReadVariableOpReadVariableOp;normalize_2_normalization_2_reshape_readvariableop_resource*
_output_shapes	
:*
dtype024
2normalize_2/normalization_2/Reshape/ReadVariableOp§
)normalize_2/normalization_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2+
)normalize_2/normalization_2/Reshape/shapeï
#normalize_2/normalization_2/ReshapeReshape:normalize_2/normalization_2/Reshape/ReadVariableOp:value:02normalize_2/normalization_2/Reshape/shape:output:0*
T0*
_output_shapes
:	2%
#normalize_2/normalization_2/Reshapeç
4normalize_2/normalization_2/Reshape_1/ReadVariableOpReadVariableOp=normalize_2_normalization_2_reshape_1_readvariableop_resource*
_output_shapes	
:*
dtype026
4normalize_2/normalization_2/Reshape_1/ReadVariableOp«
+normalize_2/normalization_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2-
+normalize_2/normalization_2/Reshape_1/shape÷
%normalize_2/normalization_2/Reshape_1Reshape<normalize_2/normalization_2/Reshape_1/ReadVariableOp:value:04normalize_2/normalization_2/Reshape_1/shape:output:0*
T0*
_output_shapes
:	2'
%normalize_2/normalization_2/Reshape_1Ë
normalize_2/normalization_2/subSubtf.nn.relu_8/Relu:activations:0,normalize_2/normalization_2/Reshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
normalize_2/normalization_2/sub¦
 normalize_2/normalization_2/SqrtSqrt.normalize_2/normalization_2/Reshape_1:output:0*
T0*
_output_shapes
:	2"
 normalize_2/normalization_2/Sqrt
%normalize_2/normalization_2/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö32'
%normalize_2/normalization_2/Maximum/yÕ
#normalize_2/normalization_2/MaximumMaximum$normalize_2/normalization_2/Sqrt:y:0.normalize_2/normalization_2/Maximum/y:output:0*
T0*
_output_shapes
:	2%
#normalize_2/normalization_2/MaximumÖ
#normalize_2/normalization_2/truedivRealDiv#normalize_2/normalization_2/sub:z:0'normalize_2/normalization_2/Maximum:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#normalize_2/normalization_2/truediv©
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02 
dense_26/MatMul/ReadVariableOp¯
dense_26/MatMulMatMul'normalize_2/normalization_2/truediv:z:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_26/MatMul§
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_26/BiasAdd/ReadVariableOp¥
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_26/BiasAdd¤
IdentityIdentitydense_26/BiasAdd:output:0 ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp ^dense_26/BiasAdd/ReadVariableOp^dense_26/MatMul/ReadVariableOp&^model_4/embedding_12/embedding_lookup&^model_4/embedding_13/embedding_lookup&^model_4/embedding_14/embedding_lookup&^model_5/embedding_15/embedding_lookup&^model_5/embedding_16/embedding_lookup&^model_5/embedding_17/embedding_lookup3^normalize_2/normalization_2/Reshape/ReadVariableOp5^normalize_2/normalization_2/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*À
_input_shapes®
«:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
: : :::: :::: : ::::::::::::::::::::2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2@
dense_25/MatMul/ReadVariableOpdense_25/MatMul/ReadVariableOp2B
dense_26/BiasAdd/ReadVariableOpdense_26/BiasAdd/ReadVariableOp2@
dense_26/MatMul/ReadVariableOpdense_26/MatMul/ReadVariableOp2N
%model_4/embedding_12/embedding_lookup%model_4/embedding_12/embedding_lookup2N
%model_4/embedding_13/embedding_lookup%model_4/embedding_13/embedding_lookup2N
%model_4/embedding_14/embedding_lookup%model_4/embedding_14/embedding_lookup2N
%model_5/embedding_15/embedding_lookup%model_5/embedding_15/embedding_lookup2N
%model_5/embedding_16/embedding_lookup%model_5/embedding_16/embedding_lookup2N
%model_5/embedding_17/embedding_lookup%model_5/embedding_17/embedding_lookup2h
2normalize_2/normalization_2/Reshape/ReadVariableOp2normalize_2/normalization_2/Reshape/ReadVariableOp2l
4normalize_2/normalization_2/Reshape_1/ReadVariableOp4normalize_2/normalization_2/Reshape_1/ReadVariableOp:S O
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/0/0:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/0/1:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ó	

K__inference_embedding_16_layer_call_and_return_conditional_losses_200073097

inputs
embedding_lookup_200073091
identity¢embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Cast
embedding_lookupResourceGatherembedding_lookup_200073091Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*-
_class#
!loc:@embedding_lookup/200073091*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_lookupñ
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@embedding_lookup/200073091*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity¡
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Á

+__inference_model_5_layer_call_fn_200071178
input_6
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_6unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_model_5_layer_call_and_return_conditional_losses_2000711672
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: :::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_6:

_output_shapes
: 
ó	

K__inference_embedding_14_layer_call_and_return_conditional_losses_200073001

inputs
embedding_lookup_200072995
identity¢embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Cast
embedding_lookupResourceGatherembedding_lookup_200072995Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*-
_class#
!loc:@embedding_lookup/200072995*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_lookupñ
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@embedding_lookup/200072995*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity¡
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

I
-__inference_flatten_4_layer_call_fn_200072991

inputs
identityÆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_flatten_4_layer_call_and_return_conditional_losses_2000707812
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾

+__inference_model_4_layer_call_fn_200072670

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_model_4_layer_call_and_return_conditional_losses_2000709862
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: :::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: 
ó	

K__inference_embedding_16_layer_call_and_return_conditional_losses_200071081

inputs
embedding_lookup_200071075
identity¢embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Cast
embedding_lookupResourceGatherembedding_lookup_200071075Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*-
_class#
!loc:@embedding_lookup/200071075*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_lookupñ
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@embedding_lookup/200071075*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity¡
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
à
G__inference_dense_21_layer_call_and_return_conditional_losses_200072830

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
­
ä
2__inference_custom_model_2_layer_call_fn_200072003

cards0

cards1
bets
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

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

unknown_29
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallcards0cards1betsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_29*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*<
_read_only_resource_inputs
	
 !*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_custom_model_2_layer_call_and_return_conditional_losses_2000719382
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*À
_input_shapes®
«:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
: : :::: :::: : ::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namecards0:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namecards1:MI
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


_user_specified_namebets:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ú	
à
G__inference_dense_18_layer_call_and_return_conditional_losses_200072791

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
à
G__inference_dense_24_layer_call_and_return_conditional_losses_200072907

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

I
-__inference_flatten_5_layer_call_fn_200073053

inputs
identityÆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_flatten_5_layer_call_and_return_conditional_losses_2000710072
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
­
ä
2__inference_custom_model_2_layer_call_fn_200071842

cards0

cards1
bets
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

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

unknown_29
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallcards0cards1betsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_29*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*<
_read_only_resource_inputs
	
 !*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_custom_model_2_layer_call_and_return_conditional_losses_2000717772
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*À
_input_shapes®
«:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
: : :::: :::: : ::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namecards0:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namecards1:MI
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


_user_specified_namebets:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ú	
à
G__inference_dense_19_layer_call_and_return_conditional_losses_200072811

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾

+__inference_model_5_layer_call_fn_200072767

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_model_5_layer_call_and_return_conditional_losses_2000711672
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: :::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: 
ç

,__inference_dense_22_layer_call_fn_200072878

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_22_layer_call_and_return_conditional_losses_2000714272
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ó	

K__inference_embedding_12_layer_call_and_return_conditional_losses_200073018

inputs
embedding_lookup_200073012
identity¢embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Cast
embedding_lookupResourceGatherembedding_lookup_200073012Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*-
_class#
!loc:@embedding_lookup/200073012*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_lookupñ
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@embedding_lookup/200073012*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity¡
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ç

,__inference_dense_24_layer_call_fn_200072916

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_24_layer_call_and_return_conditional_losses_2000714822
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ú	
à
G__inference_dense_20_layer_call_and_return_conditional_losses_200071401

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Á

+__inference_model_4_layer_call_fn_200070952
input_5
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_model_4_layer_call_and_return_conditional_losses_2000709412
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: :::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_5:

_output_shapes
: 
ó	

K__inference_embedding_13_layer_call_and_return_conditional_losses_200073035

inputs
embedding_lookup_200073029
identity¢embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Cast
embedding_lookupResourceGatherembedding_lookup_200073029Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*-
_class#
!loc:@embedding_lookup/200073029*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_lookupñ
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@embedding_lookup/200073029*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity¡
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
à
G__inference_dense_23_layer_call_and_return_conditional_losses_200072888

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ó	

K__inference_embedding_17_layer_call_and_return_conditional_losses_200073063

inputs
embedding_lookup_200073057
identity¢embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Cast
embedding_lookupResourceGatherembedding_lookup_200073057Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*-
_class#
!loc:@embedding_lookup/200073057*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_lookupñ
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@embedding_lookup/200073057*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity¡
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
à
G__inference_dense_25_layer_call_and_return_conditional_losses_200071510

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ý

/__inference_normalize_2_layer_call_fn_200072961
x
unknown
	unknown_0
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_normalize_2_layer_call_and_return_conditional_losses_2000715452
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:K G
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
ç

,__inference_dense_25_layer_call_fn_200072935

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_25_layer_call_and_return_conditional_losses_2000715102
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
à
G__inference_dense_24_layer_call_and_return_conditional_losses_200071482

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ç

,__inference_dense_20_layer_call_fn_200072859

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_20_layer_call_and_return_conditional_losses_2000714012
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ
ð
2__inference_custom_model_2_layer_call_fn_200072491

inputs_0_0

inputs_0_1
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

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

unknown_29
identity¢StatefulPartitionedCall¡
StatefulPartitionedCallStatefulPartitionedCall
inputs_0_0
inputs_0_1inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_29*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*<
_read_only_resource_inputs
	
 !*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_custom_model_2_layer_call_and_return_conditional_losses_2000717772
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*À
_input_shapes®
«:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
: : :::: :::: : ::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/0/0:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/0/1:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ú	
à
G__inference_dense_18_layer_call_and_return_conditional_losses_200071321

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
à
G__inference_dense_22_layer_call_and_return_conditional_losses_200072869

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ-
Û
F__inference_model_4_layer_call_and_return_conditional_losses_200070906
input_5*
&tf_math_greater_equal_6_greaterequal_y
embedding_14_200070888
embedding_12_200070891
embedding_13_200070896
identity¢$embedding_12/StatefulPartitionedCall¢$embedding_13/StatefulPartitionedCall¢$embedding_14/StatefulPartitionedCallÛ
flatten_4/PartitionedCallPartitionedCallinput_5*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_flatten_4_layer_call_and_return_conditional_losses_2000707812
flatten_4/PartitionedCall
*tf.clip_by_value_6/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2,
*tf.clip_by_value_6/clip_by_value/Minimum/yê
(tf.clip_by_value_6/clip_by_value/MinimumMinimum"flatten_4/PartitionedCall:output:03tf.clip_by_value_6/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(tf.clip_by_value_6/clip_by_value/Minimum
"tf.clip_by_value_6/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_6/clip_by_value/yÜ
 tf.clip_by_value_6/clip_by_valueMaximum,tf.clip_by_value_6/clip_by_value/Minimum:z:0+tf.clip_by_value_6/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 tf.clip_by_value_6/clip_by_value
#tf.compat.v1.floor_div_4/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2%
#tf.compat.v1.floor_div_4/FloorDiv/yØ
!tf.compat.v1.floor_div_4/FloorDivFloorDiv$tf.clip_by_value_6/clip_by_value:z:0,tf.compat.v1.floor_div_4/FloorDiv/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!tf.compat.v1.floor_div_4/FloorDivÚ
$tf.math.greater_equal_6/GreaterEqualGreaterEqual"flatten_4/PartitionedCall:output:0&tf_math_greater_equal_6_greaterequal_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$tf.math.greater_equal_6/GreaterEqual
tf.math.floormod_4/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2
tf.math.floormod_4/FloorMod/yÆ
tf.math.floormod_4/FloorModFloorMod$tf.clip_by_value_6/clip_by_value:z:0&tf.math.floormod_4/FloorMod/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.floormod_4/FloorModº
$embedding_14/StatefulPartitionedCallStatefulPartitionedCall$tf.clip_by_value_6/clip_by_value:z:0embedding_14_200070888*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_embedding_14_layer_call_and_return_conditional_losses_2000708092&
$embedding_14/StatefulPartitionedCall»
$embedding_12/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.floor_div_4/FloorDiv:z:0embedding_12_200070891*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_embedding_12_layer_call_and_return_conditional_losses_2000708312&
$embedding_12/StatefulPartitionedCall
tf.cast_6/CastCast(tf.math.greater_equal_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.cast_6/CastÜ
tf.__operators__.add_12/AddV2AddV2-embedding_14/StatefulPartitionedCall:output:0-embedding_12/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_12/AddV2µ
$embedding_13/StatefulPartitionedCallStatefulPartitionedCalltf.math.floormod_4/FloorMod:z:0embedding_13_200070896*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_embedding_13_layer_call_and_return_conditional_losses_2000708552&
$embedding_13/StatefulPartitionedCallÐ
tf.__operators__.add_13/AddV2AddV2!tf.__operators__.add_12/AddV2:z:0-embedding_13/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_13/AddV2
tf.expand_dims_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2!
tf.expand_dims_4/ExpandDims/dim¼
tf.expand_dims_4/ExpandDims
ExpandDimstf.cast_6/Cast:y:0(tf.expand_dims_4/ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.expand_dims_4/ExpandDims·
tf.math.multiply_4/MulMul!tf.__operators__.add_13/AddV2:z:0$tf.expand_dims_4/ExpandDims:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.multiply_4/Mul
*tf.math.reduce_sum_4/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_4/Sum/reduction_indices¿
tf.math.reduce_sum_4/SumSumtf.math.multiply_4/Mul:z:03tf.math.reduce_sum_4/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.reduce_sum_4/Sumë
IdentityIdentity!tf.math.reduce_sum_4/Sum:output:0%^embedding_12/StatefulPartitionedCall%^embedding_13/StatefulPartitionedCall%^embedding_14/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: :::2L
$embedding_12/StatefulPartitionedCall$embedding_12/StatefulPartitionedCall2L
$embedding_13/StatefulPartitionedCall$embedding_13/StatefulPartitionedCall2L
$embedding_14/StatefulPartitionedCall$embedding_14/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_5:

_output_shapes
: 
Ô
v
0__inference_embedding_12_layer_call_fn_200073025

inputs
unknown
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_embedding_12_layer_call_and_return_conditional_losses_2000708312
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
9
ø
F__inference_model_4_layer_call_and_return_conditional_losses_200072644

inputs*
&tf_math_greater_equal_6_greaterequal_y+
'embedding_14_embedding_lookup_200072618+
'embedding_12_embedding_lookup_200072624+
'embedding_13_embedding_lookup_200072632
identity¢embedding_12/embedding_lookup¢embedding_13/embedding_lookup¢embedding_14/embedding_lookups
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
flatten_4/Const
flatten_4/ReshapeReshapeinputsflatten_4/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flatten_4/Reshape
*tf.clip_by_value_6/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2,
*tf.clip_by_value_6/clip_by_value/Minimum/yâ
(tf.clip_by_value_6/clip_by_value/MinimumMinimumflatten_4/Reshape:output:03tf.clip_by_value_6/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(tf.clip_by_value_6/clip_by_value/Minimum
"tf.clip_by_value_6/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_6/clip_by_value/yÜ
 tf.clip_by_value_6/clip_by_valueMaximum,tf.clip_by_value_6/clip_by_value/Minimum:z:0+tf.clip_by_value_6/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 tf.clip_by_value_6/clip_by_value
#tf.compat.v1.floor_div_4/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2%
#tf.compat.v1.floor_div_4/FloorDiv/yØ
!tf.compat.v1.floor_div_4/FloorDivFloorDiv$tf.clip_by_value_6/clip_by_value:z:0,tf.compat.v1.floor_div_4/FloorDiv/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!tf.compat.v1.floor_div_4/FloorDivÒ
$tf.math.greater_equal_6/GreaterEqualGreaterEqualflatten_4/Reshape:output:0&tf_math_greater_equal_6_greaterequal_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$tf.math.greater_equal_6/GreaterEqual
tf.math.floormod_4/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2
tf.math.floormod_4/FloorMod/yÆ
tf.math.floormod_4/FloorModFloorMod$tf.clip_by_value_6/clip_by_value:z:0&tf.math.floormod_4/FloorMod/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.floormod_4/FloorMod
embedding_14/CastCast$tf.clip_by_value_6/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_14/CastÅ
embedding_14/embedding_lookupResourceGather'embedding_14_embedding_lookup_200072618embedding_14/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*:
_class0
.,loc:@embedding_14/embedding_lookup/200072618*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_14/embedding_lookup¥
&embedding_14/embedding_lookup/IdentityIdentity&embedding_14/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*:
_class0
.,loc:@embedding_14/embedding_lookup/200072618*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&embedding_14/embedding_lookup/IdentityÈ
(embedding_14/embedding_lookup/Identity_1Identity/embedding_14/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(embedding_14/embedding_lookup/Identity_1
embedding_12/CastCast%tf.compat.v1.floor_div_4/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_12/CastÅ
embedding_12/embedding_lookupResourceGather'embedding_12_embedding_lookup_200072624embedding_12/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*:
_class0
.,loc:@embedding_12/embedding_lookup/200072624*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_12/embedding_lookup¥
&embedding_12/embedding_lookup/IdentityIdentity&embedding_12/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*:
_class0
.,loc:@embedding_12/embedding_lookup/200072624*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&embedding_12/embedding_lookup/IdentityÈ
(embedding_12/embedding_lookup/Identity_1Identity/embedding_12/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(embedding_12/embedding_lookup/Identity_1
tf.cast_6/CastCast(tf.math.greater_equal_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.cast_6/Castä
tf.__operators__.add_12/AddV2AddV21embedding_14/embedding_lookup/Identity_1:output:01embedding_12/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_12/AddV2
embedding_13/CastCasttf.math.floormod_4/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_13/CastÅ
embedding_13/embedding_lookupResourceGather'embedding_13_embedding_lookup_200072632embedding_13/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*:
_class0
.,loc:@embedding_13/embedding_lookup/200072632*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_13/embedding_lookup¥
&embedding_13/embedding_lookup/IdentityIdentity&embedding_13/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*:
_class0
.,loc:@embedding_13/embedding_lookup/200072632*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&embedding_13/embedding_lookup/IdentityÈ
(embedding_13/embedding_lookup/Identity_1Identity/embedding_13/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(embedding_13/embedding_lookup/Identity_1Ô
tf.__operators__.add_13/AddV2AddV2!tf.__operators__.add_12/AddV2:z:01embedding_13/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_13/AddV2
tf.expand_dims_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2!
tf.expand_dims_4/ExpandDims/dim¼
tf.expand_dims_4/ExpandDims
ExpandDimstf.cast_6/Cast:y:0(tf.expand_dims_4/ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.expand_dims_4/ExpandDims·
tf.math.multiply_4/MulMul!tf.__operators__.add_13/AddV2:z:0$tf.expand_dims_4/ExpandDims:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.multiply_4/Mul
*tf.math.reduce_sum_4/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_4/Sum/reduction_indices¿
tf.math.reduce_sum_4/SumSumtf.math.multiply_4/Mul:z:03tf.math.reduce_sum_4/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.reduce_sum_4/SumÖ
IdentityIdentity!tf.math.reduce_sum_4/Sum:output:0^embedding_12/embedding_lookup^embedding_13/embedding_lookup^embedding_14/embedding_lookup*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: :::2>
embedding_12/embedding_lookupembedding_12/embedding_lookup2>
embedding_13/embedding_lookupembedding_13/embedding_lookup2>
embedding_14/embedding_lookupembedding_14/embedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: 
	
à
G__inference_dense_25_layer_call_and_return_conditional_losses_200072926

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ô
v
0__inference_embedding_13_layer_call_fn_200073042

inputs
unknown
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_embedding_13_layer_call_and_return_conditional_losses_2000708552
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ù
Ù
'__inference_signature_wrapper_200072082
bets

cards0

cards1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

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

unknown_29
identity¢StatefulPartitionedCallì
StatefulPartitionedCallStatefulPartitionedCallcards0cards1betsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_29*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*<
_read_only_resource_inputs
	
 !*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference__wrapped_model_2000707712
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*À
_input_shapes®
«:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : :::: :::: : ::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:M I
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


_user_specified_namebets:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namecards0:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namecards1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ó	

K__inference_embedding_13_layer_call_and_return_conditional_losses_200070855

inputs
embedding_lookup_200070849
identity¢embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Cast
embedding_lookupResourceGatherembedding_lookup_200070849Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*-
_class#
!loc:@embedding_lookup/200070849*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_lookupñ
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@embedding_lookup/200070849*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity¡
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ó	

K__inference_embedding_15_layer_call_and_return_conditional_losses_200071057

inputs
embedding_lookup_200071051
identity¢embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Cast
embedding_lookupResourceGatherembedding_lookup_200071051Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*-
_class#
!loc:@embedding_lookup/200071051*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_lookupñ
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@embedding_lookup/200071051*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity¡
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÕX
É

M__inference_custom_model_2_layer_call_and_return_conditional_losses_200071680

cards0

cards1
bets*
&tf_math_greater_equal_8_greaterequal_y
model_4_200071595
model_4_200071597
model_4_200071599
model_4_200071601
model_5_200071604
model_5_200071606
model_5_200071608
model_5_200071610.
*tf_clip_by_value_8_clip_by_value_minimum_y&
"tf_clip_by_value_8_clip_by_value_y
dense_18_200071622
dense_18_200071624
dense_21_200071627
dense_21_200071629
dense_19_200071632
dense_19_200071634
dense_20_200071637
dense_20_200071639
dense_22_200071642
dense_22_200071644
dense_23_200071649
dense_23_200071651
dense_24_200071655
dense_24_200071657
dense_25_200071662
dense_25_200071664
normalize_2_200071669
normalize_2_200071671
dense_26_200071674
dense_26_200071676
identity¢ dense_18/StatefulPartitionedCall¢ dense_19/StatefulPartitionedCall¢ dense_20/StatefulPartitionedCall¢ dense_21/StatefulPartitionedCall¢ dense_22/StatefulPartitionedCall¢ dense_23/StatefulPartitionedCall¢ dense_24/StatefulPartitionedCall¢ dense_25/StatefulPartitionedCall¢ dense_26/StatefulPartitionedCall¢model_4/StatefulPartitionedCall¢model_5/StatefulPartitionedCall¢#normalize_2/StatefulPartitionedCall¼
$tf.math.greater_equal_8/GreaterEqualGreaterEqualbets&tf_math_greater_equal_8_greaterequal_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2&
$tf.math.greater_equal_8/GreaterEqualÂ
model_4/StatefulPartitionedCallStatefulPartitionedCallcards0model_4_200071595model_4_200071597model_4_200071599model_4_200071601*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_model_4_layer_call_and_return_conditional_losses_2000709862!
model_4/StatefulPartitionedCallÂ
model_5/StatefulPartitionedCallStatefulPartitionedCallcards1model_5_200071604model_5_200071606model_5_200071608model_5_200071610*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_model_5_layer_call_and_return_conditional_losses_2000712122!
model_5/StatefulPartitionedCallÃ
(tf.clip_by_value_8/clip_by_value/MinimumMinimumbets*tf_clip_by_value_8_clip_by_value_minimum_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2*
(tf.clip_by_value_8/clip_by_value/MinimumÓ
 tf.clip_by_value_8/clip_by_valueMaximum,tf.clip_by_value_8/clip_by_value/Minimum:z:0"tf_clip_by_value_8_clip_by_value_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2"
 tf.clip_by_value_8/clip_by_value
tf.cast_8/CastCast(tf.math.greater_equal_8/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
tf.cast_8/Castt
tf.concat_6/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_6/concat/axisæ
tf.concat_6/concatConcatV2(model_4/StatefulPartitionedCall:output:0(model_5/StatefulPartitionedCall:output:0 tf.concat_6/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.concat_6/concat}
tf.concat_7/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
tf.concat_7/concat/axisË
tf.concat_7/concatConcatV2$tf.clip_by_value_8/clip_by_value:z:0tf.cast_8/Cast:y:0 tf.concat_7/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.concat_7/concat³
 dense_18/StatefulPartitionedCallStatefulPartitionedCalltf.concat_6/concat:output:0dense_18_200071622dense_18_200071624*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_18_layer_call_and_return_conditional_losses_2000713212"
 dense_18/StatefulPartitionedCall³
 dense_21/StatefulPartitionedCallStatefulPartitionedCalltf.concat_7/concat:output:0dense_21_200071627dense_21_200071629*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_21_layer_call_and_return_conditional_losses_2000713472"
 dense_21/StatefulPartitionedCallÁ
 dense_19/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0dense_19_200071632dense_19_200071634*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_19_layer_call_and_return_conditional_losses_2000713742"
 dense_19/StatefulPartitionedCallÁ
 dense_20/StatefulPartitionedCallStatefulPartitionedCall)dense_19/StatefulPartitionedCall:output:0dense_20_200071637dense_20_200071639*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_20_layer_call_and_return_conditional_losses_2000714012"
 dense_20/StatefulPartitionedCallÁ
 dense_22/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0dense_22_200071642dense_22_200071644*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_22_layer_call_and_return_conditional_losses_2000714272"
 dense_22/StatefulPartitionedCall}
tf.concat_8/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
tf.concat_8/concat/axisè
tf.concat_8/concatConcatV2)dense_20/StatefulPartitionedCall:output:0)dense_22/StatefulPartitionedCall:output:0 tf.concat_8/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.concat_8/concat³
 dense_23/StatefulPartitionedCallStatefulPartitionedCalltf.concat_8/concat:output:0dense_23_200071649dense_23_200071651*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_23_layer_call_and_return_conditional_losses_2000714552"
 dense_23/StatefulPartitionedCall
tf.nn.relu_6/ReluRelu)dense_23/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.nn.relu_6/Relu·
 dense_24/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_6/Relu:activations:0dense_24_200071655dense_24_200071657*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_24_layer_call_and_return_conditional_losses_2000714822"
 dense_24/StatefulPartitionedCallÆ
tf.__operators__.add_16/AddV2AddV2)dense_24/StatefulPartitionedCall:output:0tf.nn.relu_6/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_16/AddV2
tf.nn.relu_7/ReluRelu!tf.__operators__.add_16/AddV2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.nn.relu_7/Relu·
 dense_25/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_7/Relu:activations:0dense_25_200071662dense_25_200071664*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_25_layer_call_and_return_conditional_losses_2000715102"
 dense_25/StatefulPartitionedCallÆ
tf.__operators__.add_17/AddV2AddV2)dense_25/StatefulPartitionedCall:output:0tf.nn.relu_7/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_17/AddV2
tf.nn.relu_8/ReluRelu!tf.__operators__.add_17/AddV2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.nn.relu_8/ReluÆ
#normalize_2/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_8/Relu:activations:0normalize_2_200071669normalize_2_200071671*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_normalize_2_layer_call_and_return_conditional_losses_2000715452%
#normalize_2/StatefulPartitionedCallÃ
 dense_26/StatefulPartitionedCallStatefulPartitionedCall,normalize_2/StatefulPartitionedCall:output:0dense_26_200071674dense_26_200071676*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_26_layer_call_and_return_conditional_losses_2000715712"
 dense_26/StatefulPartitionedCall¢
IdentityIdentity)dense_26/StatefulPartitionedCall:output:0!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall ^model_4/StatefulPartitionedCall ^model_5/StatefulPartitionedCall$^normalize_2/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*À
_input_shapes®
«:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
: : :::: :::: : ::::::::::::::::::::2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2B
model_4/StatefulPartitionedCallmodel_4/StatefulPartitionedCall2B
model_5/StatefulPartitionedCallmodel_5/StatefulPartitionedCall2J
#normalize_2/StatefulPartitionedCall#normalize_2/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namecards0:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namecards1:MI
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


_user_specified_namebets:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
®
d
H__inference_flatten_5_layer_call_and_return_conditional_losses_200071007

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ-
Û
F__inference_model_4_layer_call_and_return_conditional_losses_200070874
input_5*
&tf_math_greater_equal_6_greaterequal_y
embedding_14_200070818
embedding_12_200070840
embedding_13_200070864
identity¢$embedding_12/StatefulPartitionedCall¢$embedding_13/StatefulPartitionedCall¢$embedding_14/StatefulPartitionedCallÛ
flatten_4/PartitionedCallPartitionedCallinput_5*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_flatten_4_layer_call_and_return_conditional_losses_2000707812
flatten_4/PartitionedCall
*tf.clip_by_value_6/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2,
*tf.clip_by_value_6/clip_by_value/Minimum/yê
(tf.clip_by_value_6/clip_by_value/MinimumMinimum"flatten_4/PartitionedCall:output:03tf.clip_by_value_6/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(tf.clip_by_value_6/clip_by_value/Minimum
"tf.clip_by_value_6/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_6/clip_by_value/yÜ
 tf.clip_by_value_6/clip_by_valueMaximum,tf.clip_by_value_6/clip_by_value/Minimum:z:0+tf.clip_by_value_6/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 tf.clip_by_value_6/clip_by_value
#tf.compat.v1.floor_div_4/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2%
#tf.compat.v1.floor_div_4/FloorDiv/yØ
!tf.compat.v1.floor_div_4/FloorDivFloorDiv$tf.clip_by_value_6/clip_by_value:z:0,tf.compat.v1.floor_div_4/FloorDiv/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!tf.compat.v1.floor_div_4/FloorDivÚ
$tf.math.greater_equal_6/GreaterEqualGreaterEqual"flatten_4/PartitionedCall:output:0&tf_math_greater_equal_6_greaterequal_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$tf.math.greater_equal_6/GreaterEqual
tf.math.floormod_4/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2
tf.math.floormod_4/FloorMod/yÆ
tf.math.floormod_4/FloorModFloorMod$tf.clip_by_value_6/clip_by_value:z:0&tf.math.floormod_4/FloorMod/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.floormod_4/FloorModº
$embedding_14/StatefulPartitionedCallStatefulPartitionedCall$tf.clip_by_value_6/clip_by_value:z:0embedding_14_200070818*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_embedding_14_layer_call_and_return_conditional_losses_2000708092&
$embedding_14/StatefulPartitionedCall»
$embedding_12/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.floor_div_4/FloorDiv:z:0embedding_12_200070840*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_embedding_12_layer_call_and_return_conditional_losses_2000708312&
$embedding_12/StatefulPartitionedCall
tf.cast_6/CastCast(tf.math.greater_equal_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.cast_6/CastÜ
tf.__operators__.add_12/AddV2AddV2-embedding_14/StatefulPartitionedCall:output:0-embedding_12/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_12/AddV2µ
$embedding_13/StatefulPartitionedCallStatefulPartitionedCalltf.math.floormod_4/FloorMod:z:0embedding_13_200070864*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_embedding_13_layer_call_and_return_conditional_losses_2000708552&
$embedding_13/StatefulPartitionedCallÐ
tf.__operators__.add_13/AddV2AddV2!tf.__operators__.add_12/AddV2:z:0-embedding_13/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_13/AddV2
tf.expand_dims_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2!
tf.expand_dims_4/ExpandDims/dim¼
tf.expand_dims_4/ExpandDims
ExpandDimstf.cast_6/Cast:y:0(tf.expand_dims_4/ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.expand_dims_4/ExpandDims·
tf.math.multiply_4/MulMul!tf.__operators__.add_13/AddV2:z:0$tf.expand_dims_4/ExpandDims:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.multiply_4/Mul
*tf.math.reduce_sum_4/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_4/Sum/reduction_indices¿
tf.math.reduce_sum_4/SumSumtf.math.multiply_4/Mul:z:03tf.math.reduce_sum_4/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.reduce_sum_4/Sumë
IdentityIdentity!tf.math.reduce_sum_4/Sum:output:0%^embedding_12/StatefulPartitionedCall%^embedding_13/StatefulPartitionedCall%^embedding_14/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: :::2L
$embedding_12/StatefulPartitionedCall$embedding_12/StatefulPartitionedCall2L
$embedding_13/StatefulPartitionedCall$embedding_13/StatefulPartitionedCall2L
$embedding_14/StatefulPartitionedCall$embedding_14/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_5:

_output_shapes
: 
Ô
v
0__inference_embedding_15_layer_call_fn_200073087

inputs
unknown
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_embedding_15_layer_call_and_return_conditional_losses_2000710572
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ì
¤
J__inference_normalize_2_layer_call_and_return_conditional_losses_200072952
x3
/normalization_2_reshape_readvariableop_resource5
1normalization_2_reshape_1_readvariableop_resource
identity¢&normalization_2/Reshape/ReadVariableOp¢(normalization_2/Reshape_1/ReadVariableOp½
&normalization_2/Reshape/ReadVariableOpReadVariableOp/normalization_2_reshape_readvariableop_resource*
_output_shapes	
:*
dtype02(
&normalization_2/Reshape/ReadVariableOp
normalization_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_2/Reshape/shape¿
normalization_2/ReshapeReshape.normalization_2/Reshape/ReadVariableOp:value:0&normalization_2/Reshape/shape:output:0*
T0*
_output_shapes
:	2
normalization_2/ReshapeÃ
(normalization_2/Reshape_1/ReadVariableOpReadVariableOp1normalization_2_reshape_1_readvariableop_resource*
_output_shapes	
:*
dtype02*
(normalization_2/Reshape_1/ReadVariableOp
normalization_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_2/Reshape_1/shapeÇ
normalization_2/Reshape_1Reshape0normalization_2/Reshape_1/ReadVariableOp:value:0(normalization_2/Reshape_1/shape:output:0*
T0*
_output_shapes
:	2
normalization_2/Reshape_1
normalization_2/subSubx normalization_2/Reshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
normalization_2/sub
normalization_2/SqrtSqrt"normalization_2/Reshape_1:output:0*
T0*
_output_shapes
:	2
normalization_2/Sqrt{
normalization_2/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö32
normalization_2/Maximum/y¥
normalization_2/MaximumMaximumnormalization_2/Sqrt:y:0"normalization_2/Maximum/y:output:0*
T0*
_output_shapes
:	2
normalization_2/Maximum¦
normalization_2/truedivRealDivnormalization_2/sub:z:0normalization_2/Maximum:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
normalization_2/truedivÄ
IdentityIdentitynormalization_2/truediv:z:0'^normalization_2/Reshape/ReadVariableOp)^normalization_2/Reshape_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::2P
&normalization_2/Reshape/ReadVariableOp&normalization_2/Reshape/ReadVariableOp2T
(normalization_2/Reshape_1/ReadVariableOp(normalization_2/Reshape_1/ReadVariableOp:K G
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
®
d
H__inference_flatten_4_layer_call_and_return_conditional_losses_200072986

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Î-
Ú
F__inference_model_4_layer_call_and_return_conditional_losses_200070986

inputs*
&tf_math_greater_equal_6_greaterequal_y
embedding_14_200070968
embedding_12_200070971
embedding_13_200070976
identity¢$embedding_12/StatefulPartitionedCall¢$embedding_13/StatefulPartitionedCall¢$embedding_14/StatefulPartitionedCallÚ
flatten_4/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_flatten_4_layer_call_and_return_conditional_losses_2000707812
flatten_4/PartitionedCall
*tf.clip_by_value_6/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2,
*tf.clip_by_value_6/clip_by_value/Minimum/yê
(tf.clip_by_value_6/clip_by_value/MinimumMinimum"flatten_4/PartitionedCall:output:03tf.clip_by_value_6/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(tf.clip_by_value_6/clip_by_value/Minimum
"tf.clip_by_value_6/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_6/clip_by_value/yÜ
 tf.clip_by_value_6/clip_by_valueMaximum,tf.clip_by_value_6/clip_by_value/Minimum:z:0+tf.clip_by_value_6/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 tf.clip_by_value_6/clip_by_value
#tf.compat.v1.floor_div_4/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2%
#tf.compat.v1.floor_div_4/FloorDiv/yØ
!tf.compat.v1.floor_div_4/FloorDivFloorDiv$tf.clip_by_value_6/clip_by_value:z:0,tf.compat.v1.floor_div_4/FloorDiv/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!tf.compat.v1.floor_div_4/FloorDivÚ
$tf.math.greater_equal_6/GreaterEqualGreaterEqual"flatten_4/PartitionedCall:output:0&tf_math_greater_equal_6_greaterequal_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$tf.math.greater_equal_6/GreaterEqual
tf.math.floormod_4/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2
tf.math.floormod_4/FloorMod/yÆ
tf.math.floormod_4/FloorModFloorMod$tf.clip_by_value_6/clip_by_value:z:0&tf.math.floormod_4/FloorMod/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.floormod_4/FloorModº
$embedding_14/StatefulPartitionedCallStatefulPartitionedCall$tf.clip_by_value_6/clip_by_value:z:0embedding_14_200070968*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_embedding_14_layer_call_and_return_conditional_losses_2000708092&
$embedding_14/StatefulPartitionedCall»
$embedding_12/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.floor_div_4/FloorDiv:z:0embedding_12_200070971*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_embedding_12_layer_call_and_return_conditional_losses_2000708312&
$embedding_12/StatefulPartitionedCall
tf.cast_6/CastCast(tf.math.greater_equal_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.cast_6/CastÜ
tf.__operators__.add_12/AddV2AddV2-embedding_14/StatefulPartitionedCall:output:0-embedding_12/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_12/AddV2µ
$embedding_13/StatefulPartitionedCallStatefulPartitionedCalltf.math.floormod_4/FloorMod:z:0embedding_13_200070976*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_embedding_13_layer_call_and_return_conditional_losses_2000708552&
$embedding_13/StatefulPartitionedCallÐ
tf.__operators__.add_13/AddV2AddV2!tf.__operators__.add_12/AddV2:z:0-embedding_13/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_13/AddV2
tf.expand_dims_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2!
tf.expand_dims_4/ExpandDims/dim¼
tf.expand_dims_4/ExpandDims
ExpandDimstf.cast_6/Cast:y:0(tf.expand_dims_4/ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.expand_dims_4/ExpandDims·
tf.math.multiply_4/MulMul!tf.__operators__.add_13/AddV2:z:0$tf.expand_dims_4/ExpandDims:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.multiply_4/Mul
*tf.math.reduce_sum_4/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_4/Sum/reduction_indices¿
tf.math.reduce_sum_4/SumSumtf.math.multiply_4/Mul:z:03tf.math.reduce_sum_4/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.reduce_sum_4/Sumë
IdentityIdentity!tf.math.reduce_sum_4/Sum:output:0%^embedding_12/StatefulPartitionedCall%^embedding_13/StatefulPartitionedCall%^embedding_14/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: :::2L
$embedding_12/StatefulPartitionedCall$embedding_12/StatefulPartitionedCall2L
$embedding_13/StatefulPartitionedCall$embedding_13/StatefulPartitionedCall2L
$embedding_14/StatefulPartitionedCall$embedding_14/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: 
	
à
G__inference_dense_23_layer_call_and_return_conditional_losses_200071455

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ô
v
0__inference_embedding_17_layer_call_fn_200073070

inputs
unknown
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_embedding_17_layer_call_and_return_conditional_losses_2000710352
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Á

+__inference_model_5_layer_call_fn_200071223
input_6
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_6unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_model_5_layer_call_and_return_conditional_losses_2000712122
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: :::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_6:

_output_shapes
: 
ÕX
É

M__inference_custom_model_2_layer_call_and_return_conditional_losses_200071588

cards0

cards1
bets*
&tf_math_greater_equal_8_greaterequal_y
model_4_200071257
model_4_200071259
model_4_200071261
model_4_200071263
model_5_200071292
model_5_200071294
model_5_200071296
model_5_200071298.
*tf_clip_by_value_8_clip_by_value_minimum_y&
"tf_clip_by_value_8_clip_by_value_y
dense_18_200071332
dense_18_200071334
dense_21_200071358
dense_21_200071360
dense_19_200071385
dense_19_200071387
dense_20_200071412
dense_20_200071414
dense_22_200071438
dense_22_200071440
dense_23_200071466
dense_23_200071468
dense_24_200071493
dense_24_200071495
dense_25_200071521
dense_25_200071523
normalize_2_200071556
normalize_2_200071558
dense_26_200071582
dense_26_200071584
identity¢ dense_18/StatefulPartitionedCall¢ dense_19/StatefulPartitionedCall¢ dense_20/StatefulPartitionedCall¢ dense_21/StatefulPartitionedCall¢ dense_22/StatefulPartitionedCall¢ dense_23/StatefulPartitionedCall¢ dense_24/StatefulPartitionedCall¢ dense_25/StatefulPartitionedCall¢ dense_26/StatefulPartitionedCall¢model_4/StatefulPartitionedCall¢model_5/StatefulPartitionedCall¢#normalize_2/StatefulPartitionedCall¼
$tf.math.greater_equal_8/GreaterEqualGreaterEqualbets&tf_math_greater_equal_8_greaterequal_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2&
$tf.math.greater_equal_8/GreaterEqualÂ
model_4/StatefulPartitionedCallStatefulPartitionedCallcards0model_4_200071257model_4_200071259model_4_200071261model_4_200071263*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_model_4_layer_call_and_return_conditional_losses_2000709412!
model_4/StatefulPartitionedCallÂ
model_5/StatefulPartitionedCallStatefulPartitionedCallcards1model_5_200071292model_5_200071294model_5_200071296model_5_200071298*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_model_5_layer_call_and_return_conditional_losses_2000711672!
model_5/StatefulPartitionedCallÃ
(tf.clip_by_value_8/clip_by_value/MinimumMinimumbets*tf_clip_by_value_8_clip_by_value_minimum_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2*
(tf.clip_by_value_8/clip_by_value/MinimumÓ
 tf.clip_by_value_8/clip_by_valueMaximum,tf.clip_by_value_8/clip_by_value/Minimum:z:0"tf_clip_by_value_8_clip_by_value_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2"
 tf.clip_by_value_8/clip_by_value
tf.cast_8/CastCast(tf.math.greater_equal_8/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
tf.cast_8/Castt
tf.concat_6/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_6/concat/axisæ
tf.concat_6/concatConcatV2(model_4/StatefulPartitionedCall:output:0(model_5/StatefulPartitionedCall:output:0 tf.concat_6/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.concat_6/concat}
tf.concat_7/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
tf.concat_7/concat/axisË
tf.concat_7/concatConcatV2$tf.clip_by_value_8/clip_by_value:z:0tf.cast_8/Cast:y:0 tf.concat_7/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.concat_7/concat³
 dense_18/StatefulPartitionedCallStatefulPartitionedCalltf.concat_6/concat:output:0dense_18_200071332dense_18_200071334*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_18_layer_call_and_return_conditional_losses_2000713212"
 dense_18/StatefulPartitionedCall³
 dense_21/StatefulPartitionedCallStatefulPartitionedCalltf.concat_7/concat:output:0dense_21_200071358dense_21_200071360*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_21_layer_call_and_return_conditional_losses_2000713472"
 dense_21/StatefulPartitionedCallÁ
 dense_19/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0dense_19_200071385dense_19_200071387*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_19_layer_call_and_return_conditional_losses_2000713742"
 dense_19/StatefulPartitionedCallÁ
 dense_20/StatefulPartitionedCallStatefulPartitionedCall)dense_19/StatefulPartitionedCall:output:0dense_20_200071412dense_20_200071414*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_20_layer_call_and_return_conditional_losses_2000714012"
 dense_20/StatefulPartitionedCallÁ
 dense_22/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0dense_22_200071438dense_22_200071440*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_22_layer_call_and_return_conditional_losses_2000714272"
 dense_22/StatefulPartitionedCall}
tf.concat_8/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
tf.concat_8/concat/axisè
tf.concat_8/concatConcatV2)dense_20/StatefulPartitionedCall:output:0)dense_22/StatefulPartitionedCall:output:0 tf.concat_8/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.concat_8/concat³
 dense_23/StatefulPartitionedCallStatefulPartitionedCalltf.concat_8/concat:output:0dense_23_200071466dense_23_200071468*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_23_layer_call_and_return_conditional_losses_2000714552"
 dense_23/StatefulPartitionedCall
tf.nn.relu_6/ReluRelu)dense_23/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.nn.relu_6/Relu·
 dense_24/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_6/Relu:activations:0dense_24_200071493dense_24_200071495*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_24_layer_call_and_return_conditional_losses_2000714822"
 dense_24/StatefulPartitionedCallÆ
tf.__operators__.add_16/AddV2AddV2)dense_24/StatefulPartitionedCall:output:0tf.nn.relu_6/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_16/AddV2
tf.nn.relu_7/ReluRelu!tf.__operators__.add_16/AddV2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.nn.relu_7/Relu·
 dense_25/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_7/Relu:activations:0dense_25_200071521dense_25_200071523*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_25_layer_call_and_return_conditional_losses_2000715102"
 dense_25/StatefulPartitionedCallÆ
tf.__operators__.add_17/AddV2AddV2)dense_25/StatefulPartitionedCall:output:0tf.nn.relu_7/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_17/AddV2
tf.nn.relu_8/ReluRelu!tf.__operators__.add_17/AddV2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.nn.relu_8/ReluÆ
#normalize_2/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_8/Relu:activations:0normalize_2_200071556normalize_2_200071558*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_normalize_2_layer_call_and_return_conditional_losses_2000715452%
#normalize_2/StatefulPartitionedCallÃ
 dense_26/StatefulPartitionedCallStatefulPartitionedCall,normalize_2/StatefulPartitionedCall:output:0dense_26_200071582dense_26_200071584*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_26_layer_call_and_return_conditional_losses_2000715712"
 dense_26/StatefulPartitionedCall¢
IdentityIdentity)dense_26/StatefulPartitionedCall:output:0!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall ^model_4/StatefulPartitionedCall ^model_5/StatefulPartitionedCall$^normalize_2/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*À
_input_shapes®
«:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
: : :::: :::: : ::::::::::::::::::::2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2B
model_4/StatefulPartitionedCallmodel_4/StatefulPartitionedCall2B
model_5/StatefulPartitionedCallmodel_5/StatefulPartitionedCall2J
#normalize_2/StatefulPartitionedCall#normalize_2/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namecards0:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namecards1:MI
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


_user_specified_namebets:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
9
ø
F__inference_model_5_layer_call_and_return_conditional_losses_200072754

inputs*
&tf_math_greater_equal_7_greaterequal_y+
'embedding_17_embedding_lookup_200072728+
'embedding_15_embedding_lookup_200072734+
'embedding_16_embedding_lookup_200072742
identity¢embedding_15/embedding_lookup¢embedding_16/embedding_lookup¢embedding_17/embedding_lookups
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
flatten_5/Const
flatten_5/ReshapeReshapeinputsflatten_5/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flatten_5/Reshape
*tf.clip_by_value_7/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2,
*tf.clip_by_value_7/clip_by_value/Minimum/yâ
(tf.clip_by_value_7/clip_by_value/MinimumMinimumflatten_5/Reshape:output:03tf.clip_by_value_7/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(tf.clip_by_value_7/clip_by_value/Minimum
"tf.clip_by_value_7/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_7/clip_by_value/yÜ
 tf.clip_by_value_7/clip_by_valueMaximum,tf.clip_by_value_7/clip_by_value/Minimum:z:0+tf.clip_by_value_7/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 tf.clip_by_value_7/clip_by_value
#tf.compat.v1.floor_div_5/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2%
#tf.compat.v1.floor_div_5/FloorDiv/yØ
!tf.compat.v1.floor_div_5/FloorDivFloorDiv$tf.clip_by_value_7/clip_by_value:z:0,tf.compat.v1.floor_div_5/FloorDiv/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!tf.compat.v1.floor_div_5/FloorDivÒ
$tf.math.greater_equal_7/GreaterEqualGreaterEqualflatten_5/Reshape:output:0&tf_math_greater_equal_7_greaterequal_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$tf.math.greater_equal_7/GreaterEqual
tf.math.floormod_5/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @2
tf.math.floormod_5/FloorMod/yÆ
tf.math.floormod_5/FloorModFloorMod$tf.clip_by_value_7/clip_by_value:z:0&tf.math.floormod_5/FloorMod/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.floormod_5/FloorMod
embedding_17/CastCast$tf.clip_by_value_7/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_17/CastÅ
embedding_17/embedding_lookupResourceGather'embedding_17_embedding_lookup_200072728embedding_17/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*:
_class0
.,loc:@embedding_17/embedding_lookup/200072728*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_17/embedding_lookup¥
&embedding_17/embedding_lookup/IdentityIdentity&embedding_17/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*:
_class0
.,loc:@embedding_17/embedding_lookup/200072728*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&embedding_17/embedding_lookup/IdentityÈ
(embedding_17/embedding_lookup/Identity_1Identity/embedding_17/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(embedding_17/embedding_lookup/Identity_1
embedding_15/CastCast%tf.compat.v1.floor_div_5/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_15/CastÅ
embedding_15/embedding_lookupResourceGather'embedding_15_embedding_lookup_200072734embedding_15/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*:
_class0
.,loc:@embedding_15/embedding_lookup/200072734*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_15/embedding_lookup¥
&embedding_15/embedding_lookup/IdentityIdentity&embedding_15/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*:
_class0
.,loc:@embedding_15/embedding_lookup/200072734*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&embedding_15/embedding_lookup/IdentityÈ
(embedding_15/embedding_lookup/Identity_1Identity/embedding_15/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(embedding_15/embedding_lookup/Identity_1
tf.cast_7/CastCast(tf.math.greater_equal_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.cast_7/Castä
tf.__operators__.add_14/AddV2AddV21embedding_17/embedding_lookup/Identity_1:output:01embedding_15/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_14/AddV2
embedding_16/CastCasttf.math.floormod_5/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_16/CastÅ
embedding_16/embedding_lookupResourceGather'embedding_16_embedding_lookup_200072742embedding_16/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*:
_class0
.,loc:@embedding_16/embedding_lookup/200072742*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_16/embedding_lookup¥
&embedding_16/embedding_lookup/IdentityIdentity&embedding_16/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*:
_class0
.,loc:@embedding_16/embedding_lookup/200072742*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&embedding_16/embedding_lookup/IdentityÈ
(embedding_16/embedding_lookup/Identity_1Identity/embedding_16/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(embedding_16/embedding_lookup/Identity_1Ô
tf.__operators__.add_15/AddV2AddV2!tf.__operators__.add_14/AddV2:z:01embedding_16/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.__operators__.add_15/AddV2
tf.expand_dims_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2!
tf.expand_dims_5/ExpandDims/dim¼
tf.expand_dims_5/ExpandDims
ExpandDimstf.cast_7/Cast:y:0(tf.expand_dims_5/ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.expand_dims_5/ExpandDims·
tf.math.multiply_5/MulMul!tf.__operators__.add_15/AddV2:z:0$tf.expand_dims_5/ExpandDims:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.multiply_5/Mul
*tf.math.reduce_sum_5/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_5/Sum/reduction_indices¿
tf.math.reduce_sum_5/SumSumtf.math.multiply_5/Mul:z:03tf.math.reduce_sum_5/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf.math.reduce_sum_5/SumÖ
IdentityIdentity!tf.math.reduce_sum_5/Sum:output:0^embedding_15/embedding_lookup^embedding_16/embedding_lookup^embedding_17/embedding_lookup*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: :::2>
embedding_15/embedding_lookupembedding_15/embedding_lookup2>
embedding_16/embedding_lookupembedding_16/embedding_lookup2>
embedding_17/embedding_lookupembedding_17/embedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: "±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*
serving_default
5
bets-
serving_default_bets:0ÿÿÿÿÿÿÿÿÿ

9
cards0/
serving_default_cards0:0ÿÿÿÿÿÿÿÿÿ
9
cards1/
serving_default_cards1:0ÿÿÿÿÿÿÿÿÿ<
dense_260
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:Ûö
¢
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer-5
layer-6
layer-7
	layer-8

layer_with_weights-2

layer-9
layer-10
layer_with_weights-3
layer-11
layer_with_weights-4
layer-12
layer_with_weights-5
layer-13
layer_with_weights-6
layer-14
layer-15
layer_with_weights-7
layer-16
layer-17
layer_with_weights-8
layer-18
layer-19
layer-20
layer_with_weights-9
layer-21
layer-22
layer-23
layer_with_weights-10
layer-24
layer_with_weights-11
layer-25
	optimizer
loss
	variables
trainable_variables
regularization_losses
 	keras_api
!
signatures
ñ_default_save_signature
+ò&call_and_return_all_conditional_losses
ó__call__"
_tf_keras_networkû{"class_name": "CustomModel", "name": "custom_model_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "custom_model_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cards0"}, "name": "cards0", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cards1"}, "name": "cards1", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "bets"}, "name": "bets", "inbound_nodes": []}, {"class_name": "Functional", "config": {"name": "model_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}, "name": "input_5", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_4", "inbound_nodes": [[["input_5", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_6", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_6", "inbound_nodes": [["flatten_4", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_4", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_4", "inbound_nodes": [["tf.clip_by_value_6", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_14", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_14", "inbound_nodes": [[["tf.clip_by_value_6", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_12", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_12", "inbound_nodes": [[["tf.compat.v1.floor_div_4", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_4", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_4", "inbound_nodes": [["tf.clip_by_value_6", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_6", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_6", "inbound_nodes": [["flatten_4", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_12", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_12", "inbound_nodes": [["embedding_14", 0, 0, {"y": ["embedding_12", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_13", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_13", "inbound_nodes": [[["tf.math.floormod_4", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_6", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_6", "inbound_nodes": [["tf.math.greater_equal_6", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_13", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_13", "inbound_nodes": [["tf.__operators__.add_12", 0, 0, {"y": ["embedding_13", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_4", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_4", "inbound_nodes": [["tf.cast_6", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_4", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_4", "inbound_nodes": [["tf.__operators__.add_13", 0, 0, {"y": ["tf.expand_dims_4", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_4", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_4", "inbound_nodes": [["tf.math.multiply_4", 0, 0, {"axis": 1}]]}], "input_layers": [["input_5", 0, 0]], "output_layers": [["tf.math.reduce_sum_4", 0, 0]]}, "name": "model_4", "inbound_nodes": [[["cards0", 0, 0, {}]]]}, {"class_name": "Functional", "config": {"name": "model_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_6"}, "name": "input_6", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_5", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_5", "inbound_nodes": [[["input_6", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_7", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_7", "inbound_nodes": [["flatten_5", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_5", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_5", "inbound_nodes": [["tf.clip_by_value_7", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_17", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_17", "inbound_nodes": [[["tf.clip_by_value_7", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_15", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_15", "inbound_nodes": [[["tf.compat.v1.floor_div_5", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_5", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_5", "inbound_nodes": [["tf.clip_by_value_7", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_7", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_7", "inbound_nodes": [["flatten_5", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_14", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_14", "inbound_nodes": [["embedding_17", 0, 0, {"y": ["embedding_15", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_16", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_16", "inbound_nodes": [[["tf.math.floormod_5", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_7", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_7", "inbound_nodes": [["tf.math.greater_equal_7", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_15", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_15", "inbound_nodes": [["tf.__operators__.add_14", 0, 0, {"y": ["embedding_16", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_5", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_5", "inbound_nodes": [["tf.cast_7", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_5", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_5", "inbound_nodes": [["tf.__operators__.add_15", 0, 0, {"y": ["tf.expand_dims_5", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_5", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_5", "inbound_nodes": [["tf.math.multiply_5", 0, 0, {"axis": 1}]]}], "input_layers": [["input_6", 0, 0]], "output_layers": [["tf.math.reduce_sum_5", 0, 0]]}, "name": "model_5", "inbound_nodes": [[["cards1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_8", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_8", "inbound_nodes": [["bets", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_6", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_6", "inbound_nodes": [[["model_4", 1, 0, {"axis": 1}], ["model_5", 1, 0, {"axis": 1}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_8", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_8", "inbound_nodes": [["bets", 0, 0, {"clip_value_min": 0.0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_8", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_8", "inbound_nodes": [["tf.math.greater_equal_8", 0, 0, {"dtype": "float32"}]]}, {"class_name": "Dense", "config": {"name": "dense_18", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_18", "inbound_nodes": [[["tf.concat_6", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_7", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_7", "inbound_nodes": [[["tf.clip_by_value_8", 0, 0, {"axis": -1}], ["tf.cast_8", 0, 0, {"axis": -1}]]]}, {"class_name": "Dense", "config": {"name": "dense_19", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_19", "inbound_nodes": [[["dense_18", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_21", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_21", "inbound_nodes": [[["tf.concat_7", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_20", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_20", "inbound_nodes": [[["dense_19", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_22", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_22", "inbound_nodes": [[["dense_21", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_8", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_8", "inbound_nodes": [[["dense_20", 0, 0, {"axis": -1}], ["dense_22", 0, 0, {"axis": -1}]]]}, {"class_name": "Dense", "config": {"name": "dense_23", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_23", "inbound_nodes": [[["tf.concat_8", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_6", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu_6", "inbound_nodes": [["dense_23", 0, 0, {"name": null}]]}, {"class_name": "Dense", "config": {"name": "dense_24", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_24", "inbound_nodes": [[["tf.nn.relu_6", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_16", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_16", "inbound_nodes": [["dense_24", 0, 0, {"y": ["tf.nn.relu_6", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_7", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu_7", "inbound_nodes": [["tf.__operators__.add_16", 0, 0, {"name": null}]]}, {"class_name": "Dense", "config": {"name": "dense_25", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_25", "inbound_nodes": [[["tf.nn.relu_7", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_17", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_17", "inbound_nodes": [["dense_25", 0, 0, {"y": ["tf.nn.relu_7", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_8", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu_8", "inbound_nodes": [["tf.__operators__.add_17", 0, 0, {"name": null}]]}, {"class_name": "Normalize", "config": {"name": "normalize_2", "trainable": true, "dtype": "float32"}, "name": "normalize_2", "inbound_nodes": [[["tf.nn.relu_8", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_26", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Constant", "config": {"value": 0}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_26", "inbound_nodes": [[["normalize_2", 0, 0, {}]]]}], "input_layers": [[["cards0", 0, 0], ["cards1", 0, 0]], ["bets", 0, 0]], "output_layers": [["dense_26", 0, 0]]}, "build_input_shape": [[{"class_name": "TensorShape", "items": [null, 2]}, {"class_name": "TensorShape", "items": [null, 3]}], {"class_name": "TensorShape", "items": [null, 10]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "CustomModel", "config": {"name": "custom_model_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cards0"}, "name": "cards0", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cards1"}, "name": "cards1", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "bets"}, "name": "bets", "inbound_nodes": []}, {"class_name": "Functional", "config": {"name": "model_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}, "name": "input_5", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_4", "inbound_nodes": [[["input_5", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_6", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_6", "inbound_nodes": [["flatten_4", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_4", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_4", "inbound_nodes": [["tf.clip_by_value_6", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_14", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_14", "inbound_nodes": [[["tf.clip_by_value_6", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_12", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_12", "inbound_nodes": [[["tf.compat.v1.floor_div_4", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_4", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_4", "inbound_nodes": [["tf.clip_by_value_6", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_6", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_6", "inbound_nodes": [["flatten_4", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_12", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_12", "inbound_nodes": [["embedding_14", 0, 0, {"y": ["embedding_12", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_13", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_13", "inbound_nodes": [[["tf.math.floormod_4", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_6", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_6", "inbound_nodes": [["tf.math.greater_equal_6", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_13", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_13", "inbound_nodes": [["tf.__operators__.add_12", 0, 0, {"y": ["embedding_13", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_4", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_4", "inbound_nodes": [["tf.cast_6", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_4", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_4", "inbound_nodes": [["tf.__operators__.add_13", 0, 0, {"y": ["tf.expand_dims_4", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_4", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_4", "inbound_nodes": [["tf.math.multiply_4", 0, 0, {"axis": 1}]]}], "input_layers": [["input_5", 0, 0]], "output_layers": [["tf.math.reduce_sum_4", 0, 0]]}, "name": "model_4", "inbound_nodes": [[["cards0", 0, 0, {}]]]}, {"class_name": "Functional", "config": {"name": "model_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_6"}, "name": "input_6", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_5", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_5", "inbound_nodes": [[["input_6", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_7", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_7", "inbound_nodes": [["flatten_5", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_5", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_5", "inbound_nodes": [["tf.clip_by_value_7", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_17", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_17", "inbound_nodes": [[["tf.clip_by_value_7", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_15", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_15", "inbound_nodes": [[["tf.compat.v1.floor_div_5", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_5", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_5", "inbound_nodes": [["tf.clip_by_value_7", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_7", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_7", "inbound_nodes": [["flatten_5", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_14", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_14", "inbound_nodes": [["embedding_17", 0, 0, {"y": ["embedding_15", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_16", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_16", "inbound_nodes": [[["tf.math.floormod_5", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_7", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_7", "inbound_nodes": [["tf.math.greater_equal_7", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_15", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_15", "inbound_nodes": [["tf.__operators__.add_14", 0, 0, {"y": ["embedding_16", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_5", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_5", "inbound_nodes": [["tf.cast_7", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_5", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_5", "inbound_nodes": [["tf.__operators__.add_15", 0, 0, {"y": ["tf.expand_dims_5", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_5", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_5", "inbound_nodes": [["tf.math.multiply_5", 0, 0, {"axis": 1}]]}], "input_layers": [["input_6", 0, 0]], "output_layers": [["tf.math.reduce_sum_5", 0, 0]]}, "name": "model_5", "inbound_nodes": [[["cards1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_8", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_8", "inbound_nodes": [["bets", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_6", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_6", "inbound_nodes": [[["model_4", 1, 0, {"axis": 1}], ["model_5", 1, 0, {"axis": 1}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_8", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_8", "inbound_nodes": [["bets", 0, 0, {"clip_value_min": 0.0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_8", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_8", "inbound_nodes": [["tf.math.greater_equal_8", 0, 0, {"dtype": "float32"}]]}, {"class_name": "Dense", "config": {"name": "dense_18", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_18", "inbound_nodes": [[["tf.concat_6", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_7", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_7", "inbound_nodes": [[["tf.clip_by_value_8", 0, 0, {"axis": -1}], ["tf.cast_8", 0, 0, {"axis": -1}]]]}, {"class_name": "Dense", "config": {"name": "dense_19", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_19", "inbound_nodes": [[["dense_18", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_21", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_21", "inbound_nodes": [[["tf.concat_7", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_20", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_20", "inbound_nodes": [[["dense_19", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_22", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_22", "inbound_nodes": [[["dense_21", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_8", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_8", "inbound_nodes": [[["dense_20", 0, 0, {"axis": -1}], ["dense_22", 0, 0, {"axis": -1}]]]}, {"class_name": "Dense", "config": {"name": "dense_23", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_23", "inbound_nodes": [[["tf.concat_8", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_6", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu_6", "inbound_nodes": [["dense_23", 0, 0, {"name": null}]]}, {"class_name": "Dense", "config": {"name": "dense_24", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_24", "inbound_nodes": [[["tf.nn.relu_6", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_16", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_16", "inbound_nodes": [["dense_24", 0, 0, {"y": ["tf.nn.relu_6", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_7", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu_7", "inbound_nodes": [["tf.__operators__.add_16", 0, 0, {"name": null}]]}, {"class_name": "Dense", "config": {"name": "dense_25", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_25", "inbound_nodes": [[["tf.nn.relu_7", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_17", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_17", "inbound_nodes": [["dense_25", 0, 0, {"y": ["tf.nn.relu_7", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_8", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu_8", "inbound_nodes": [["tf.__operators__.add_17", 0, 0, {"name": null}]]}, {"class_name": "Normalize", "config": {"name": "normalize_2", "trainable": true, "dtype": "float32"}, "name": "normalize_2", "inbound_nodes": [[["tf.nn.relu_8", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_26", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Constant", "config": {"value": 0}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_26", "inbound_nodes": [[["normalize_2", 0, 0, {}]]]}], "input_layers": [[["cards0", 0, 0], ["cards1", 0, 0]], ["bets", 0, 0]], "output_layers": [["dense_26", 0, 0]]}}, "training_config": {"loss": null, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0020000000949949026, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ç"ä
_tf_keras_input_layerÄ{"class_name": "InputLayer", "name": "cards0", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cards0"}}
ç"ä
_tf_keras_input_layerÄ{"class_name": "InputLayer", "name": "cards1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cards1"}}
å"â
_tf_keras_input_layerÂ{"class_name": "InputLayer", "name": "bets", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "bets"}}
ÅQ
"layer-0
#layer-1
$layer-2
%layer-3
&layer_with_weights-0
&layer-4
'layer_with_weights-1
'layer-5
(layer-6
)layer-7
*layer-8
+layer_with_weights-2
+layer-9
,layer-10
-layer-11
.layer-12
/layer-13
0layer-14
1	variables
2trainable_variables
3regularization_losses
4	keras_api
+ô&call_and_return_all_conditional_losses
õ__call__"N
_tf_keras_networkN{"class_name": "Functional", "name": "model_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}, "name": "input_5", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_4", "inbound_nodes": [[["input_5", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_6", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_6", "inbound_nodes": [["flatten_4", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_4", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_4", "inbound_nodes": [["tf.clip_by_value_6", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_14", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_14", "inbound_nodes": [[["tf.clip_by_value_6", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_12", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_12", "inbound_nodes": [[["tf.compat.v1.floor_div_4", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_4", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_4", "inbound_nodes": [["tf.clip_by_value_6", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_6", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_6", "inbound_nodes": [["flatten_4", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_12", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_12", "inbound_nodes": [["embedding_14", 0, 0, {"y": ["embedding_12", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_13", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_13", "inbound_nodes": [[["tf.math.floormod_4", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_6", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_6", "inbound_nodes": [["tf.math.greater_equal_6", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_13", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_13", "inbound_nodes": [["tf.__operators__.add_12", 0, 0, {"y": ["embedding_13", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_4", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_4", "inbound_nodes": [["tf.cast_6", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_4", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_4", "inbound_nodes": [["tf.__operators__.add_13", 0, 0, {"y": ["tf.expand_dims_4", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_4", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_4", "inbound_nodes": [["tf.math.multiply_4", 0, 0, {"axis": 1}]]}], "input_layers": [["input_5", 0, 0]], "output_layers": [["tf.math.reduce_sum_4", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 2]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}, "name": "input_5", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_4", "inbound_nodes": [[["input_5", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_6", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_6", "inbound_nodes": [["flatten_4", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_4", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_4", "inbound_nodes": [["tf.clip_by_value_6", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_14", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_14", "inbound_nodes": [[["tf.clip_by_value_6", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_12", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_12", "inbound_nodes": [[["tf.compat.v1.floor_div_4", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_4", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_4", "inbound_nodes": [["tf.clip_by_value_6", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_6", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_6", "inbound_nodes": [["flatten_4", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_12", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_12", "inbound_nodes": [["embedding_14", 0, 0, {"y": ["embedding_12", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_13", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_13", "inbound_nodes": [[["tf.math.floormod_4", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_6", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_6", "inbound_nodes": [["tf.math.greater_equal_6", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_13", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_13", "inbound_nodes": [["tf.__operators__.add_12", 0, 0, {"y": ["embedding_13", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_4", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_4", "inbound_nodes": [["tf.cast_6", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_4", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_4", "inbound_nodes": [["tf.__operators__.add_13", 0, 0, {"y": ["tf.expand_dims_4", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_4", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_4", "inbound_nodes": [["tf.math.multiply_4", 0, 0, {"axis": 1}]]}], "input_layers": [["input_5", 0, 0]], "output_layers": [["tf.math.reduce_sum_4", 0, 0]]}}}
ÅQ
5layer-0
6layer-1
7layer-2
8layer-3
9layer_with_weights-0
9layer-4
:layer_with_weights-1
:layer-5
;layer-6
<layer-7
=layer-8
>layer_with_weights-2
>layer-9
?layer-10
@layer-11
Alayer-12
Blayer-13
Clayer-14
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
+ö&call_and_return_all_conditional_losses
÷__call__"N
_tf_keras_networkN{"class_name": "Functional", "name": "model_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_6"}, "name": "input_6", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_5", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_5", "inbound_nodes": [[["input_6", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_7", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_7", "inbound_nodes": [["flatten_5", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_5", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_5", "inbound_nodes": [["tf.clip_by_value_7", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_17", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_17", "inbound_nodes": [[["tf.clip_by_value_7", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_15", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_15", "inbound_nodes": [[["tf.compat.v1.floor_div_5", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_5", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_5", "inbound_nodes": [["tf.clip_by_value_7", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_7", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_7", "inbound_nodes": [["flatten_5", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_14", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_14", "inbound_nodes": [["embedding_17", 0, 0, {"y": ["embedding_15", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_16", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_16", "inbound_nodes": [[["tf.math.floormod_5", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_7", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_7", "inbound_nodes": [["tf.math.greater_equal_7", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_15", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_15", "inbound_nodes": [["tf.__operators__.add_14", 0, 0, {"y": ["embedding_16", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_5", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_5", "inbound_nodes": [["tf.cast_7", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_5", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_5", "inbound_nodes": [["tf.__operators__.add_15", 0, 0, {"y": ["tf.expand_dims_5", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_5", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_5", "inbound_nodes": [["tf.math.multiply_5", 0, 0, {"axis": 1}]]}], "input_layers": [["input_6", 0, 0]], "output_layers": [["tf.math.reduce_sum_5", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 3]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_6"}, "name": "input_6", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_5", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_5", "inbound_nodes": [[["input_6", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_7", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_7", "inbound_nodes": [["flatten_5", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_5", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_5", "inbound_nodes": [["tf.clip_by_value_7", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_17", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_17", "inbound_nodes": [[["tf.clip_by_value_7", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_15", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_15", "inbound_nodes": [[["tf.compat.v1.floor_div_5", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_5", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_5", "inbound_nodes": [["tf.clip_by_value_7", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_7", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_7", "inbound_nodes": [["flatten_5", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_14", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_14", "inbound_nodes": [["embedding_17", 0, 0, {"y": ["embedding_15", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_16", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_16", "inbound_nodes": [[["tf.math.floormod_5", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_7", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_7", "inbound_nodes": [["tf.math.greater_equal_7", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_15", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_15", "inbound_nodes": [["tf.__operators__.add_14", 0, 0, {"y": ["embedding_16", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_5", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_5", "inbound_nodes": [["tf.cast_7", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_5", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_5", "inbound_nodes": [["tf.__operators__.add_15", 0, 0, {"y": ["tf.expand_dims_5", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_5", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_5", "inbound_nodes": [["tf.math.multiply_5", 0, 0, {"axis": 1}]]}], "input_layers": [["input_6", 0, 0]], "output_layers": [["tf.math.reduce_sum_5", 0, 0]]}}}
ù
H	keras_api"ç
_tf_keras_layerÍ{"class_name": "TFOpLambda", "name": "tf.math.greater_equal_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.greater_equal_8", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}}
Õ
I	keras_api"Ã
_tf_keras_layer©{"class_name": "TFOpLambda", "name": "tf.concat_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.concat_6", "trainable": true, "dtype": "float32", "function": "concat"}}
ê
J	keras_api"Ø
_tf_keras_layer¾{"class_name": "TFOpLambda", "name": "tf.clip_by_value_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.clip_by_value_8", "trainable": true, "dtype": "float32", "function": "clip_by_value"}}
Ï
K	keras_api"½
_tf_keras_layer£{"class_name": "TFOpLambda", "name": "tf.cast_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.cast_8", "trainable": true, "dtype": "float32", "function": "cast"}}
÷

Lkernel
Mbias
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
+ø&call_and_return_all_conditional_losses
ù__call__"Ð
_tf_keras_layer¶{"class_name": "Dense", "name": "dense_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_18", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
Õ
R	keras_api"Ã
_tf_keras_layer©{"class_name": "TFOpLambda", "name": "tf.concat_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.concat_7", "trainable": true, "dtype": "float32", "function": "concat"}}
÷

Skernel
Tbias
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
+ú&call_and_return_all_conditional_losses
û__call__"Ð
_tf_keras_layer¶{"class_name": "Dense", "name": "dense_19", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_19", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
÷

Ykernel
Zbias
[	variables
\trainable_variables
]regularization_losses
^	keras_api
+ü&call_and_return_all_conditional_losses
ý__call__"Ð
_tf_keras_layer¶{"class_name": "Dense", "name": "dense_21", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_21", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20]}}
÷

_kernel
`bias
a	variables
btrainable_variables
cregularization_losses
d	keras_api
+þ&call_and_return_all_conditional_losses
ÿ__call__"Ð
_tf_keras_layer¶{"class_name": "Dense", "name": "dense_20", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_20", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
ù

ekernel
fbias
g	variables
htrainable_variables
iregularization_losses
j	keras_api
+&call_and_return_all_conditional_losses
__call__"Ò
_tf_keras_layer¸{"class_name": "Dense", "name": "dense_22", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_22", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
Õ
k	keras_api"Ã
_tf_keras_layer©{"class_name": "TFOpLambda", "name": "tf.concat_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.concat_8", "trainable": true, "dtype": "float32", "function": "concat"}}
ù

lkernel
mbias
n	variables
otrainable_variables
pregularization_losses
q	keras_api
+&call_and_return_all_conditional_losses
__call__"Ò
_tf_keras_layer¸{"class_name": "Dense", "name": "dense_23", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_23", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
Ø
r	keras_api"Æ
_tf_keras_layer¬{"class_name": "TFOpLambda", "name": "tf.nn.relu_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.nn.relu_6", "trainable": true, "dtype": "float32", "function": "nn.relu"}}
ù

skernel
tbias
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
+&call_and_return_all_conditional_losses
__call__"Ò
_tf_keras_layer¸{"class_name": "Dense", "name": "dense_24", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_24", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
ø
y	keras_api"æ
_tf_keras_layerÌ{"class_name": "TFOpLambda", "name": "tf.__operators__.add_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_16", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
Ø
z	keras_api"Æ
_tf_keras_layer¬{"class_name": "TFOpLambda", "name": "tf.nn.relu_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.nn.relu_7", "trainable": true, "dtype": "float32", "function": "nn.relu"}}
ú

{kernel
|bias
}	variables
~trainable_variables
regularization_losses
	keras_api
+&call_and_return_all_conditional_losses
__call__"Ò
_tf_keras_layer¸{"class_name": "Dense", "name": "dense_25", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_25", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
ù
	keras_api"æ
_tf_keras_layerÌ{"class_name": "TFOpLambda", "name": "tf.__operators__.add_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_17", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
Ù
	keras_api"Æ
_tf_keras_layer¬{"class_name": "TFOpLambda", "name": "tf.nn.relu_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.nn.relu_8", "trainable": true, "dtype": "float32", "function": "nn.relu"}}
Ò
	normalize
	variables
trainable_variables
regularization_losses
	keras_api
+&call_and_return_all_conditional_losses
__call__"­
_tf_keras_layer{"class_name": "Normalize", "name": "normalize_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "normalize_2", "trainable": true, "dtype": "float32"}}

kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
+&call_and_return_all_conditional_losses
__call__"Ý
_tf_keras_layerÃ{"class_name": "Dense", "name": "dense_26", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_26", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Constant", "config": {"value": 0}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
È
	iter
beta_1
beta_2

decay
learning_rateLmÁMmÂSmÃTmÄYmÅZmÆ_mÇ`mÈemÉfmÊlmËmmÌsmÍtmÎ{mÏ|mÐ	mÑ	mÒ	mÓ	mÔ	mÕ	mÖ	m×	mØLvÙMvÚSvÛTvÜYvÝZvÞ_vß`vàeváfvâlvãmväsvåtvæ{vç|vè	vé	vê	vë	vì	ví	vî	vï	vð"
	optimizer
 "
trackable_dict_wrapper
ù
0
1
2
3
4
5
L6
M7
S8
T9
Y10
Z11
_12
`13
e14
f15
l16
m17
s18
t19
{20
|21
22
23
24
25
26"
trackable_list_wrapper
Þ
0
1
2
3
4
5
L6
M7
S8
T9
Y10
Z11
_12
`13
e14
f15
l16
m17
s18
t19
{20
|21
22
23"
trackable_list_wrapper
 "
trackable_list_wrapper
Ó
 layer_regularization_losses
layers
	variables
non_trainable_variables
trainable_variables
regularization_losses
layer_metrics
 metrics
ó__call__
ñ_default_save_signature
+ò&call_and_return_all_conditional_losses
'ò"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
é"æ
_tf_keras_input_layerÆ{"class_name": "InputLayer", "name": "input_5", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}}
ì
¡	variables
¢trainable_variables
£regularization_losses
¤	keras_api
+&call_and_return_all_conditional_losses
__call__"×
_tf_keras_layer½{"class_name": "Flatten", "name": "flatten_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ë
¥	keras_api"Ø
_tf_keras_layer¾{"class_name": "TFOpLambda", "name": "tf.clip_by_value_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.clip_by_value_6", "trainable": true, "dtype": "float32", "function": "clip_by_value"}}
ý
¦	keras_api"ê
_tf_keras_layerÐ{"class_name": "TFOpLambda", "name": "tf.compat.v1.floor_div_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.compat.v1.floor_div_4", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}}
µ

embeddings
§	variables
¨trainable_variables
©regularization_losses
ª	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layerõ{"class_name": "Embedding", "name": "embedding_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_14", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}}
µ

embeddings
«	variables
¬trainable_variables
­regularization_losses
®	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layerõ{"class_name": "Embedding", "name": "embedding_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_12", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}}
ë
¯	keras_api"Ø
_tf_keras_layer¾{"class_name": "TFOpLambda", "name": "tf.math.floormod_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.floormod_4", "trainable": true, "dtype": "float32", "function": "math.floormod"}}
ú
°	keras_api"ç
_tf_keras_layerÍ{"class_name": "TFOpLambda", "name": "tf.math.greater_equal_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.greater_equal_6", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}}
ù
±	keras_api"æ
_tf_keras_layerÌ{"class_name": "TFOpLambda", "name": "tf.__operators__.add_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_12", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
´

embeddings
²	variables
³trainable_variables
´regularization_losses
µ	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layerô{"class_name": "Embedding", "name": "embedding_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_13", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}}
Ð
¶	keras_api"½
_tf_keras_layer£{"class_name": "TFOpLambda", "name": "tf.cast_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.cast_6", "trainable": true, "dtype": "float32", "function": "cast"}}
ù
·	keras_api"æ
_tf_keras_layerÌ{"class_name": "TFOpLambda", "name": "tf.__operators__.add_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_13", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
å
¸	keras_api"Ò
_tf_keras_layer¸{"class_name": "TFOpLambda", "name": "tf.expand_dims_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.expand_dims_4", "trainable": true, "dtype": "float32", "function": "expand_dims"}}
ë
¹	keras_api"Ø
_tf_keras_layer¾{"class_name": "TFOpLambda", "name": "tf.math.multiply_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_4", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
ñ
º	keras_api"Þ
_tf_keras_layerÄ{"class_name": "TFOpLambda", "name": "tf.math.reduce_sum_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.reduce_sum_4", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}}
8
0
1
2"
trackable_list_wrapper
8
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 »layer_regularization_losses
¼layers
1	variables
½non_trainable_variables
2trainable_variables
3regularization_losses
¾layer_metrics
¿metrics
õ__call__
+ô&call_and_return_all_conditional_losses
'ô"call_and_return_conditional_losses"
_generic_user_object
é"æ
_tf_keras_input_layerÆ{"class_name": "InputLayer", "name": "input_6", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_6"}}
ì
À	variables
Átrainable_variables
Âregularization_losses
Ã	keras_api
+&call_and_return_all_conditional_losses
__call__"×
_tf_keras_layer½{"class_name": "Flatten", "name": "flatten_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_5", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ë
Ä	keras_api"Ø
_tf_keras_layer¾{"class_name": "TFOpLambda", "name": "tf.clip_by_value_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.clip_by_value_7", "trainable": true, "dtype": "float32", "function": "clip_by_value"}}
ý
Å	keras_api"ê
_tf_keras_layerÐ{"class_name": "TFOpLambda", "name": "tf.compat.v1.floor_div_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.compat.v1.floor_div_5", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}}
µ

embeddings
Æ	variables
Çtrainable_variables
Èregularization_losses
É	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layerõ{"class_name": "Embedding", "name": "embedding_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_17", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3]}}
µ

embeddings
Ê	variables
Ëtrainable_variables
Ìregularization_losses
Í	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layerõ{"class_name": "Embedding", "name": "embedding_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_15", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3]}}
ë
Î	keras_api"Ø
_tf_keras_layer¾{"class_name": "TFOpLambda", "name": "tf.math.floormod_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.floormod_5", "trainable": true, "dtype": "float32", "function": "math.floormod"}}
ú
Ï	keras_api"ç
_tf_keras_layerÍ{"class_name": "TFOpLambda", "name": "tf.math.greater_equal_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.greater_equal_7", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}}
ù
Ð	keras_api"æ
_tf_keras_layerÌ{"class_name": "TFOpLambda", "name": "tf.__operators__.add_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_14", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
´

embeddings
Ñ	variables
Òtrainable_variables
Óregularization_losses
Ô	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layerô{"class_name": "Embedding", "name": "embedding_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_16", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3]}}
Ð
Õ	keras_api"½
_tf_keras_layer£{"class_name": "TFOpLambda", "name": "tf.cast_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.cast_7", "trainable": true, "dtype": "float32", "function": "cast"}}
ù
Ö	keras_api"æ
_tf_keras_layerÌ{"class_name": "TFOpLambda", "name": "tf.__operators__.add_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_15", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
å
×	keras_api"Ò
_tf_keras_layer¸{"class_name": "TFOpLambda", "name": "tf.expand_dims_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.expand_dims_5", "trainable": true, "dtype": "float32", "function": "expand_dims"}}
ë
Ø	keras_api"Ø
_tf_keras_layer¾{"class_name": "TFOpLambda", "name": "tf.math.multiply_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_5", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
ñ
Ù	keras_api"Þ
_tf_keras_layerÄ{"class_name": "TFOpLambda", "name": "tf.math.reduce_sum_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.reduce_sum_5", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}}
8
0
1
2"
trackable_list_wrapper
8
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 Úlayer_regularization_losses
Ûlayers
D	variables
Ünon_trainable_variables
Etrainable_variables
Fregularization_losses
Ýlayer_metrics
Þmetrics
÷__call__
+ö&call_and_return_all_conditional_losses
'ö"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
#:!
2dense_18/kernel
:2dense_18/bias
.
L0
M1"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 ßlayer_regularization_losses
àlayers
N	variables
ánon_trainable_variables
Otrainable_variables
Pregularization_losses
âlayer_metrics
ãmetrics
ù__call__
+ø&call_and_return_all_conditional_losses
'ø"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
#:!
2dense_19/kernel
:2dense_19/bias
.
S0
T1"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 älayer_regularization_losses
ålayers
U	variables
ænon_trainable_variables
Vtrainable_variables
Wregularization_losses
çlayer_metrics
èmetrics
û__call__
+ú&call_and_return_all_conditional_losses
'ú"call_and_return_conditional_losses"
_generic_user_object
": 	2dense_21/kernel
:2dense_21/bias
.
Y0
Z1"
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 élayer_regularization_losses
êlayers
[	variables
ënon_trainable_variables
\trainable_variables
]regularization_losses
ìlayer_metrics
ímetrics
ý__call__
+ü&call_and_return_all_conditional_losses
'ü"call_and_return_conditional_losses"
_generic_user_object
#:!
2dense_20/kernel
:2dense_20/bias
.
_0
`1"
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 îlayer_regularization_losses
ïlayers
a	variables
ðnon_trainable_variables
btrainable_variables
cregularization_losses
ñlayer_metrics
òmetrics
ÿ__call__
+þ&call_and_return_all_conditional_losses
'þ"call_and_return_conditional_losses"
_generic_user_object
#:!
2dense_22/kernel
:2dense_22/bias
.
e0
f1"
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 ólayer_regularization_losses
ôlayers
g	variables
õnon_trainable_variables
htrainable_variables
iregularization_losses
ölayer_metrics
÷metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
#:!
2dense_23/kernel
:2dense_23/bias
.
l0
m1"
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 ølayer_regularization_losses
ùlayers
n	variables
únon_trainable_variables
otrainable_variables
pregularization_losses
ûlayer_metrics
ümetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
#:!
2dense_24/kernel
:2dense_24/bias
.
s0
t1"
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 ýlayer_regularization_losses
þlayers
u	variables
ÿnon_trainable_variables
vtrainable_variables
wregularization_losses
layer_metrics
metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
#:!
2dense_25/kernel
:2dense_25/bias
.
{0
|1"
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 layer_regularization_losses
layers
}	variables
non_trainable_variables
~trainable_variables
regularization_losses
layer_metrics
metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object

state_variables
_broadcast_shape
	mean
variance

count
	keras_api"¶
_tf_keras_layer{"class_name": "Normalization", "name": "normalization_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "normalization_2", "trainable": true, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [-1]}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
8
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
 layer_regularization_losses
layers
	variables
non_trainable_variables
trainable_variables
regularization_losses
layer_metrics
metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
": 	2dense_26/kernel
:2dense_26/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
 layer_regularization_losses
layers
	variables
non_trainable_variables
trainable_variables
regularization_losses
layer_metrics
metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
*:(	42embedding_14/embeddings
*:(	2embedding_12/embeddings
*:(	2embedding_13/embeddings
*:(	42embedding_17/embeddings
*:(	2embedding_15/embeddings
*:(	2embedding_16/embeddings
-:+2 normalize_2/normalization_2/mean
1:/2$normalize_2/normalization_2/variance
):'	 2!normalize_2/normalization_2/count
 "
trackable_list_wrapper
æ
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
21
22
23
24
25"
trackable_list_wrapper
8
0
1
2"
trackable_list_wrapper
 "
trackable_dict_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
 layer_regularization_losses
layers
¡	variables
non_trainable_variables
¢trainable_variables
£regularization_losses
layer_metrics
metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
(
0"
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
 layer_regularization_losses
layers
§	variables
non_trainable_variables
¨trainable_variables
©regularization_losses
layer_metrics
metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
(
0"
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
 layer_regularization_losses
 layers
«	variables
¡non_trainable_variables
¬trainable_variables
­regularization_losses
¢layer_metrics
£metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
(
0"
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
 ¤layer_regularization_losses
¥layers
²	variables
¦non_trainable_variables
³trainable_variables
´regularization_losses
§layer_metrics
¨metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper

"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10
-11
.12
/13
014"
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
¸
 ©layer_regularization_losses
ªlayers
À	variables
«non_trainable_variables
Átrainable_variables
Âregularization_losses
¬layer_metrics
­metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
(
0"
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
 ®layer_regularization_losses
¯layers
Æ	variables
°non_trainable_variables
Çtrainable_variables
Èregularization_losses
±layer_metrics
²metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
(
0"
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
 ³layer_regularization_losses
´layers
Ê	variables
µnon_trainable_variables
Ëtrainable_variables
Ìregularization_losses
¶layer_metrics
·metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
(
0"
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
 ¸layer_regularization_losses
¹layers
Ñ	variables
ºnon_trainable_variables
Òtrainable_variables
Óregularization_losses
»layer_metrics
¼metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper

50
61
72
83
94
:5
;6
<7
=8
>9
?10
@11
A12
B13
C14"
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
F
	mean
variance

count"
trackable_dict_wrapper
 "
trackable_list_wrapper
"
_generic_user_object
 "
trackable_list_wrapper
(
0"
trackable_list_wrapper
8
0
1
2"
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
¿

½total

¾count
¿	variables
À	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
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
:  (2total
:  (2count
0
½0
¾1"
trackable_list_wrapper
.
¿	variables"
_generic_user_object
(:&
2Adam/dense_18/kernel/m
!:2Adam/dense_18/bias/m
(:&
2Adam/dense_19/kernel/m
!:2Adam/dense_19/bias/m
':%	2Adam/dense_21/kernel/m
!:2Adam/dense_21/bias/m
(:&
2Adam/dense_20/kernel/m
!:2Adam/dense_20/bias/m
(:&
2Adam/dense_22/kernel/m
!:2Adam/dense_22/bias/m
(:&
2Adam/dense_23/kernel/m
!:2Adam/dense_23/bias/m
(:&
2Adam/dense_24/kernel/m
!:2Adam/dense_24/bias/m
(:&
2Adam/dense_25/kernel/m
!:2Adam/dense_25/bias/m
':%	2Adam/dense_26/kernel/m
 :2Adam/dense_26/bias/m
/:-	42Adam/embedding_14/embeddings/m
/:-	2Adam/embedding_12/embeddings/m
/:-	2Adam/embedding_13/embeddings/m
/:-	42Adam/embedding_17/embeddings/m
/:-	2Adam/embedding_15/embeddings/m
/:-	2Adam/embedding_16/embeddings/m
(:&
2Adam/dense_18/kernel/v
!:2Adam/dense_18/bias/v
(:&
2Adam/dense_19/kernel/v
!:2Adam/dense_19/bias/v
':%	2Adam/dense_21/kernel/v
!:2Adam/dense_21/bias/v
(:&
2Adam/dense_20/kernel/v
!:2Adam/dense_20/bias/v
(:&
2Adam/dense_22/kernel/v
!:2Adam/dense_22/bias/v
(:&
2Adam/dense_23/kernel/v
!:2Adam/dense_23/bias/v
(:&
2Adam/dense_24/kernel/v
!:2Adam/dense_24/bias/v
(:&
2Adam/dense_25/kernel/v
!:2Adam/dense_25/bias/v
':%	2Adam/dense_26/kernel/v
 :2Adam/dense_26/bias/v
/:-	42Adam/embedding_14/embeddings/v
/:-	2Adam/embedding_12/embeddings/v
/:-	2Adam/embedding_13/embeddings/v
/:-	42Adam/embedding_17/embeddings/v
/:-	2Adam/embedding_15/embeddings/v
/:-	2Adam/embedding_16/embeddings/v
­2ª
$__inference__wrapped_model_200070771
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *q¢n
li
GD
 
cards0ÿÿÿÿÿÿÿÿÿ
 
cards1ÿÿÿÿÿÿÿÿÿ

betsÿÿÿÿÿÿÿÿÿ

2ÿ
M__inference_custom_model_2_layer_call_and_return_conditional_losses_200071680
M__inference_custom_model_2_layer_call_and_return_conditional_losses_200072422
M__inference_custom_model_2_layer_call_and_return_conditional_losses_200071588
M__inference_custom_model_2_layer_call_and_return_conditional_losses_200072252À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
2__inference_custom_model_2_layer_call_fn_200071842
2__inference_custom_model_2_layer_call_fn_200072491
2__inference_custom_model_2_layer_call_fn_200072003
2__inference_custom_model_2_layer_call_fn_200072560À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
æ2ã
F__inference_model_4_layer_call_and_return_conditional_losses_200072602
F__inference_model_4_layer_call_and_return_conditional_losses_200070874
F__inference_model_4_layer_call_and_return_conditional_losses_200070906
F__inference_model_4_layer_call_and_return_conditional_losses_200072644À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ú2÷
+__inference_model_4_layer_call_fn_200072670
+__inference_model_4_layer_call_fn_200070952
+__inference_model_4_layer_call_fn_200072657
+__inference_model_4_layer_call_fn_200070997À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
æ2ã
F__inference_model_5_layer_call_and_return_conditional_losses_200072754
F__inference_model_5_layer_call_and_return_conditional_losses_200072712
F__inference_model_5_layer_call_and_return_conditional_losses_200071100
F__inference_model_5_layer_call_and_return_conditional_losses_200071132À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ú2÷
+__inference_model_5_layer_call_fn_200072780
+__inference_model_5_layer_call_fn_200071178
+__inference_model_5_layer_call_fn_200072767
+__inference_model_5_layer_call_fn_200071223À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ñ2î
G__inference_dense_18_layer_call_and_return_conditional_losses_200072791¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_dense_18_layer_call_fn_200072800¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_dense_19_layer_call_and_return_conditional_losses_200072811¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_dense_19_layer_call_fn_200072820¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_dense_21_layer_call_and_return_conditional_losses_200072830¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_dense_21_layer_call_fn_200072839¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_dense_20_layer_call_and_return_conditional_losses_200072850¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_dense_20_layer_call_fn_200072859¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_dense_22_layer_call_and_return_conditional_losses_200072869¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_dense_22_layer_call_fn_200072878¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_dense_23_layer_call_and_return_conditional_losses_200072888¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_dense_23_layer_call_fn_200072897¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_dense_24_layer_call_and_return_conditional_losses_200072907¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_dense_24_layer_call_fn_200072916¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_dense_25_layer_call_and_return_conditional_losses_200072926¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_dense_25_layer_call_fn_200072935¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
J__inference_normalize_2_layer_call_and_return_conditional_losses_200072952
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
/__inference_normalize_2_layer_call_fn_200072961
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_dense_26_layer_call_and_return_conditional_losses_200072971¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_dense_26_layer_call_fn_200072980¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÙBÖ
'__inference_signature_wrapper_200072082betscards0cards1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_flatten_4_layer_call_and_return_conditional_losses_200072986¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
×2Ô
-__inference_flatten_4_layer_call_fn_200072991¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
õ2ò
K__inference_embedding_14_layer_call_and_return_conditional_losses_200073001¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ú2×
0__inference_embedding_14_layer_call_fn_200073008¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
õ2ò
K__inference_embedding_12_layer_call_and_return_conditional_losses_200073018¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ú2×
0__inference_embedding_12_layer_call_fn_200073025¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
õ2ò
K__inference_embedding_13_layer_call_and_return_conditional_losses_200073035¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ú2×
0__inference_embedding_13_layer_call_fn_200073042¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_flatten_5_layer_call_and_return_conditional_losses_200073048¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
×2Ô
-__inference_flatten_5_layer_call_fn_200073053¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
õ2ò
K__inference_embedding_17_layer_call_and_return_conditional_losses_200073063¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ú2×
0__inference_embedding_17_layer_call_fn_200073070¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
õ2ò
K__inference_embedding_15_layer_call_and_return_conditional_losses_200073080¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ú2×
0__inference_embedding_15_layer_call_fn_200073087¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
õ2ò
K__inference_embedding_16_layer_call_and_return_conditional_losses_200073097¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ú2×
0__inference_embedding_16_layer_call_fn_200073104¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
	J
Const
J	
Const_1
J	
Const_2
J	
Const_3
J	
Const_4
$__inference__wrapped_model_200070771â. ¡LMYZST_`eflmst{|{¢x
q¢n
li
GD
 
cards0ÿÿÿÿÿÿÿÿÿ
 
cards1ÿÿÿÿÿÿÿÿÿ

betsÿÿÿÿÿÿÿÿÿ

ª "3ª0
.
dense_26"
dense_26ÿÿÿÿÿÿÿÿÿ°
M__inference_custom_model_2_layer_call_and_return_conditional_losses_200071588Þ. ¡LMYZST_`eflmst{|¢
y¢v
li
GD
 
cards0ÿÿÿÿÿÿÿÿÿ
 
cards1ÿÿÿÿÿÿÿÿÿ

betsÿÿÿÿÿÿÿÿÿ

p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 °
M__inference_custom_model_2_layer_call_and_return_conditional_losses_200071680Þ. ¡LMYZST_`eflmst{|¢
y¢v
li
GD
 
cards0ÿÿÿÿÿÿÿÿÿ
 
cards1ÿÿÿÿÿÿÿÿÿ

betsÿÿÿÿÿÿÿÿÿ

p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¾
M__inference_custom_model_2_layer_call_and_return_conditional_losses_200072252ì. ¡LMYZST_`eflmst{|¢
¢
xu
OL
$!

inputs/0/0ÿÿÿÿÿÿÿÿÿ
$!

inputs/0/1ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ

p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¾
M__inference_custom_model_2_layer_call_and_return_conditional_losses_200072422ì. ¡LMYZST_`eflmst{|¢
¢
xu
OL
$!

inputs/0/0ÿÿÿÿÿÿÿÿÿ
$!

inputs/0/1ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ

p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
2__inference_custom_model_2_layer_call_fn_200071842Ñ. ¡LMYZST_`eflmst{|¢
y¢v
li
GD
 
cards0ÿÿÿÿÿÿÿÿÿ
 
cards1ÿÿÿÿÿÿÿÿÿ

betsÿÿÿÿÿÿÿÿÿ

p

 
ª "ÿÿÿÿÿÿÿÿÿ
2__inference_custom_model_2_layer_call_fn_200072003Ñ. ¡LMYZST_`eflmst{|¢
y¢v
li
GD
 
cards0ÿÿÿÿÿÿÿÿÿ
 
cards1ÿÿÿÿÿÿÿÿÿ

betsÿÿÿÿÿÿÿÿÿ

p 

 
ª "ÿÿÿÿÿÿÿÿÿ
2__inference_custom_model_2_layer_call_fn_200072491ß. ¡LMYZST_`eflmst{|¢
¢
xu
OL
$!

inputs/0/0ÿÿÿÿÿÿÿÿÿ
$!

inputs/0/1ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ

p

 
ª "ÿÿÿÿÿÿÿÿÿ
2__inference_custom_model_2_layer_call_fn_200072560ß. ¡LMYZST_`eflmst{|¢
¢
xu
OL
$!

inputs/0/0ÿÿÿÿÿÿÿÿÿ
$!

inputs/0/1ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ

p 

 
ª "ÿÿÿÿÿÿÿÿÿ©
G__inference_dense_18_layer_call_and_return_conditional_losses_200072791^LM0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_dense_18_layer_call_fn_200072800QLM0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ©
G__inference_dense_19_layer_call_and_return_conditional_losses_200072811^ST0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_dense_19_layer_call_fn_200072820QST0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ©
G__inference_dense_20_layer_call_and_return_conditional_losses_200072850^_`0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_dense_20_layer_call_fn_200072859Q_`0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¨
G__inference_dense_21_layer_call_and_return_conditional_losses_200072830]YZ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_dense_21_layer_call_fn_200072839PYZ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ©
G__inference_dense_22_layer_call_and_return_conditional_losses_200072869^ef0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_dense_22_layer_call_fn_200072878Qef0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ©
G__inference_dense_23_layer_call_and_return_conditional_losses_200072888^lm0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_dense_23_layer_call_fn_200072897Qlm0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ©
G__inference_dense_24_layer_call_and_return_conditional_losses_200072907^st0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_dense_24_layer_call_fn_200072916Qst0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ©
G__inference_dense_25_layer_call_and_return_conditional_losses_200072926^{|0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_dense_25_layer_call_fn_200072935Q{|0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿª
G__inference_dense_26_layer_call_and_return_conditional_losses_200072971_0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_dense_26_layer_call_fn_200072980R0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ°
K__inference_embedding_12_layer_call_and_return_conditional_losses_200073018a/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 
0__inference_embedding_12_layer_call_fn_200073025T/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ°
K__inference_embedding_13_layer_call_and_return_conditional_losses_200073035a/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 
0__inference_embedding_13_layer_call_fn_200073042T/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ°
K__inference_embedding_14_layer_call_and_return_conditional_losses_200073001a/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 
0__inference_embedding_14_layer_call_fn_200073008T/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ°
K__inference_embedding_15_layer_call_and_return_conditional_losses_200073080a/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 
0__inference_embedding_15_layer_call_fn_200073087T/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ°
K__inference_embedding_16_layer_call_and_return_conditional_losses_200073097a/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 
0__inference_embedding_16_layer_call_fn_200073104T/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ°
K__inference_embedding_17_layer_call_and_return_conditional_losses_200073063a/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 
0__inference_embedding_17_layer_call_fn_200073070T/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¤
H__inference_flatten_4_layer_call_and_return_conditional_losses_200072986X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 |
-__inference_flatten_4_layer_call_fn_200072991K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¤
H__inference_flatten_5_layer_call_and_return_conditional_losses_200073048X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 |
-__inference_flatten_5_layer_call_fn_200073053K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¶
F__inference_model_4_layer_call_and_return_conditional_losses_200070874l8¢5
.¢+
!
input_5ÿÿÿÿÿÿÿÿÿ
p

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ¶
F__inference_model_4_layer_call_and_return_conditional_losses_200070906l8¢5
.¢+
!
input_5ÿÿÿÿÿÿÿÿÿ
p 

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 µ
F__inference_model_4_layer_call_and_return_conditional_losses_200072602k7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 µ
F__inference_model_4_layer_call_and_return_conditional_losses_200072644k7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_model_4_layer_call_fn_200070952_8¢5
.¢+
!
input_5ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_model_4_layer_call_fn_200070997_8¢5
.¢+
!
input_5ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_model_4_layer_call_fn_200072657^7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_model_4_layer_call_fn_200072670^7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ¶
F__inference_model_5_layer_call_and_return_conditional_losses_200071100l8¢5
.¢+
!
input_6ÿÿÿÿÿÿÿÿÿ
p

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ¶
F__inference_model_5_layer_call_and_return_conditional_losses_200071132l8¢5
.¢+
!
input_6ÿÿÿÿÿÿÿÿÿ
p 

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 µ
F__inference_model_5_layer_call_and_return_conditional_losses_200072712k7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 µ
F__inference_model_5_layer_call_and_return_conditional_losses_200072754k7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_model_5_layer_call_fn_200071178_8¢5
.¢+
!
input_6ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_model_5_layer_call_fn_200071223_8¢5
.¢+
!
input_6ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_model_5_layer_call_fn_200072767^7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_model_5_layer_call_fn_200072780^7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ©
J__inference_normalize_2_layer_call_and_return_conditional_losses_200072952[+¢(
!¢

xÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
/__inference_normalize_2_layer_call_fn_200072961N+¢(
!¢

xÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¤
'__inference_signature_wrapper_200072082ø. ¡LMYZST_`eflmst{|¢
¢ 
ª
&
bets
betsÿÿÿÿÿÿÿÿÿ

*
cards0 
cards0ÿÿÿÿÿÿÿÿÿ
*
cards1 
cards1ÿÿÿÿÿÿÿÿÿ"3ª0
.
dense_26"
dense_26ÿÿÿÿÿÿÿÿÿ
??!
??
B
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
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
delete_old_dirsbool(?
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
2	?
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
dtypetype?
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
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
?
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
executor_typestring ?
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
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8??
|
dense_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_18/kernel
u
#dense_18/kernel/Read/ReadVariableOpReadVariableOpdense_18/kernel* 
_output_shapes
:
??*
dtype0
s
dense_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_18/bias
l
!dense_18/bias/Read/ReadVariableOpReadVariableOpdense_18/bias*
_output_shapes	
:?*
dtype0
|
dense_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_19/kernel
u
#dense_19/kernel/Read/ReadVariableOpReadVariableOpdense_19/kernel* 
_output_shapes
:
??*
dtype0
s
dense_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_19/bias
l
!dense_19/bias/Read/ReadVariableOpReadVariableOpdense_19/bias*
_output_shapes	
:?*
dtype0
{
dense_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_namedense_21/kernel
t
#dense_21/kernel/Read/ReadVariableOpReadVariableOpdense_21/kernel*
_output_shapes
:	?*
dtype0
s
dense_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_21/bias
l
!dense_21/bias/Read/ReadVariableOpReadVariableOpdense_21/bias*
_output_shapes	
:?*
dtype0
|
dense_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_20/kernel
u
#dense_20/kernel/Read/ReadVariableOpReadVariableOpdense_20/kernel* 
_output_shapes
:
??*
dtype0
s
dense_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_20/bias
l
!dense_20/bias/Read/ReadVariableOpReadVariableOpdense_20/bias*
_output_shapes	
:?*
dtype0
|
dense_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_22/kernel
u
#dense_22/kernel/Read/ReadVariableOpReadVariableOpdense_22/kernel* 
_output_shapes
:
??*
dtype0
s
dense_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_22/bias
l
!dense_22/bias/Read/ReadVariableOpReadVariableOpdense_22/bias*
_output_shapes	
:?*
dtype0
|
dense_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_23/kernel
u
#dense_23/kernel/Read/ReadVariableOpReadVariableOpdense_23/kernel* 
_output_shapes
:
??*
dtype0
s
dense_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_23/bias
l
!dense_23/bias/Read/ReadVariableOpReadVariableOpdense_23/bias*
_output_shapes	
:?*
dtype0
|
dense_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_24/kernel
u
#dense_24/kernel/Read/ReadVariableOpReadVariableOpdense_24/kernel* 
_output_shapes
:
??*
dtype0
s
dense_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_24/bias
l
!dense_24/bias/Read/ReadVariableOpReadVariableOpdense_24/bias*
_output_shapes	
:?*
dtype0
|
dense_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_25/kernel
u
#dense_25/kernel/Read/ReadVariableOpReadVariableOpdense_25/kernel* 
_output_shapes
:
??*
dtype0
s
dense_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_25/bias
l
!dense_25/bias/Read/ReadVariableOpReadVariableOpdense_25/bias*
_output_shapes	
:?*
dtype0
{
dense_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_namedense_26/kernel
t
#dense_26/kernel/Read/ReadVariableOpReadVariableOpdense_26/kernel*
_output_shapes
:	?*
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
?
embedding_14/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	4?*(
shared_nameembedding_14/embeddings
?
+embedding_14/embeddings/Read/ReadVariableOpReadVariableOpembedding_14/embeddings*
_output_shapes
:	4?*
dtype0
?
embedding_12/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*(
shared_nameembedding_12/embeddings
?
+embedding_12/embeddings/Read/ReadVariableOpReadVariableOpembedding_12/embeddings*
_output_shapes
:	?*
dtype0
?
embedding_13/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*(
shared_nameembedding_13/embeddings
?
+embedding_13/embeddings/Read/ReadVariableOpReadVariableOpembedding_13/embeddings*
_output_shapes
:	?*
dtype0
?
embedding_17/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	4?*(
shared_nameembedding_17/embeddings
?
+embedding_17/embeddings/Read/ReadVariableOpReadVariableOpembedding_17/embeddings*
_output_shapes
:	4?*
dtype0
?
embedding_15/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*(
shared_nameembedding_15/embeddings
?
+embedding_15/embeddings/Read/ReadVariableOpReadVariableOpembedding_15/embeddings*
_output_shapes
:	?*
dtype0
?
embedding_16/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*(
shared_nameembedding_16/embeddings
?
+embedding_16/embeddings/Read/ReadVariableOpReadVariableOpembedding_16/embeddings*
_output_shapes
:	?*
dtype0
?
 normalize_2/normalization_2/meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" normalize_2/normalization_2/mean
?
4normalize_2/normalization_2/mean/Read/ReadVariableOpReadVariableOp normalize_2/normalization_2/mean*
_output_shapes	
:?*
dtype0
?
$normalize_2/normalization_2/varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*5
shared_name&$normalize_2/normalization_2/variance
?
8normalize_2/normalization_2/variance/Read/ReadVariableOpReadVariableOp$normalize_2/normalization_2/variance*
_output_shapes	
:?*
dtype0
?
!normalize_2/normalization_2/countVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *2
shared_name#!normalize_2/normalization_2/count
?
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
?
Adam/dense_18/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_18/kernel/m
?
*Adam/dense_18/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_18/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_18/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_18/bias/m
z
(Adam/dense_18/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_18/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_19/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_19/kernel/m
?
*Adam/dense_19/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_19/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_19/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_19/bias/m
z
(Adam/dense_19/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_19/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_21/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_21/kernel/m
?
*Adam/dense_21/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_21/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam/dense_21/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_21/bias/m
z
(Adam/dense_21/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_21/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_20/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_20/kernel/m
?
*Adam/dense_20/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_20/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_20/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_20/bias/m
z
(Adam/dense_20/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_20/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_22/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_22/kernel/m
?
*Adam/dense_22/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_22/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_22/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_22/bias/m
z
(Adam/dense_22/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_22/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_23/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_23/kernel/m
?
*Adam/dense_23/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_23/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_23/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_23/bias/m
z
(Adam/dense_23/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_23/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_24/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_24/kernel/m
?
*Adam/dense_24/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_24/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_24/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_24/bias/m
z
(Adam/dense_24/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_24/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_25/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_25/kernel/m
?
*Adam/dense_25/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_25/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_25/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_25/bias/m
z
(Adam/dense_25/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_25/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_26/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_26/kernel/m
?
*Adam/dense_26/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_26/kernel/m*
_output_shapes
:	?*
dtype0
?
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
?
Adam/embedding_14/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	4?*/
shared_name Adam/embedding_14/embeddings/m
?
2Adam/embedding_14/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_14/embeddings/m*
_output_shapes
:	4?*
dtype0
?
Adam/embedding_12/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name Adam/embedding_12/embeddings/m
?
2Adam/embedding_12/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_12/embeddings/m*
_output_shapes
:	?*
dtype0
?
Adam/embedding_13/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name Adam/embedding_13/embeddings/m
?
2Adam/embedding_13/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_13/embeddings/m*
_output_shapes
:	?*
dtype0
?
Adam/embedding_17/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	4?*/
shared_name Adam/embedding_17/embeddings/m
?
2Adam/embedding_17/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_17/embeddings/m*
_output_shapes
:	4?*
dtype0
?
Adam/embedding_15/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name Adam/embedding_15/embeddings/m
?
2Adam/embedding_15/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_15/embeddings/m*
_output_shapes
:	?*
dtype0
?
Adam/embedding_16/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name Adam/embedding_16/embeddings/m
?
2Adam/embedding_16/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_16/embeddings/m*
_output_shapes
:	?*
dtype0
?
Adam/dense_18/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_18/kernel/v
?
*Adam/dense_18/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_18/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_18/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_18/bias/v
z
(Adam/dense_18/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_18/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_19/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_19/kernel/v
?
*Adam/dense_19/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_19/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_19/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_19/bias/v
z
(Adam/dense_19/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_19/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_21/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_21/kernel/v
?
*Adam/dense_21/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_21/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/dense_21/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_21/bias/v
z
(Adam/dense_21/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_21/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_20/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_20/kernel/v
?
*Adam/dense_20/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_20/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_20/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_20/bias/v
z
(Adam/dense_20/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_20/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_22/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_22/kernel/v
?
*Adam/dense_22/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_22/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_22/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_22/bias/v
z
(Adam/dense_22/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_22/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_23/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_23/kernel/v
?
*Adam/dense_23/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_23/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_23/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_23/bias/v
z
(Adam/dense_23/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_23/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_24/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_24/kernel/v
?
*Adam/dense_24/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_24/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_24/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_24/bias/v
z
(Adam/dense_24/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_24/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_25/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_25/kernel/v
?
*Adam/dense_25/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_25/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_25/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_25/bias/v
z
(Adam/dense_25/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_25/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_26/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_26/kernel/v
?
*Adam/dense_26/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_26/kernel/v*
_output_shapes
:	?*
dtype0
?
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
?
Adam/embedding_14/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	4?*/
shared_name Adam/embedding_14/embeddings/v
?
2Adam/embedding_14/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_14/embeddings/v*
_output_shapes
:	4?*
dtype0
?
Adam/embedding_12/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name Adam/embedding_12/embeddings/v
?
2Adam/embedding_12/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_12/embeddings/v*
_output_shapes
:	?*
dtype0
?
Adam/embedding_13/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name Adam/embedding_13/embeddings/v
?
2Adam/embedding_13/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_13/embeddings/v*
_output_shapes
:	?*
dtype0
?
Adam/embedding_17/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	4?*/
shared_name Adam/embedding_17/embeddings/v
?
2Adam/embedding_17/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_17/embeddings/v*
_output_shapes
:	4?*
dtype0
?
Adam/embedding_15/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name Adam/embedding_15/embeddings/v
?
2Adam/embedding_15/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_15/embeddings/v*
_output_shapes
:	?*
dtype0
?
Adam/embedding_16/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name Adam/embedding_16/embeddings/v
?
2Adam/embedding_16/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_16/embeddings/v*
_output_shapes
:	?*
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
??
Const_5Const"/device:CPU:0*
_output_shapes
: *
dtype0*ݘ
valueҘBΘ BƘ
?
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
trainable_variables
	variables
regularization_losses
 	keras_api
!
signatures
 
 
 
?
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
1trainable_variables
2	variables
3regularization_losses
4	keras_api
?
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
Dtrainable_variables
E	variables
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
Ntrainable_variables
O	variables
Pregularization_losses
Q	keras_api

R	keras_api
h

Skernel
Tbias
Utrainable_variables
V	variables
Wregularization_losses
X	keras_api
h

Ykernel
Zbias
[trainable_variables
\	variables
]regularization_losses
^	keras_api
h

_kernel
`bias
atrainable_variables
b	variables
cregularization_losses
d	keras_api
h

ekernel
fbias
gtrainable_variables
h	variables
iregularization_losses
j	keras_api

k	keras_api
h

lkernel
mbias
ntrainable_variables
o	variables
pregularization_losses
q	keras_api

r	keras_api
h

skernel
tbias
utrainable_variables
v	variables
wregularization_losses
x	keras_api

y	keras_api

z	keras_api
i

{kernel
|bias
}trainable_variables
~	variables
regularization_losses
?	keras_api

?	keras_api

?	keras_api
f
?	normalize
?trainable_variables
?	variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
	?iter
?beta_1
?beta_2

?decay
?learning_rateLm?Mm?Sm?Tm?Ym?Zm?_m?`m?em?fm?lm?mm?sm?tm?{m?|m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?Lv?Mv?Sv?Tv?Yv?Zv?_v?`v?ev?fv?lv?mv?sv?tv?{v?|v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?
 
?
?0
?1
?2
?3
?4
?5
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
?22
?23
?
?0
?1
?2
?3
?4
?5
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
?22
?23
?24
?25
?26
 
?
?layers
trainable_variables
 ?layer_regularization_losses
	variables
regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
 
 
V
?trainable_variables
?	variables
?regularization_losses
?	keras_api

?	keras_api

?	keras_api
g
?
embeddings
?trainable_variables
?	variables
?regularization_losses
?	keras_api
g
?
embeddings
?trainable_variables
?	variables
?regularization_losses
?	keras_api

?	keras_api

?	keras_api

?	keras_api
g
?
embeddings
?trainable_variables
?	variables
?regularization_losses
?	keras_api

?	keras_api

?	keras_api

?	keras_api

?	keras_api

?	keras_api

?0
?1
?2

?0
?1
?2
 
?
?layers
1trainable_variables
 ?layer_regularization_losses
2	variables
3regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
 
V
?trainable_variables
?	variables
?regularization_losses
?	keras_api

?	keras_api

?	keras_api
g
?
embeddings
?trainable_variables
?	variables
?regularization_losses
?	keras_api
g
?
embeddings
?trainable_variables
?	variables
?regularization_losses
?	keras_api

?	keras_api

?	keras_api

?	keras_api
g
?
embeddings
?trainable_variables
?	variables
?regularization_losses
?	keras_api

?	keras_api

?	keras_api

?	keras_api

?	keras_api

?	keras_api

?0
?1
?2

?0
?1
?2
 
?
?layers
Dtrainable_variables
 ?layer_regularization_losses
E	variables
Fregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
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
?
?layers
Ntrainable_variables
 ?layer_regularization_losses
O	variables
Pregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
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
?
?layers
Utrainable_variables
 ?layer_regularization_losses
V	variables
Wregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
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
?
?layers
[trainable_variables
 ?layer_regularization_losses
\	variables
]regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
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
?
?layers
atrainable_variables
 ?layer_regularization_losses
b	variables
cregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
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
?
?layers
gtrainable_variables
 ?layer_regularization_losses
h	variables
iregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
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
?
?layers
ntrainable_variables
 ?layer_regularization_losses
o	variables
pregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
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
?
?layers
utrainable_variables
 ?layer_regularization_losses
v	variables
wregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
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
?
?layers
}trainable_variables
 ?layer_regularization_losses
~	variables
regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
 
 
c
?state_variables
?_broadcast_shape
	?mean
?variance

?count
?	keras_api
 

?0
?1
?2
 
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
\Z
VARIABLE_VALUEdense_26/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_26/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
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
][
VARIABLE_VALUEembedding_14/embeddings0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEembedding_12/embeddings0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEembedding_13/embeddings0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEembedding_17/embeddings0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEembedding_15/embeddings0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEembedding_16/embeddings0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE normalize_2/normalization_2/mean'variables/22/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE$normalize_2/normalization_2/variance'variables/23/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!normalize_2/normalization_2/count'variables/24/.ATTRIBUTES/VARIABLE_VALUE
?
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
 
 

?0

?0
?1
?2
 
 
 
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
 
 

?0

?0
 
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables

?0

?0
 
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
 
 
 

?0

?0
 
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
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
 
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
 
 

?0

?0
 
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables

?0

?0
 
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
 
 
 

?0

?0
 
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
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
 
&
	?mean
?variance

?count
 
 

?0
 
 
 

?0
?1
?2
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
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
?0
?1

?	variables
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
?~
VARIABLE_VALUEAdam/embedding_14/embeddings/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/embedding_12/embeddings/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/embedding_13/embeddings/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/embedding_17/embeddings/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/embedding_15/embeddings/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/embedding_16/embeddings/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
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
?~
VARIABLE_VALUEAdam/embedding_14/embeddings/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/embedding_12/embeddings/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/embedding_13/embeddings/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/embedding_17/embeddings/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/embedding_15/embeddings/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/embedding_16/embeddings/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
w
serving_default_betsPlaceholder*'
_output_shapes
:?????????
*
dtype0*
shape:?????????

y
serving_default_cards0Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
y
serving_default_cards1Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_betsserving_default_cards0serving_default_cards1ConstConst_1embedding_14/embeddingsembedding_12/embeddingsembedding_13/embeddingsConst_2embedding_17/embeddingsembedding_15/embeddingsembedding_16/embeddingsConst_3Const_4dense_18/kerneldense_18/biasdense_21/kerneldense_21/biasdense_19/kerneldense_19/biasdense_20/kerneldense_20/biasdense_22/kerneldense_22/biasdense_23/kerneldense_23/biasdense_24/kerneldense_24/biasdense_25/kerneldense_25/bias normalize_2/normalization_2/mean$normalize_2/normalization_2/variancedense_26/kerneldense_26/bias*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
 !*2
config_proto" 

CPU

GPU2 *0J 8? */
f*R(
&__inference_signature_wrapper_10264690
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
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
 *2
config_proto" 

CPU

GPU2 *0J 8? **
f%R#
!__inference__traced_save_10265988
?
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
 *2
config_proto" 

CPU

GPU2 *0J 8? *-
f(R&
$__inference__traced_restore_10266244??
?
?
1__inference_custom_model_2_layer_call_fn_10264450

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
identity??StatefulPartitionedCall?
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
:?????????*<
_read_only_resource_inputs
	
 !*2
config_proto" 

CPU

GPU2 *0J 8? *U
fPRN
L__inference_custom_model_2_layer_call_and_return_conditional_losses_102643852
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
: : :::: :::: : ::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_namecards0:OK
'
_output_shapes
:?????????
 
_user_specified_namecards1:MI
'
_output_shapes
:?????????


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
??
?
L__inference_custom_model_2_layer_call_and_return_conditional_losses_10264860

inputs_0_0

inputs_0_1
inputs_1*
&tf_math_greater_equal_8_greaterequal_y2
.model_4_tf_math_greater_equal_6_greaterequal_y2
.model_4_embedding_14_embedding_lookup_102647102
.model_4_embedding_12_embedding_lookup_102647162
.model_4_embedding_13_embedding_lookup_102647242
.model_5_tf_math_greater_equal_7_greaterequal_y2
.model_5_embedding_17_embedding_lookup_102647482
.model_5_embedding_15_embedding_lookup_102647542
.model_5_embedding_16_embedding_lookup_10264762.
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
identity??dense_18/BiasAdd/ReadVariableOp?dense_18/MatMul/ReadVariableOp?dense_19/BiasAdd/ReadVariableOp?dense_19/MatMul/ReadVariableOp?dense_20/BiasAdd/ReadVariableOp?dense_20/MatMul/ReadVariableOp?dense_21/BiasAdd/ReadVariableOp?dense_21/MatMul/ReadVariableOp?dense_22/BiasAdd/ReadVariableOp?dense_22/MatMul/ReadVariableOp?dense_23/BiasAdd/ReadVariableOp?dense_23/MatMul/ReadVariableOp?dense_24/BiasAdd/ReadVariableOp?dense_24/MatMul/ReadVariableOp?dense_25/BiasAdd/ReadVariableOp?dense_25/MatMul/ReadVariableOp?dense_26/BiasAdd/ReadVariableOp?dense_26/MatMul/ReadVariableOp?%model_4/embedding_12/embedding_lookup?%model_4/embedding_13/embedding_lookup?%model_4/embedding_14/embedding_lookup?%model_5/embedding_15/embedding_lookup?%model_5/embedding_16/embedding_lookup?%model_5/embedding_17/embedding_lookup?2normalize_2/normalization_2/Reshape/ReadVariableOp?4normalize_2/normalization_2/Reshape_1/ReadVariableOp?
$tf.math.greater_equal_8/GreaterEqualGreaterEqualinputs_1&tf_math_greater_equal_8_greaterequal_y*
T0*'
_output_shapes
:?????????
2&
$tf.math.greater_equal_8/GreaterEqual?
model_4/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
model_4/flatten_4/Const?
model_4/flatten_4/ReshapeReshape
inputs_0_0 model_4/flatten_4/Const:output:0*
T0*'
_output_shapes
:?????????2
model_4/flatten_4/Reshape?
2model_4/tf.clip_by_value_6/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI24
2model_4/tf.clip_by_value_6/clip_by_value/Minimum/y?
0model_4/tf.clip_by_value_6/clip_by_value/MinimumMinimum"model_4/flatten_4/Reshape:output:0;model_4/tf.clip_by_value_6/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????22
0model_4/tf.clip_by_value_6/clip_by_value/Minimum?
*model_4/tf.clip_by_value_6/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*model_4/tf.clip_by_value_6/clip_by_value/y?
(model_4/tf.clip_by_value_6/clip_by_valueMaximum4model_4/tf.clip_by_value_6/clip_by_value/Minimum:z:03model_4/tf.clip_by_value_6/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2*
(model_4/tf.clip_by_value_6/clip_by_value?
+model_4/tf.compat.v1.floor_div_4/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2-
+model_4/tf.compat.v1.floor_div_4/FloorDiv/y?
)model_4/tf.compat.v1.floor_div_4/FloorDivFloorDiv,model_4/tf.clip_by_value_6/clip_by_value:z:04model_4/tf.compat.v1.floor_div_4/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2+
)model_4/tf.compat.v1.floor_div_4/FloorDiv?
,model_4/tf.math.greater_equal_6/GreaterEqualGreaterEqual"model_4/flatten_4/Reshape:output:0.model_4_tf_math_greater_equal_6_greaterequal_y*
T0*'
_output_shapes
:?????????2.
,model_4/tf.math.greater_equal_6/GreaterEqual?
%model_4/tf.math.floormod_4/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2'
%model_4/tf.math.floormod_4/FloorMod/y?
#model_4/tf.math.floormod_4/FloorModFloorMod,model_4/tf.clip_by_value_6/clip_by_value:z:0.model_4/tf.math.floormod_4/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2%
#model_4/tf.math.floormod_4/FloorMod?
model_4/embedding_14/CastCast,model_4/tf.clip_by_value_6/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_4/embedding_14/Cast?
%model_4/embedding_14/embedding_lookupResourceGather.model_4_embedding_14_embedding_lookup_10264710model_4/embedding_14/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*A
_class7
53loc:@model_4/embedding_14/embedding_lookup/10264710*,
_output_shapes
:??????????*
dtype02'
%model_4/embedding_14/embedding_lookup?
.model_4/embedding_14/embedding_lookup/IdentityIdentity.model_4/embedding_14/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@model_4/embedding_14/embedding_lookup/10264710*,
_output_shapes
:??????????20
.model_4/embedding_14/embedding_lookup/Identity?
0model_4/embedding_14/embedding_lookup/Identity_1Identity7model_4/embedding_14/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????22
0model_4/embedding_14/embedding_lookup/Identity_1?
model_4/embedding_12/CastCast-model_4/tf.compat.v1.floor_div_4/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_4/embedding_12/Cast?
%model_4/embedding_12/embedding_lookupResourceGather.model_4_embedding_12_embedding_lookup_10264716model_4/embedding_12/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*A
_class7
53loc:@model_4/embedding_12/embedding_lookup/10264716*,
_output_shapes
:??????????*
dtype02'
%model_4/embedding_12/embedding_lookup?
.model_4/embedding_12/embedding_lookup/IdentityIdentity.model_4/embedding_12/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@model_4/embedding_12/embedding_lookup/10264716*,
_output_shapes
:??????????20
.model_4/embedding_12/embedding_lookup/Identity?
0model_4/embedding_12/embedding_lookup/Identity_1Identity7model_4/embedding_12/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????22
0model_4/embedding_12/embedding_lookup/Identity_1?
model_4/tf.cast_6/CastCast0model_4/tf.math.greater_equal_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
model_4/tf.cast_6/Cast?
%model_4/tf.__operators__.add_12/AddV2AddV29model_4/embedding_14/embedding_lookup/Identity_1:output:09model_4/embedding_12/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2'
%model_4/tf.__operators__.add_12/AddV2?
model_4/embedding_13/CastCast'model_4/tf.math.floormod_4/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_4/embedding_13/Cast?
%model_4/embedding_13/embedding_lookupResourceGather.model_4_embedding_13_embedding_lookup_10264724model_4/embedding_13/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*A
_class7
53loc:@model_4/embedding_13/embedding_lookup/10264724*,
_output_shapes
:??????????*
dtype02'
%model_4/embedding_13/embedding_lookup?
.model_4/embedding_13/embedding_lookup/IdentityIdentity.model_4/embedding_13/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@model_4/embedding_13/embedding_lookup/10264724*,
_output_shapes
:??????????20
.model_4/embedding_13/embedding_lookup/Identity?
0model_4/embedding_13/embedding_lookup/Identity_1Identity7model_4/embedding_13/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????22
0model_4/embedding_13/embedding_lookup/Identity_1?
%model_4/tf.__operators__.add_13/AddV2AddV2)model_4/tf.__operators__.add_12/AddV2:z:09model_4/embedding_13/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2'
%model_4/tf.__operators__.add_13/AddV2?
'model_4/tf.expand_dims_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'model_4/tf.expand_dims_4/ExpandDims/dim?
#model_4/tf.expand_dims_4/ExpandDims
ExpandDimsmodel_4/tf.cast_6/Cast:y:00model_4/tf.expand_dims_4/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2%
#model_4/tf.expand_dims_4/ExpandDims?
model_4/tf.math.multiply_4/MulMul)model_4/tf.__operators__.add_13/AddV2:z:0,model_4/tf.expand_dims_4/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2 
model_4/tf.math.multiply_4/Mul?
2model_4/tf.math.reduce_sum_4/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :24
2model_4/tf.math.reduce_sum_4/Sum/reduction_indices?
 model_4/tf.math.reduce_sum_4/SumSum"model_4/tf.math.multiply_4/Mul:z:0;model_4/tf.math.reduce_sum_4/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2"
 model_4/tf.math.reduce_sum_4/Sum?
model_5/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
model_5/flatten_5/Const?
model_5/flatten_5/ReshapeReshape
inputs_0_1 model_5/flatten_5/Const:output:0*
T0*'
_output_shapes
:?????????2
model_5/flatten_5/Reshape?
2model_5/tf.clip_by_value_7/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI24
2model_5/tf.clip_by_value_7/clip_by_value/Minimum/y?
0model_5/tf.clip_by_value_7/clip_by_value/MinimumMinimum"model_5/flatten_5/Reshape:output:0;model_5/tf.clip_by_value_7/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????22
0model_5/tf.clip_by_value_7/clip_by_value/Minimum?
*model_5/tf.clip_by_value_7/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*model_5/tf.clip_by_value_7/clip_by_value/y?
(model_5/tf.clip_by_value_7/clip_by_valueMaximum4model_5/tf.clip_by_value_7/clip_by_value/Minimum:z:03model_5/tf.clip_by_value_7/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2*
(model_5/tf.clip_by_value_7/clip_by_value?
+model_5/tf.compat.v1.floor_div_5/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2-
+model_5/tf.compat.v1.floor_div_5/FloorDiv/y?
)model_5/tf.compat.v1.floor_div_5/FloorDivFloorDiv,model_5/tf.clip_by_value_7/clip_by_value:z:04model_5/tf.compat.v1.floor_div_5/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2+
)model_5/tf.compat.v1.floor_div_5/FloorDiv?
,model_5/tf.math.greater_equal_7/GreaterEqualGreaterEqual"model_5/flatten_5/Reshape:output:0.model_5_tf_math_greater_equal_7_greaterequal_y*
T0*'
_output_shapes
:?????????2.
,model_5/tf.math.greater_equal_7/GreaterEqual?
%model_5/tf.math.floormod_5/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2'
%model_5/tf.math.floormod_5/FloorMod/y?
#model_5/tf.math.floormod_5/FloorModFloorMod,model_5/tf.clip_by_value_7/clip_by_value:z:0.model_5/tf.math.floormod_5/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2%
#model_5/tf.math.floormod_5/FloorMod?
model_5/embedding_17/CastCast,model_5/tf.clip_by_value_7/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_5/embedding_17/Cast?
%model_5/embedding_17/embedding_lookupResourceGather.model_5_embedding_17_embedding_lookup_10264748model_5/embedding_17/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*A
_class7
53loc:@model_5/embedding_17/embedding_lookup/10264748*,
_output_shapes
:??????????*
dtype02'
%model_5/embedding_17/embedding_lookup?
.model_5/embedding_17/embedding_lookup/IdentityIdentity.model_5/embedding_17/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@model_5/embedding_17/embedding_lookup/10264748*,
_output_shapes
:??????????20
.model_5/embedding_17/embedding_lookup/Identity?
0model_5/embedding_17/embedding_lookup/Identity_1Identity7model_5/embedding_17/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????22
0model_5/embedding_17/embedding_lookup/Identity_1?
model_5/embedding_15/CastCast-model_5/tf.compat.v1.floor_div_5/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_5/embedding_15/Cast?
%model_5/embedding_15/embedding_lookupResourceGather.model_5_embedding_15_embedding_lookup_10264754model_5/embedding_15/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*A
_class7
53loc:@model_5/embedding_15/embedding_lookup/10264754*,
_output_shapes
:??????????*
dtype02'
%model_5/embedding_15/embedding_lookup?
.model_5/embedding_15/embedding_lookup/IdentityIdentity.model_5/embedding_15/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@model_5/embedding_15/embedding_lookup/10264754*,
_output_shapes
:??????????20
.model_5/embedding_15/embedding_lookup/Identity?
0model_5/embedding_15/embedding_lookup/Identity_1Identity7model_5/embedding_15/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????22
0model_5/embedding_15/embedding_lookup/Identity_1?
model_5/tf.cast_7/CastCast0model_5/tf.math.greater_equal_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
model_5/tf.cast_7/Cast?
%model_5/tf.__operators__.add_14/AddV2AddV29model_5/embedding_17/embedding_lookup/Identity_1:output:09model_5/embedding_15/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2'
%model_5/tf.__operators__.add_14/AddV2?
model_5/embedding_16/CastCast'model_5/tf.math.floormod_5/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_5/embedding_16/Cast?
%model_5/embedding_16/embedding_lookupResourceGather.model_5_embedding_16_embedding_lookup_10264762model_5/embedding_16/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*A
_class7
53loc:@model_5/embedding_16/embedding_lookup/10264762*,
_output_shapes
:??????????*
dtype02'
%model_5/embedding_16/embedding_lookup?
.model_5/embedding_16/embedding_lookup/IdentityIdentity.model_5/embedding_16/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@model_5/embedding_16/embedding_lookup/10264762*,
_output_shapes
:??????????20
.model_5/embedding_16/embedding_lookup/Identity?
0model_5/embedding_16/embedding_lookup/Identity_1Identity7model_5/embedding_16/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????22
0model_5/embedding_16/embedding_lookup/Identity_1?
%model_5/tf.__operators__.add_15/AddV2AddV2)model_5/tf.__operators__.add_14/AddV2:z:09model_5/embedding_16/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2'
%model_5/tf.__operators__.add_15/AddV2?
'model_5/tf.expand_dims_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'model_5/tf.expand_dims_5/ExpandDims/dim?
#model_5/tf.expand_dims_5/ExpandDims
ExpandDimsmodel_5/tf.cast_7/Cast:y:00model_5/tf.expand_dims_5/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2%
#model_5/tf.expand_dims_5/ExpandDims?
model_5/tf.math.multiply_5/MulMul)model_5/tf.__operators__.add_15/AddV2:z:0,model_5/tf.expand_dims_5/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2 
model_5/tf.math.multiply_5/Mul?
2model_5/tf.math.reduce_sum_5/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :24
2model_5/tf.math.reduce_sum_5/Sum/reduction_indices?
 model_5/tf.math.reduce_sum_5/SumSum"model_5/tf.math.multiply_5/Mul:z:0;model_5/tf.math.reduce_sum_5/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2"
 model_5/tf.math.reduce_sum_5/Sum?
(tf.clip_by_value_8/clip_by_value/MinimumMinimuminputs_1*tf_clip_by_value_8_clip_by_value_minimum_y*
T0*'
_output_shapes
:?????????
2*
(tf.clip_by_value_8/clip_by_value/Minimum?
 tf.clip_by_value_8/clip_by_valueMaximum,tf.clip_by_value_8/clip_by_value/Minimum:z:0"tf_clip_by_value_8_clip_by_value_y*
T0*'
_output_shapes
:?????????
2"
 tf.clip_by_value_8/clip_by_value?
tf.cast_8/CastCast(tf.math.greater_equal_8/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
2
tf.cast_8/Castt
tf.concat_6/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_6/concat/axis?
tf.concat_6/concatConcatV2)model_4/tf.math.reduce_sum_4/Sum:output:0)model_5/tf.math.reduce_sum_5/Sum:output:0 tf.concat_6/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_6/concat}
tf.concat_7/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_7/concat/axis?
tf.concat_7/concatConcatV2$tf.clip_by_value_8/clip_by_value:z:0tf.cast_8/Cast:y:0 tf.concat_7/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
tf.concat_7/concat?
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_18/MatMul/ReadVariableOp?
dense_18/MatMulMatMultf.concat_6/concat:output:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_18/MatMul?
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_18/BiasAdd/ReadVariableOp?
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_18/BiasAddt
dense_18/ReluReludense_18/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_18/Relu?
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_21/MatMul/ReadVariableOp?
dense_21/MatMulMatMultf.concat_7/concat:output:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_21/MatMul?
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_21/BiasAdd/ReadVariableOp?
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_21/BiasAdd?
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_19/MatMul/ReadVariableOp?
dense_19/MatMulMatMuldense_18/Relu:activations:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_19/MatMul?
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_19/BiasAdd/ReadVariableOp?
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_19/BiasAddt
dense_19/ReluReludense_19/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_19/Relu?
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_20/MatMul/ReadVariableOp?
dense_20/MatMulMatMuldense_19/Relu:activations:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_20/MatMul?
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_20/BiasAdd/ReadVariableOp?
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_20/BiasAddt
dense_20/ReluReludense_20/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_20/Relu?
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_22/MatMul/ReadVariableOp?
dense_22/MatMulMatMuldense_21/BiasAdd:output:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_22/MatMul?
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_22/BiasAdd/ReadVariableOp?
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_22/BiasAdd}
tf.concat_8/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_8/concat/axis?
tf.concat_8/concatConcatV2dense_20/Relu:activations:0dense_22/BiasAdd:output:0 tf.concat_8/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_8/concat?
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_23/MatMul/ReadVariableOp?
dense_23/MatMulMatMultf.concat_8/concat:output:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_23/MatMul?
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_23/BiasAdd/ReadVariableOp?
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_23/BiasAdd|
tf.nn.relu_6/ReluReludense_23/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_6/Relu?
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_24/MatMul/ReadVariableOp?
dense_24/MatMulMatMultf.nn.relu_6/Relu:activations:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_24/MatMul?
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_24/BiasAdd/ReadVariableOp?
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_24/BiasAdd?
tf.__operators__.add_16/AddV2AddV2dense_24/BiasAdd:output:0tf.nn.relu_6/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_16/AddV2?
tf.nn.relu_7/ReluRelu!tf.__operators__.add_16/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_7/Relu?
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_25/MatMul/ReadVariableOp?
dense_25/MatMulMatMultf.nn.relu_7/Relu:activations:0&dense_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_25/MatMul?
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_25/BiasAdd/ReadVariableOp?
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_25/BiasAdd?
tf.__operators__.add_17/AddV2AddV2dense_25/BiasAdd:output:0tf.nn.relu_7/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_17/AddV2?
tf.nn.relu_8/ReluRelu!tf.__operators__.add_17/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_8/Relu?
2normalize_2/normalization_2/Reshape/ReadVariableOpReadVariableOp;normalize_2_normalization_2_reshape_readvariableop_resource*
_output_shapes	
:?*
dtype024
2normalize_2/normalization_2/Reshape/ReadVariableOp?
)normalize_2/normalization_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2+
)normalize_2/normalization_2/Reshape/shape?
#normalize_2/normalization_2/ReshapeReshape:normalize_2/normalization_2/Reshape/ReadVariableOp:value:02normalize_2/normalization_2/Reshape/shape:output:0*
T0*
_output_shapes
:	?2%
#normalize_2/normalization_2/Reshape?
4normalize_2/normalization_2/Reshape_1/ReadVariableOpReadVariableOp=normalize_2_normalization_2_reshape_1_readvariableop_resource*
_output_shapes	
:?*
dtype026
4normalize_2/normalization_2/Reshape_1/ReadVariableOp?
+normalize_2/normalization_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2-
+normalize_2/normalization_2/Reshape_1/shape?
%normalize_2/normalization_2/Reshape_1Reshape<normalize_2/normalization_2/Reshape_1/ReadVariableOp:value:04normalize_2/normalization_2/Reshape_1/shape:output:0*
T0*
_output_shapes
:	?2'
%normalize_2/normalization_2/Reshape_1?
normalize_2/normalization_2/subSubtf.nn.relu_8/Relu:activations:0,normalize_2/normalization_2/Reshape:output:0*
T0*(
_output_shapes
:??????????2!
normalize_2/normalization_2/sub?
 normalize_2/normalization_2/SqrtSqrt.normalize_2/normalization_2/Reshape_1:output:0*
T0*
_output_shapes
:	?2"
 normalize_2/normalization_2/Sqrt?
%normalize_2/normalization_2/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32'
%normalize_2/normalization_2/Maximum/y?
#normalize_2/normalization_2/MaximumMaximum$normalize_2/normalization_2/Sqrt:y:0.normalize_2/normalization_2/Maximum/y:output:0*
T0*
_output_shapes
:	?2%
#normalize_2/normalization_2/Maximum?
#normalize_2/normalization_2/truedivRealDiv#normalize_2/normalization_2/sub:z:0'normalize_2/normalization_2/Maximum:z:0*
T0*(
_output_shapes
:??????????2%
#normalize_2/normalization_2/truediv?
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_26/MatMul/ReadVariableOp?
dense_26/MatMulMatMul'normalize_2/normalization_2/truediv:z:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_26/MatMul?
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_26/BiasAdd/ReadVariableOp?
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_26/BiasAdd?
IdentityIdentitydense_26/BiasAdd:output:0 ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp ^dense_26/BiasAdd/ReadVariableOp^dense_26/MatMul/ReadVariableOp&^model_4/embedding_12/embedding_lookup&^model_4/embedding_13/embedding_lookup&^model_4/embedding_14/embedding_lookup&^model_5/embedding_15/embedding_lookup&^model_5/embedding_16/embedding_lookup&^model_5/embedding_17/embedding_lookup3^normalize_2/normalization_2/Reshape/ReadVariableOp5^normalize_2/normalization_2/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
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
:?????????
$
_user_specified_name
inputs/0/0:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/0/1:QM
'
_output_shapes
:?????????

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
?
?
+__inference_dense_18_layer_call_fn_10265408

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_18_layer_call_and_return_conditional_losses_102639292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
J__inference_embedding_13_layer_call_and_return_conditional_losses_10265643

inputs
embedding_lookup_10265637
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_10265637Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*,
_class"
 loc:@embedding_lookup/10265637*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@embedding_lookup/10265637*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
u
/__inference_embedding_14_layer_call_fn_10265616

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_14_layer_call_and_return_conditional_losses_102634172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
J__inference_embedding_14_layer_call_and_return_conditional_losses_10263417

inputs
embedding_lookup_10263411
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_10263411Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*,
_class"
 loc:@embedding_lookup/10263411*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@embedding_lookup/10263411*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
I__inference_normalize_2_layer_call_and_return_conditional_losses_10265560
x3
/normalization_2_reshape_readvariableop_resource5
1normalization_2_reshape_1_readvariableop_resource
identity??&normalization_2/Reshape/ReadVariableOp?(normalization_2/Reshape_1/ReadVariableOp?
&normalization_2/Reshape/ReadVariableOpReadVariableOp/normalization_2_reshape_readvariableop_resource*
_output_shapes	
:?*
dtype02(
&normalization_2/Reshape/ReadVariableOp?
normalization_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_2/Reshape/shape?
normalization_2/ReshapeReshape.normalization_2/Reshape/ReadVariableOp:value:0&normalization_2/Reshape/shape:output:0*
T0*
_output_shapes
:	?2
normalization_2/Reshape?
(normalization_2/Reshape_1/ReadVariableOpReadVariableOp1normalization_2_reshape_1_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(normalization_2/Reshape_1/ReadVariableOp?
normalization_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_2/Reshape_1/shape?
normalization_2/Reshape_1Reshape0normalization_2/Reshape_1/ReadVariableOp:value:0(normalization_2/Reshape_1/shape:output:0*
T0*
_output_shapes
:	?2
normalization_2/Reshape_1?
normalization_2/subSubx normalization_2/Reshape:output:0*
T0*(
_output_shapes
:??????????2
normalization_2/sub?
normalization_2/SqrtSqrt"normalization_2/Reshape_1:output:0*
T0*
_output_shapes
:	?2
normalization_2/Sqrt{
normalization_2/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_2/Maximum/y?
normalization_2/MaximumMaximumnormalization_2/Sqrt:y:0"normalization_2/Maximum/y:output:0*
T0*
_output_shapes
:	?2
normalization_2/Maximum?
normalization_2/truedivRealDivnormalization_2/sub:z:0normalization_2/Maximum:z:0*
T0*(
_output_shapes
:??????????2
normalization_2/truediv?
IdentityIdentitynormalization_2/truediv:z:0'^normalization_2/Reshape/ReadVariableOp)^normalization_2/Reshape_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::2P
&normalization_2/Reshape/ReadVariableOp&normalization_2/Reshape/ReadVariableOp2T
(normalization_2/Reshape_1/ReadVariableOp(normalization_2/Reshape_1/ReadVariableOp:K G
(
_output_shapes
:??????????

_user_specified_namex
?	
?
J__inference_embedding_17_layer_call_and_return_conditional_losses_10263643

inputs
embedding_lookup_10263637
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_10263637Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*,
_class"
 loc:@embedding_lookup/10263637*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@embedding_lookup/10263637*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
#__inference__wrapped_model_10263379

cards0

cards1
bets9
5custom_model_2_tf_math_greater_equal_8_greaterequal_yA
=custom_model_2_model_4_tf_math_greater_equal_6_greaterequal_yA
=custom_model_2_model_4_embedding_14_embedding_lookup_10263229A
=custom_model_2_model_4_embedding_12_embedding_lookup_10263235A
=custom_model_2_model_4_embedding_13_embedding_lookup_10263243A
=custom_model_2_model_5_tf_math_greater_equal_7_greaterequal_yA
=custom_model_2_model_5_embedding_17_embedding_lookup_10263267A
=custom_model_2_model_5_embedding_15_embedding_lookup_10263273A
=custom_model_2_model_5_embedding_16_embedding_lookup_10263281=
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
identity??.custom_model_2/dense_18/BiasAdd/ReadVariableOp?-custom_model_2/dense_18/MatMul/ReadVariableOp?.custom_model_2/dense_19/BiasAdd/ReadVariableOp?-custom_model_2/dense_19/MatMul/ReadVariableOp?.custom_model_2/dense_20/BiasAdd/ReadVariableOp?-custom_model_2/dense_20/MatMul/ReadVariableOp?.custom_model_2/dense_21/BiasAdd/ReadVariableOp?-custom_model_2/dense_21/MatMul/ReadVariableOp?.custom_model_2/dense_22/BiasAdd/ReadVariableOp?-custom_model_2/dense_22/MatMul/ReadVariableOp?.custom_model_2/dense_23/BiasAdd/ReadVariableOp?-custom_model_2/dense_23/MatMul/ReadVariableOp?.custom_model_2/dense_24/BiasAdd/ReadVariableOp?-custom_model_2/dense_24/MatMul/ReadVariableOp?.custom_model_2/dense_25/BiasAdd/ReadVariableOp?-custom_model_2/dense_25/MatMul/ReadVariableOp?.custom_model_2/dense_26/BiasAdd/ReadVariableOp?-custom_model_2/dense_26/MatMul/ReadVariableOp?4custom_model_2/model_4/embedding_12/embedding_lookup?4custom_model_2/model_4/embedding_13/embedding_lookup?4custom_model_2/model_4/embedding_14/embedding_lookup?4custom_model_2/model_5/embedding_15/embedding_lookup?4custom_model_2/model_5/embedding_16/embedding_lookup?4custom_model_2/model_5/embedding_17/embedding_lookup?Acustom_model_2/normalize_2/normalization_2/Reshape/ReadVariableOp?Ccustom_model_2/normalize_2/normalization_2/Reshape_1/ReadVariableOp?
3custom_model_2/tf.math.greater_equal_8/GreaterEqualGreaterEqualbets5custom_model_2_tf_math_greater_equal_8_greaterequal_y*
T0*'
_output_shapes
:?????????
25
3custom_model_2/tf.math.greater_equal_8/GreaterEqual?
&custom_model_2/model_4/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2(
&custom_model_2/model_4/flatten_4/Const?
(custom_model_2/model_4/flatten_4/ReshapeReshapecards0/custom_model_2/model_4/flatten_4/Const:output:0*
T0*'
_output_shapes
:?????????2*
(custom_model_2/model_4/flatten_4/Reshape?
Acustom_model_2/model_4/tf.clip_by_value_6/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2C
Acustom_model_2/model_4/tf.clip_by_value_6/clip_by_value/Minimum/y?
?custom_model_2/model_4/tf.clip_by_value_6/clip_by_value/MinimumMinimum1custom_model_2/model_4/flatten_4/Reshape:output:0Jcustom_model_2/model_4/tf.clip_by_value_6/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2A
?custom_model_2/model_4/tf.clip_by_value_6/clip_by_value/Minimum?
9custom_model_2/model_4/tf.clip_by_value_6/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2;
9custom_model_2/model_4/tf.clip_by_value_6/clip_by_value/y?
7custom_model_2/model_4/tf.clip_by_value_6/clip_by_valueMaximumCcustom_model_2/model_4/tf.clip_by_value_6/clip_by_value/Minimum:z:0Bcustom_model_2/model_4/tf.clip_by_value_6/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????29
7custom_model_2/model_4/tf.clip_by_value_6/clip_by_value?
:custom_model_2/model_4/tf.compat.v1.floor_div_4/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2<
:custom_model_2/model_4/tf.compat.v1.floor_div_4/FloorDiv/y?
8custom_model_2/model_4/tf.compat.v1.floor_div_4/FloorDivFloorDiv;custom_model_2/model_4/tf.clip_by_value_6/clip_by_value:z:0Ccustom_model_2/model_4/tf.compat.v1.floor_div_4/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2:
8custom_model_2/model_4/tf.compat.v1.floor_div_4/FloorDiv?
;custom_model_2/model_4/tf.math.greater_equal_6/GreaterEqualGreaterEqual1custom_model_2/model_4/flatten_4/Reshape:output:0=custom_model_2_model_4_tf_math_greater_equal_6_greaterequal_y*
T0*'
_output_shapes
:?????????2=
;custom_model_2/model_4/tf.math.greater_equal_6/GreaterEqual?
4custom_model_2/model_4/tf.math.floormod_4/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@26
4custom_model_2/model_4/tf.math.floormod_4/FloorMod/y?
2custom_model_2/model_4/tf.math.floormod_4/FloorModFloorMod;custom_model_2/model_4/tf.clip_by_value_6/clip_by_value:z:0=custom_model_2/model_4/tf.math.floormod_4/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????24
2custom_model_2/model_4/tf.math.floormod_4/FloorMod?
(custom_model_2/model_4/embedding_14/CastCast;custom_model_2/model_4/tf.clip_by_value_6/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2*
(custom_model_2/model_4/embedding_14/Cast?
4custom_model_2/model_4/embedding_14/embedding_lookupResourceGather=custom_model_2_model_4_embedding_14_embedding_lookup_10263229,custom_model_2/model_4/embedding_14/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*P
_classF
DBloc:@custom_model_2/model_4/embedding_14/embedding_lookup/10263229*,
_output_shapes
:??????????*
dtype026
4custom_model_2/model_4/embedding_14/embedding_lookup?
=custom_model_2/model_4/embedding_14/embedding_lookup/IdentityIdentity=custom_model_2/model_4/embedding_14/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*P
_classF
DBloc:@custom_model_2/model_4/embedding_14/embedding_lookup/10263229*,
_output_shapes
:??????????2?
=custom_model_2/model_4/embedding_14/embedding_lookup/Identity?
?custom_model_2/model_4/embedding_14/embedding_lookup/Identity_1IdentityFcustom_model_2/model_4/embedding_14/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2A
?custom_model_2/model_4/embedding_14/embedding_lookup/Identity_1?
(custom_model_2/model_4/embedding_12/CastCast<custom_model_2/model_4/tf.compat.v1.floor_div_4/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2*
(custom_model_2/model_4/embedding_12/Cast?
4custom_model_2/model_4/embedding_12/embedding_lookupResourceGather=custom_model_2_model_4_embedding_12_embedding_lookup_10263235,custom_model_2/model_4/embedding_12/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*P
_classF
DBloc:@custom_model_2/model_4/embedding_12/embedding_lookup/10263235*,
_output_shapes
:??????????*
dtype026
4custom_model_2/model_4/embedding_12/embedding_lookup?
=custom_model_2/model_4/embedding_12/embedding_lookup/IdentityIdentity=custom_model_2/model_4/embedding_12/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*P
_classF
DBloc:@custom_model_2/model_4/embedding_12/embedding_lookup/10263235*,
_output_shapes
:??????????2?
=custom_model_2/model_4/embedding_12/embedding_lookup/Identity?
?custom_model_2/model_4/embedding_12/embedding_lookup/Identity_1IdentityFcustom_model_2/model_4/embedding_12/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2A
?custom_model_2/model_4/embedding_12/embedding_lookup/Identity_1?
%custom_model_2/model_4/tf.cast_6/CastCast?custom_model_2/model_4/tf.math.greater_equal_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2'
%custom_model_2/model_4/tf.cast_6/Cast?
4custom_model_2/model_4/tf.__operators__.add_12/AddV2AddV2Hcustom_model_2/model_4/embedding_14/embedding_lookup/Identity_1:output:0Hcustom_model_2/model_4/embedding_12/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????26
4custom_model_2/model_4/tf.__operators__.add_12/AddV2?
(custom_model_2/model_4/embedding_13/CastCast6custom_model_2/model_4/tf.math.floormod_4/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2*
(custom_model_2/model_4/embedding_13/Cast?
4custom_model_2/model_4/embedding_13/embedding_lookupResourceGather=custom_model_2_model_4_embedding_13_embedding_lookup_10263243,custom_model_2/model_4/embedding_13/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*P
_classF
DBloc:@custom_model_2/model_4/embedding_13/embedding_lookup/10263243*,
_output_shapes
:??????????*
dtype026
4custom_model_2/model_4/embedding_13/embedding_lookup?
=custom_model_2/model_4/embedding_13/embedding_lookup/IdentityIdentity=custom_model_2/model_4/embedding_13/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*P
_classF
DBloc:@custom_model_2/model_4/embedding_13/embedding_lookup/10263243*,
_output_shapes
:??????????2?
=custom_model_2/model_4/embedding_13/embedding_lookup/Identity?
?custom_model_2/model_4/embedding_13/embedding_lookup/Identity_1IdentityFcustom_model_2/model_4/embedding_13/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2A
?custom_model_2/model_4/embedding_13/embedding_lookup/Identity_1?
4custom_model_2/model_4/tf.__operators__.add_13/AddV2AddV28custom_model_2/model_4/tf.__operators__.add_12/AddV2:z:0Hcustom_model_2/model_4/embedding_13/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????26
4custom_model_2/model_4/tf.__operators__.add_13/AddV2?
6custom_model_2/model_4/tf.expand_dims_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????28
6custom_model_2/model_4/tf.expand_dims_4/ExpandDims/dim?
2custom_model_2/model_4/tf.expand_dims_4/ExpandDims
ExpandDims)custom_model_2/model_4/tf.cast_6/Cast:y:0?custom_model_2/model_4/tf.expand_dims_4/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????24
2custom_model_2/model_4/tf.expand_dims_4/ExpandDims?
-custom_model_2/model_4/tf.math.multiply_4/MulMul8custom_model_2/model_4/tf.__operators__.add_13/AddV2:z:0;custom_model_2/model_4/tf.expand_dims_4/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2/
-custom_model_2/model_4/tf.math.multiply_4/Mul?
Acustom_model_2/model_4/tf.math.reduce_sum_4/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2C
Acustom_model_2/model_4/tf.math.reduce_sum_4/Sum/reduction_indices?
/custom_model_2/model_4/tf.math.reduce_sum_4/SumSum1custom_model_2/model_4/tf.math.multiply_4/Mul:z:0Jcustom_model_2/model_4/tf.math.reduce_sum_4/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????21
/custom_model_2/model_4/tf.math.reduce_sum_4/Sum?
&custom_model_2/model_5/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2(
&custom_model_2/model_5/flatten_5/Const?
(custom_model_2/model_5/flatten_5/ReshapeReshapecards1/custom_model_2/model_5/flatten_5/Const:output:0*
T0*'
_output_shapes
:?????????2*
(custom_model_2/model_5/flatten_5/Reshape?
Acustom_model_2/model_5/tf.clip_by_value_7/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2C
Acustom_model_2/model_5/tf.clip_by_value_7/clip_by_value/Minimum/y?
?custom_model_2/model_5/tf.clip_by_value_7/clip_by_value/MinimumMinimum1custom_model_2/model_5/flatten_5/Reshape:output:0Jcustom_model_2/model_5/tf.clip_by_value_7/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2A
?custom_model_2/model_5/tf.clip_by_value_7/clip_by_value/Minimum?
9custom_model_2/model_5/tf.clip_by_value_7/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2;
9custom_model_2/model_5/tf.clip_by_value_7/clip_by_value/y?
7custom_model_2/model_5/tf.clip_by_value_7/clip_by_valueMaximumCcustom_model_2/model_5/tf.clip_by_value_7/clip_by_value/Minimum:z:0Bcustom_model_2/model_5/tf.clip_by_value_7/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????29
7custom_model_2/model_5/tf.clip_by_value_7/clip_by_value?
:custom_model_2/model_5/tf.compat.v1.floor_div_5/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2<
:custom_model_2/model_5/tf.compat.v1.floor_div_5/FloorDiv/y?
8custom_model_2/model_5/tf.compat.v1.floor_div_5/FloorDivFloorDiv;custom_model_2/model_5/tf.clip_by_value_7/clip_by_value:z:0Ccustom_model_2/model_5/tf.compat.v1.floor_div_5/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2:
8custom_model_2/model_5/tf.compat.v1.floor_div_5/FloorDiv?
;custom_model_2/model_5/tf.math.greater_equal_7/GreaterEqualGreaterEqual1custom_model_2/model_5/flatten_5/Reshape:output:0=custom_model_2_model_5_tf_math_greater_equal_7_greaterequal_y*
T0*'
_output_shapes
:?????????2=
;custom_model_2/model_5/tf.math.greater_equal_7/GreaterEqual?
4custom_model_2/model_5/tf.math.floormod_5/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@26
4custom_model_2/model_5/tf.math.floormod_5/FloorMod/y?
2custom_model_2/model_5/tf.math.floormod_5/FloorModFloorMod;custom_model_2/model_5/tf.clip_by_value_7/clip_by_value:z:0=custom_model_2/model_5/tf.math.floormod_5/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????24
2custom_model_2/model_5/tf.math.floormod_5/FloorMod?
(custom_model_2/model_5/embedding_17/CastCast;custom_model_2/model_5/tf.clip_by_value_7/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2*
(custom_model_2/model_5/embedding_17/Cast?
4custom_model_2/model_5/embedding_17/embedding_lookupResourceGather=custom_model_2_model_5_embedding_17_embedding_lookup_10263267,custom_model_2/model_5/embedding_17/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*P
_classF
DBloc:@custom_model_2/model_5/embedding_17/embedding_lookup/10263267*,
_output_shapes
:??????????*
dtype026
4custom_model_2/model_5/embedding_17/embedding_lookup?
=custom_model_2/model_5/embedding_17/embedding_lookup/IdentityIdentity=custom_model_2/model_5/embedding_17/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*P
_classF
DBloc:@custom_model_2/model_5/embedding_17/embedding_lookup/10263267*,
_output_shapes
:??????????2?
=custom_model_2/model_5/embedding_17/embedding_lookup/Identity?
?custom_model_2/model_5/embedding_17/embedding_lookup/Identity_1IdentityFcustom_model_2/model_5/embedding_17/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2A
?custom_model_2/model_5/embedding_17/embedding_lookup/Identity_1?
(custom_model_2/model_5/embedding_15/CastCast<custom_model_2/model_5/tf.compat.v1.floor_div_5/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2*
(custom_model_2/model_5/embedding_15/Cast?
4custom_model_2/model_5/embedding_15/embedding_lookupResourceGather=custom_model_2_model_5_embedding_15_embedding_lookup_10263273,custom_model_2/model_5/embedding_15/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*P
_classF
DBloc:@custom_model_2/model_5/embedding_15/embedding_lookup/10263273*,
_output_shapes
:??????????*
dtype026
4custom_model_2/model_5/embedding_15/embedding_lookup?
=custom_model_2/model_5/embedding_15/embedding_lookup/IdentityIdentity=custom_model_2/model_5/embedding_15/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*P
_classF
DBloc:@custom_model_2/model_5/embedding_15/embedding_lookup/10263273*,
_output_shapes
:??????????2?
=custom_model_2/model_5/embedding_15/embedding_lookup/Identity?
?custom_model_2/model_5/embedding_15/embedding_lookup/Identity_1IdentityFcustom_model_2/model_5/embedding_15/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2A
?custom_model_2/model_5/embedding_15/embedding_lookup/Identity_1?
%custom_model_2/model_5/tf.cast_7/CastCast?custom_model_2/model_5/tf.math.greater_equal_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2'
%custom_model_2/model_5/tf.cast_7/Cast?
4custom_model_2/model_5/tf.__operators__.add_14/AddV2AddV2Hcustom_model_2/model_5/embedding_17/embedding_lookup/Identity_1:output:0Hcustom_model_2/model_5/embedding_15/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????26
4custom_model_2/model_5/tf.__operators__.add_14/AddV2?
(custom_model_2/model_5/embedding_16/CastCast6custom_model_2/model_5/tf.math.floormod_5/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2*
(custom_model_2/model_5/embedding_16/Cast?
4custom_model_2/model_5/embedding_16/embedding_lookupResourceGather=custom_model_2_model_5_embedding_16_embedding_lookup_10263281,custom_model_2/model_5/embedding_16/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*P
_classF
DBloc:@custom_model_2/model_5/embedding_16/embedding_lookup/10263281*,
_output_shapes
:??????????*
dtype026
4custom_model_2/model_5/embedding_16/embedding_lookup?
=custom_model_2/model_5/embedding_16/embedding_lookup/IdentityIdentity=custom_model_2/model_5/embedding_16/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*P
_classF
DBloc:@custom_model_2/model_5/embedding_16/embedding_lookup/10263281*,
_output_shapes
:??????????2?
=custom_model_2/model_5/embedding_16/embedding_lookup/Identity?
?custom_model_2/model_5/embedding_16/embedding_lookup/Identity_1IdentityFcustom_model_2/model_5/embedding_16/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2A
?custom_model_2/model_5/embedding_16/embedding_lookup/Identity_1?
4custom_model_2/model_5/tf.__operators__.add_15/AddV2AddV28custom_model_2/model_5/tf.__operators__.add_14/AddV2:z:0Hcustom_model_2/model_5/embedding_16/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????26
4custom_model_2/model_5/tf.__operators__.add_15/AddV2?
6custom_model_2/model_5/tf.expand_dims_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????28
6custom_model_2/model_5/tf.expand_dims_5/ExpandDims/dim?
2custom_model_2/model_5/tf.expand_dims_5/ExpandDims
ExpandDims)custom_model_2/model_5/tf.cast_7/Cast:y:0?custom_model_2/model_5/tf.expand_dims_5/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????24
2custom_model_2/model_5/tf.expand_dims_5/ExpandDims?
-custom_model_2/model_5/tf.math.multiply_5/MulMul8custom_model_2/model_5/tf.__operators__.add_15/AddV2:z:0;custom_model_2/model_5/tf.expand_dims_5/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2/
-custom_model_2/model_5/tf.math.multiply_5/Mul?
Acustom_model_2/model_5/tf.math.reduce_sum_5/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2C
Acustom_model_2/model_5/tf.math.reduce_sum_5/Sum/reduction_indices?
/custom_model_2/model_5/tf.math.reduce_sum_5/SumSum1custom_model_2/model_5/tf.math.multiply_5/Mul:z:0Jcustom_model_2/model_5/tf.math.reduce_sum_5/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????21
/custom_model_2/model_5/tf.math.reduce_sum_5/Sum?
7custom_model_2/tf.clip_by_value_8/clip_by_value/MinimumMinimumbets9custom_model_2_tf_clip_by_value_8_clip_by_value_minimum_y*
T0*'
_output_shapes
:?????????
29
7custom_model_2/tf.clip_by_value_8/clip_by_value/Minimum?
/custom_model_2/tf.clip_by_value_8/clip_by_valueMaximum;custom_model_2/tf.clip_by_value_8/clip_by_value/Minimum:z:01custom_model_2_tf_clip_by_value_8_clip_by_value_y*
T0*'
_output_shapes
:?????????
21
/custom_model_2/tf.clip_by_value_8/clip_by_value?
custom_model_2/tf.cast_8/CastCast7custom_model_2/tf.math.greater_equal_8/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
2
custom_model_2/tf.cast_8/Cast?
&custom_model_2/tf.concat_6/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2(
&custom_model_2/tf.concat_6/concat/axis?
!custom_model_2/tf.concat_6/concatConcatV28custom_model_2/model_4/tf.math.reduce_sum_4/Sum:output:08custom_model_2/model_5/tf.math.reduce_sum_5/Sum:output:0/custom_model_2/tf.concat_6/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2#
!custom_model_2/tf.concat_6/concat?
&custom_model_2/tf.concat_7/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2(
&custom_model_2/tf.concat_7/concat/axis?
!custom_model_2/tf.concat_7/concatConcatV23custom_model_2/tf.clip_by_value_8/clip_by_value:z:0!custom_model_2/tf.cast_8/Cast:y:0/custom_model_2/tf.concat_7/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2#
!custom_model_2/tf.concat_7/concat?
-custom_model_2/dense_18/MatMul/ReadVariableOpReadVariableOp6custom_model_2_dense_18_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-custom_model_2/dense_18/MatMul/ReadVariableOp?
custom_model_2/dense_18/MatMulMatMul*custom_model_2/tf.concat_6/concat:output:05custom_model_2/dense_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
custom_model_2/dense_18/MatMul?
.custom_model_2/dense_18/BiasAdd/ReadVariableOpReadVariableOp7custom_model_2_dense_18_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.custom_model_2/dense_18/BiasAdd/ReadVariableOp?
custom_model_2/dense_18/BiasAddBiasAdd(custom_model_2/dense_18/MatMul:product:06custom_model_2/dense_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
custom_model_2/dense_18/BiasAdd?
custom_model_2/dense_18/ReluRelu(custom_model_2/dense_18/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
custom_model_2/dense_18/Relu?
-custom_model_2/dense_21/MatMul/ReadVariableOpReadVariableOp6custom_model_2_dense_21_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02/
-custom_model_2/dense_21/MatMul/ReadVariableOp?
custom_model_2/dense_21/MatMulMatMul*custom_model_2/tf.concat_7/concat:output:05custom_model_2/dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
custom_model_2/dense_21/MatMul?
.custom_model_2/dense_21/BiasAdd/ReadVariableOpReadVariableOp7custom_model_2_dense_21_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.custom_model_2/dense_21/BiasAdd/ReadVariableOp?
custom_model_2/dense_21/BiasAddBiasAdd(custom_model_2/dense_21/MatMul:product:06custom_model_2/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
custom_model_2/dense_21/BiasAdd?
-custom_model_2/dense_19/MatMul/ReadVariableOpReadVariableOp6custom_model_2_dense_19_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-custom_model_2/dense_19/MatMul/ReadVariableOp?
custom_model_2/dense_19/MatMulMatMul*custom_model_2/dense_18/Relu:activations:05custom_model_2/dense_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
custom_model_2/dense_19/MatMul?
.custom_model_2/dense_19/BiasAdd/ReadVariableOpReadVariableOp7custom_model_2_dense_19_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.custom_model_2/dense_19/BiasAdd/ReadVariableOp?
custom_model_2/dense_19/BiasAddBiasAdd(custom_model_2/dense_19/MatMul:product:06custom_model_2/dense_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
custom_model_2/dense_19/BiasAdd?
custom_model_2/dense_19/ReluRelu(custom_model_2/dense_19/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
custom_model_2/dense_19/Relu?
-custom_model_2/dense_20/MatMul/ReadVariableOpReadVariableOp6custom_model_2_dense_20_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-custom_model_2/dense_20/MatMul/ReadVariableOp?
custom_model_2/dense_20/MatMulMatMul*custom_model_2/dense_19/Relu:activations:05custom_model_2/dense_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
custom_model_2/dense_20/MatMul?
.custom_model_2/dense_20/BiasAdd/ReadVariableOpReadVariableOp7custom_model_2_dense_20_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.custom_model_2/dense_20/BiasAdd/ReadVariableOp?
custom_model_2/dense_20/BiasAddBiasAdd(custom_model_2/dense_20/MatMul:product:06custom_model_2/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
custom_model_2/dense_20/BiasAdd?
custom_model_2/dense_20/ReluRelu(custom_model_2/dense_20/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
custom_model_2/dense_20/Relu?
-custom_model_2/dense_22/MatMul/ReadVariableOpReadVariableOp6custom_model_2_dense_22_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-custom_model_2/dense_22/MatMul/ReadVariableOp?
custom_model_2/dense_22/MatMulMatMul(custom_model_2/dense_21/BiasAdd:output:05custom_model_2/dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
custom_model_2/dense_22/MatMul?
.custom_model_2/dense_22/BiasAdd/ReadVariableOpReadVariableOp7custom_model_2_dense_22_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.custom_model_2/dense_22/BiasAdd/ReadVariableOp?
custom_model_2/dense_22/BiasAddBiasAdd(custom_model_2/dense_22/MatMul:product:06custom_model_2/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
custom_model_2/dense_22/BiasAdd?
&custom_model_2/tf.concat_8/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2(
&custom_model_2/tf.concat_8/concat/axis?
!custom_model_2/tf.concat_8/concatConcatV2*custom_model_2/dense_20/Relu:activations:0(custom_model_2/dense_22/BiasAdd:output:0/custom_model_2/tf.concat_8/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2#
!custom_model_2/tf.concat_8/concat?
-custom_model_2/dense_23/MatMul/ReadVariableOpReadVariableOp6custom_model_2_dense_23_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-custom_model_2/dense_23/MatMul/ReadVariableOp?
custom_model_2/dense_23/MatMulMatMul*custom_model_2/tf.concat_8/concat:output:05custom_model_2/dense_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
custom_model_2/dense_23/MatMul?
.custom_model_2/dense_23/BiasAdd/ReadVariableOpReadVariableOp7custom_model_2_dense_23_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.custom_model_2/dense_23/BiasAdd/ReadVariableOp?
custom_model_2/dense_23/BiasAddBiasAdd(custom_model_2/dense_23/MatMul:product:06custom_model_2/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
custom_model_2/dense_23/BiasAdd?
 custom_model_2/tf.nn.relu_6/ReluRelu(custom_model_2/dense_23/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2"
 custom_model_2/tf.nn.relu_6/Relu?
-custom_model_2/dense_24/MatMul/ReadVariableOpReadVariableOp6custom_model_2_dense_24_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-custom_model_2/dense_24/MatMul/ReadVariableOp?
custom_model_2/dense_24/MatMulMatMul.custom_model_2/tf.nn.relu_6/Relu:activations:05custom_model_2/dense_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
custom_model_2/dense_24/MatMul?
.custom_model_2/dense_24/BiasAdd/ReadVariableOpReadVariableOp7custom_model_2_dense_24_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.custom_model_2/dense_24/BiasAdd/ReadVariableOp?
custom_model_2/dense_24/BiasAddBiasAdd(custom_model_2/dense_24/MatMul:product:06custom_model_2/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
custom_model_2/dense_24/BiasAdd?
,custom_model_2/tf.__operators__.add_16/AddV2AddV2(custom_model_2/dense_24/BiasAdd:output:0.custom_model_2/tf.nn.relu_6/Relu:activations:0*
T0*(
_output_shapes
:??????????2.
,custom_model_2/tf.__operators__.add_16/AddV2?
 custom_model_2/tf.nn.relu_7/ReluRelu0custom_model_2/tf.__operators__.add_16/AddV2:z:0*
T0*(
_output_shapes
:??????????2"
 custom_model_2/tf.nn.relu_7/Relu?
-custom_model_2/dense_25/MatMul/ReadVariableOpReadVariableOp6custom_model_2_dense_25_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-custom_model_2/dense_25/MatMul/ReadVariableOp?
custom_model_2/dense_25/MatMulMatMul.custom_model_2/tf.nn.relu_7/Relu:activations:05custom_model_2/dense_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
custom_model_2/dense_25/MatMul?
.custom_model_2/dense_25/BiasAdd/ReadVariableOpReadVariableOp7custom_model_2_dense_25_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.custom_model_2/dense_25/BiasAdd/ReadVariableOp?
custom_model_2/dense_25/BiasAddBiasAdd(custom_model_2/dense_25/MatMul:product:06custom_model_2/dense_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
custom_model_2/dense_25/BiasAdd?
,custom_model_2/tf.__operators__.add_17/AddV2AddV2(custom_model_2/dense_25/BiasAdd:output:0.custom_model_2/tf.nn.relu_7/Relu:activations:0*
T0*(
_output_shapes
:??????????2.
,custom_model_2/tf.__operators__.add_17/AddV2?
 custom_model_2/tf.nn.relu_8/ReluRelu0custom_model_2/tf.__operators__.add_17/AddV2:z:0*
T0*(
_output_shapes
:??????????2"
 custom_model_2/tf.nn.relu_8/Relu?
Acustom_model_2/normalize_2/normalization_2/Reshape/ReadVariableOpReadVariableOpJcustom_model_2_normalize_2_normalization_2_reshape_readvariableop_resource*
_output_shapes	
:?*
dtype02C
Acustom_model_2/normalize_2/normalization_2/Reshape/ReadVariableOp?
8custom_model_2/normalize_2/normalization_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2:
8custom_model_2/normalize_2/normalization_2/Reshape/shape?
2custom_model_2/normalize_2/normalization_2/ReshapeReshapeIcustom_model_2/normalize_2/normalization_2/Reshape/ReadVariableOp:value:0Acustom_model_2/normalize_2/normalization_2/Reshape/shape:output:0*
T0*
_output_shapes
:	?24
2custom_model_2/normalize_2/normalization_2/Reshape?
Ccustom_model_2/normalize_2/normalization_2/Reshape_1/ReadVariableOpReadVariableOpLcustom_model_2_normalize_2_normalization_2_reshape_1_readvariableop_resource*
_output_shapes	
:?*
dtype02E
Ccustom_model_2/normalize_2/normalization_2/Reshape_1/ReadVariableOp?
:custom_model_2/normalize_2/normalization_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2<
:custom_model_2/normalize_2/normalization_2/Reshape_1/shape?
4custom_model_2/normalize_2/normalization_2/Reshape_1ReshapeKcustom_model_2/normalize_2/normalization_2/Reshape_1/ReadVariableOp:value:0Ccustom_model_2/normalize_2/normalization_2/Reshape_1/shape:output:0*
T0*
_output_shapes
:	?26
4custom_model_2/normalize_2/normalization_2/Reshape_1?
.custom_model_2/normalize_2/normalization_2/subSub.custom_model_2/tf.nn.relu_8/Relu:activations:0;custom_model_2/normalize_2/normalization_2/Reshape:output:0*
T0*(
_output_shapes
:??????????20
.custom_model_2/normalize_2/normalization_2/sub?
/custom_model_2/normalize_2/normalization_2/SqrtSqrt=custom_model_2/normalize_2/normalization_2/Reshape_1:output:0*
T0*
_output_shapes
:	?21
/custom_model_2/normalize_2/normalization_2/Sqrt?
4custom_model_2/normalize_2/normalization_2/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???326
4custom_model_2/normalize_2/normalization_2/Maximum/y?
2custom_model_2/normalize_2/normalization_2/MaximumMaximum3custom_model_2/normalize_2/normalization_2/Sqrt:y:0=custom_model_2/normalize_2/normalization_2/Maximum/y:output:0*
T0*
_output_shapes
:	?24
2custom_model_2/normalize_2/normalization_2/Maximum?
2custom_model_2/normalize_2/normalization_2/truedivRealDiv2custom_model_2/normalize_2/normalization_2/sub:z:06custom_model_2/normalize_2/normalization_2/Maximum:z:0*
T0*(
_output_shapes
:??????????24
2custom_model_2/normalize_2/normalization_2/truediv?
-custom_model_2/dense_26/MatMul/ReadVariableOpReadVariableOp6custom_model_2_dense_26_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02/
-custom_model_2/dense_26/MatMul/ReadVariableOp?
custom_model_2/dense_26/MatMulMatMul6custom_model_2/normalize_2/normalization_2/truediv:z:05custom_model_2/dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2 
custom_model_2/dense_26/MatMul?
.custom_model_2/dense_26/BiasAdd/ReadVariableOpReadVariableOp7custom_model_2_dense_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.custom_model_2/dense_26/BiasAdd/ReadVariableOp?
custom_model_2/dense_26/BiasAddBiasAdd(custom_model_2/dense_26/MatMul:product:06custom_model_2/dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2!
custom_model_2/dense_26/BiasAdd?
IdentityIdentity(custom_model_2/dense_26/BiasAdd:output:0/^custom_model_2/dense_18/BiasAdd/ReadVariableOp.^custom_model_2/dense_18/MatMul/ReadVariableOp/^custom_model_2/dense_19/BiasAdd/ReadVariableOp.^custom_model_2/dense_19/MatMul/ReadVariableOp/^custom_model_2/dense_20/BiasAdd/ReadVariableOp.^custom_model_2/dense_20/MatMul/ReadVariableOp/^custom_model_2/dense_21/BiasAdd/ReadVariableOp.^custom_model_2/dense_21/MatMul/ReadVariableOp/^custom_model_2/dense_22/BiasAdd/ReadVariableOp.^custom_model_2/dense_22/MatMul/ReadVariableOp/^custom_model_2/dense_23/BiasAdd/ReadVariableOp.^custom_model_2/dense_23/MatMul/ReadVariableOp/^custom_model_2/dense_24/BiasAdd/ReadVariableOp.^custom_model_2/dense_24/MatMul/ReadVariableOp/^custom_model_2/dense_25/BiasAdd/ReadVariableOp.^custom_model_2/dense_25/MatMul/ReadVariableOp/^custom_model_2/dense_26/BiasAdd/ReadVariableOp.^custom_model_2/dense_26/MatMul/ReadVariableOp5^custom_model_2/model_4/embedding_12/embedding_lookup5^custom_model_2/model_4/embedding_13/embedding_lookup5^custom_model_2/model_4/embedding_14/embedding_lookup5^custom_model_2/model_5/embedding_15/embedding_lookup5^custom_model_2/model_5/embedding_16/embedding_lookup5^custom_model_2/model_5/embedding_17/embedding_lookupB^custom_model_2/normalize_2/normalization_2/Reshape/ReadVariableOpD^custom_model_2/normalize_2/normalization_2/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
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
4custom_model_2/model_5/embedding_17/embedding_lookup4custom_model_2/model_5/embedding_17/embedding_lookup2?
Acustom_model_2/normalize_2/normalization_2/Reshape/ReadVariableOpAcustom_model_2/normalize_2/normalization_2/Reshape/ReadVariableOp2?
Ccustom_model_2/normalize_2/normalization_2/Reshape_1/ReadVariableOpCcustom_model_2/normalize_2/normalization_2/Reshape_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_namecards0:OK
'
_output_shapes
:?????????
 
_user_specified_namecards1:MI
'
_output_shapes
:?????????


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
?
?
1__inference_custom_model_2_layer_call_fn_10265168

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
identity??StatefulPartitionedCall?
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
:?????????*<
_read_only_resource_inputs
	
 !*2
config_proto" 

CPU

GPU2 *0J 8? *U
fPRN
L__inference_custom_model_2_layer_call_and_return_conditional_losses_102645462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
: : :::: :::: : ::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:?????????
$
_user_specified_name
inputs/0/0:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/0/1:QM
'
_output_shapes
:?????????

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
?	
?
F__inference_dense_18_layer_call_and_return_conditional_losses_10263929

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
G__inference_flatten_4_layer_call_and_return_conditional_losses_10263389

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?-
?
E__inference_model_5_layer_call_and_return_conditional_losses_10263820

inputs*
&tf_math_greater_equal_7_greaterequal_y
embedding_17_10263802
embedding_15_10263805
embedding_16_10263810
identity??$embedding_15/StatefulPartitionedCall?$embedding_16/StatefulPartitionedCall?$embedding_17/StatefulPartitionedCall?
flatten_5/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_flatten_5_layer_call_and_return_conditional_losses_102636152
flatten_5/PartitionedCall?
*tf.clip_by_value_7/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2,
*tf.clip_by_value_7/clip_by_value/Minimum/y?
(tf.clip_by_value_7/clip_by_value/MinimumMinimum"flatten_5/PartitionedCall:output:03tf.clip_by_value_7/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2*
(tf.clip_by_value_7/clip_by_value/Minimum?
"tf.clip_by_value_7/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_7/clip_by_value/y?
 tf.clip_by_value_7/clip_by_valueMaximum,tf.clip_by_value_7/clip_by_value/Minimum:z:0+tf.clip_by_value_7/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2"
 tf.clip_by_value_7/clip_by_value?
#tf.compat.v1.floor_div_5/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2%
#tf.compat.v1.floor_div_5/FloorDiv/y?
!tf.compat.v1.floor_div_5/FloorDivFloorDiv$tf.clip_by_value_7/clip_by_value:z:0,tf.compat.v1.floor_div_5/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.compat.v1.floor_div_5/FloorDiv?
$tf.math.greater_equal_7/GreaterEqualGreaterEqual"flatten_5/PartitionedCall:output:0&tf_math_greater_equal_7_greaterequal_y*
T0*'
_output_shapes
:?????????2&
$tf.math.greater_equal_7/GreaterEqual?
tf.math.floormod_5/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod_5/FloorMod/y?
tf.math.floormod_5/FloorModFloorMod$tf.clip_by_value_7/clip_by_value:z:0&tf.math.floormod_5/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_5/FloorMod?
$embedding_17/StatefulPartitionedCallStatefulPartitionedCall$tf.clip_by_value_7/clip_by_value:z:0embedding_17_10263802*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_17_layer_call_and_return_conditional_losses_102636432&
$embedding_17/StatefulPartitionedCall?
$embedding_15/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.floor_div_5/FloorDiv:z:0embedding_15_10263805*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_15_layer_call_and_return_conditional_losses_102636652&
$embedding_15/StatefulPartitionedCall?
tf.cast_7/CastCast(tf.math.greater_equal_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_7/Cast?
tf.__operators__.add_14/AddV2AddV2-embedding_17/StatefulPartitionedCall:output:0-embedding_15/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_14/AddV2?
$embedding_16/StatefulPartitionedCallStatefulPartitionedCalltf.math.floormod_5/FloorMod:z:0embedding_16_10263810*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_16_layer_call_and_return_conditional_losses_102636892&
$embedding_16/StatefulPartitionedCall?
tf.__operators__.add_15/AddV2AddV2!tf.__operators__.add_14/AddV2:z:0-embedding_16/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_15/AddV2?
tf.expand_dims_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_5/ExpandDims/dim?
tf.expand_dims_5/ExpandDims
ExpandDimstf.cast_7/Cast:y:0(tf.expand_dims_5/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_5/ExpandDims?
tf.math.multiply_5/MulMul!tf.__operators__.add_15/AddV2:z:0$tf.expand_dims_5/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_5/Mul?
*tf.math.reduce_sum_5/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_5/Sum/reduction_indices?
tf.math.reduce_sum_5/SumSumtf.math.multiply_5/Mul:z:03tf.math.reduce_sum_5/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_5/Sum?
IdentityIdentity!tf.math.reduce_sum_5/Sum:output:0%^embedding_15/StatefulPartitionedCall%^embedding_16/StatefulPartitionedCall%^embedding_17/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2L
$embedding_15/StatefulPartitionedCall$embedding_15/StatefulPartitionedCall2L
$embedding_16/StatefulPartitionedCall$embedding_16/StatefulPartitionedCall2L
$embedding_17/StatefulPartitionedCall$embedding_17/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
u
/__inference_embedding_12_layer_call_fn_10265633

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_12_layer_call_and_return_conditional_losses_102634392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
F__inference_dense_23_layer_call_and_return_conditional_losses_10265496

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?#
!__inference__traced_save_10265988
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

identity_1??MergeV2Checkpoints?
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
Const_1?
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
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?-
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*?,
value?,B?,SB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*?
value?B?SB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?!
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_18_kernel_read_readvariableop(savev2_dense_18_bias_read_readvariableop*savev2_dense_19_kernel_read_readvariableop(savev2_dense_19_bias_read_readvariableop*savev2_dense_21_kernel_read_readvariableop(savev2_dense_21_bias_read_readvariableop*savev2_dense_20_kernel_read_readvariableop(savev2_dense_20_bias_read_readvariableop*savev2_dense_22_kernel_read_readvariableop(savev2_dense_22_bias_read_readvariableop*savev2_dense_23_kernel_read_readvariableop(savev2_dense_23_bias_read_readvariableop*savev2_dense_24_kernel_read_readvariableop(savev2_dense_24_bias_read_readvariableop*savev2_dense_25_kernel_read_readvariableop(savev2_dense_25_bias_read_readvariableop*savev2_dense_26_kernel_read_readvariableop(savev2_dense_26_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop2savev2_embedding_14_embeddings_read_readvariableop2savev2_embedding_12_embeddings_read_readvariableop2savev2_embedding_13_embeddings_read_readvariableop2savev2_embedding_17_embeddings_read_readvariableop2savev2_embedding_15_embeddings_read_readvariableop2savev2_embedding_16_embeddings_read_readvariableop;savev2_normalize_2_normalization_2_mean_read_readvariableop?savev2_normalize_2_normalization_2_variance_read_readvariableop<savev2_normalize_2_normalization_2_count_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_18_kernel_m_read_readvariableop/savev2_adam_dense_18_bias_m_read_readvariableop1savev2_adam_dense_19_kernel_m_read_readvariableop/savev2_adam_dense_19_bias_m_read_readvariableop1savev2_adam_dense_21_kernel_m_read_readvariableop/savev2_adam_dense_21_bias_m_read_readvariableop1savev2_adam_dense_20_kernel_m_read_readvariableop/savev2_adam_dense_20_bias_m_read_readvariableop1savev2_adam_dense_22_kernel_m_read_readvariableop/savev2_adam_dense_22_bias_m_read_readvariableop1savev2_adam_dense_23_kernel_m_read_readvariableop/savev2_adam_dense_23_bias_m_read_readvariableop1savev2_adam_dense_24_kernel_m_read_readvariableop/savev2_adam_dense_24_bias_m_read_readvariableop1savev2_adam_dense_25_kernel_m_read_readvariableop/savev2_adam_dense_25_bias_m_read_readvariableop1savev2_adam_dense_26_kernel_m_read_readvariableop/savev2_adam_dense_26_bias_m_read_readvariableop9savev2_adam_embedding_14_embeddings_m_read_readvariableop9savev2_adam_embedding_12_embeddings_m_read_readvariableop9savev2_adam_embedding_13_embeddings_m_read_readvariableop9savev2_adam_embedding_17_embeddings_m_read_readvariableop9savev2_adam_embedding_15_embeddings_m_read_readvariableop9savev2_adam_embedding_16_embeddings_m_read_readvariableop1savev2_adam_dense_18_kernel_v_read_readvariableop/savev2_adam_dense_18_bias_v_read_readvariableop1savev2_adam_dense_19_kernel_v_read_readvariableop/savev2_adam_dense_19_bias_v_read_readvariableop1savev2_adam_dense_21_kernel_v_read_readvariableop/savev2_adam_dense_21_bias_v_read_readvariableop1savev2_adam_dense_20_kernel_v_read_readvariableop/savev2_adam_dense_20_bias_v_read_readvariableop1savev2_adam_dense_22_kernel_v_read_readvariableop/savev2_adam_dense_22_bias_v_read_readvariableop1savev2_adam_dense_23_kernel_v_read_readvariableop/savev2_adam_dense_23_bias_v_read_readvariableop1savev2_adam_dense_24_kernel_v_read_readvariableop/savev2_adam_dense_24_bias_v_read_readvariableop1savev2_adam_dense_25_kernel_v_read_readvariableop/savev2_adam_dense_25_bias_v_read_readvariableop1savev2_adam_dense_26_kernel_v_read_readvariableop/savev2_adam_dense_26_bias_v_read_readvariableop9savev2_adam_embedding_14_embeddings_v_read_readvariableop9savev2_adam_embedding_12_embeddings_v_read_readvariableop9savev2_adam_embedding_13_embeddings_v_read_readvariableop9savev2_adam_embedding_17_embeddings_v_read_readvariableop9savev2_adam_embedding_15_embeddings_v_read_readvariableop9savev2_adam_embedding_16_embeddings_v_read_readvariableopsavev2_const_5"/device:CPU:0*
_output_shapes
 *a
dtypesW
U2S		2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :
??:?:
??:?:	?:?:
??:?:
??:?:
??:?:
??:?:
??:?:	?:: : : : : :	4?:	?:	?:	4?:	?:	?:?:?: : : :
??:?:
??:?:	?:?:
??:?:
??:?:
??:?:
??:?:
??:?:	?::	4?:	?:	?:	4?:	?:	?:
??:?:
??:?:	?:?:
??:?:
??:?:
??:?:
??:?:
??:?:	?::	4?:	?:	?:	4?:	?:	?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&	"
 
_output_shapes
:
??:!


_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 
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
:	4?:%!

_output_shapes
:	?:%!

_output_shapes
:	?:%!

_output_shapes
:	4?:%!

_output_shapes
:	?:%!

_output_shapes
:	?:!

_output_shapes	
:?:!

_output_shapes	
:?: 
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
??:!$

_output_shapes	
:?:&%"
 
_output_shapes
:
??:!&

_output_shapes	
:?:%'!

_output_shapes
:	?:!(

_output_shapes	
:?:&)"
 
_output_shapes
:
??:!*

_output_shapes	
:?:&+"
 
_output_shapes
:
??:!,

_output_shapes	
:?:&-"
 
_output_shapes
:
??:!.

_output_shapes	
:?:&/"
 
_output_shapes
:
??:!0

_output_shapes	
:?:&1"
 
_output_shapes
:
??:!2

_output_shapes	
:?:%3!

_output_shapes
:	?: 4

_output_shapes
::%5!

_output_shapes
:	4?:%6!

_output_shapes
:	?:%7!

_output_shapes
:	?:%8!

_output_shapes
:	4?:%9!

_output_shapes
:	?:%:!

_output_shapes
:	?:&;"
 
_output_shapes
:
??:!<

_output_shapes	
:?:&="
 
_output_shapes
:
??:!>

_output_shapes	
:?:%?!

_output_shapes
:	?:!@

_output_shapes	
:?:&A"
 
_output_shapes
:
??:!B

_output_shapes	
:?:&C"
 
_output_shapes
:
??:!D

_output_shapes	
:?:&E"
 
_output_shapes
:
??:!F

_output_shapes	
:?:&G"
 
_output_shapes
:
??:!H

_output_shapes	
:?:&I"
 
_output_shapes
:
??:!J

_output_shapes	
:?:%K!

_output_shapes
:	?: L

_output_shapes
::%M!

_output_shapes
:	4?:%N!

_output_shapes
:	?:%O!

_output_shapes
:	?:%P!

_output_shapes
:	4?:%Q!

_output_shapes
:	?:%R!

_output_shapes
:	?:S

_output_shapes
: 
?	
?
J__inference_embedding_16_layer_call_and_return_conditional_losses_10265705

inputs
embedding_lookup_10265699
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_10265699Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*,
_class"
 loc:@embedding_lookup/10265699*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@embedding_lookup/10265699*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
F__inference_dense_20_layer_call_and_return_conditional_losses_10265458

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
1__inference_custom_model_2_layer_call_fn_10264611

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
identity??StatefulPartitionedCall?
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
:?????????*<
_read_only_resource_inputs
	
 !*2
config_proto" 

CPU

GPU2 *0J 8? *U
fPRN
L__inference_custom_model_2_layer_call_and_return_conditional_losses_102645462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
: : :::: :::: : ::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_namecards0:OK
'
_output_shapes
:?????????
 
_user_specified_namecards1:MI
'
_output_shapes
:?????????


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
?
c
G__inference_flatten_5_layer_call_and_return_conditional_losses_10265656

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
u
/__inference_embedding_17_layer_call_fn_10265678

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_17_layer_call_and_return_conditional_losses_102636432
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
*__inference_model_5_layer_call_fn_10265375

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_model_5_layer_call_and_return_conditional_losses_102637752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
+__inference_dense_25_layer_call_fn_10265543

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_25_layer_call_and_return_conditional_losses_102641182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
&__inference_signature_wrapper_10264690
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
identity??StatefulPartitionedCall?
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
:?????????*<
_read_only_resource_inputs
	
 !*2
config_proto" 

CPU

GPU2 *0J 8? *,
f'R%
#__inference__wrapped_model_102633792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????
:?????????:?????????: : :::: :::: : ::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:M I
'
_output_shapes
:?????????


_user_specified_namebets:OK
'
_output_shapes
:?????????
 
_user_specified_namecards0:OK
'
_output_shapes
:?????????
 
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
?X
?

L__inference_custom_model_2_layer_call_and_return_conditional_losses_10264288

cards0

cards1
bets*
&tf_math_greater_equal_8_greaterequal_y
model_4_10264203
model_4_10264205
model_4_10264207
model_4_10264209
model_5_10264212
model_5_10264214
model_5_10264216
model_5_10264218.
*tf_clip_by_value_8_clip_by_value_minimum_y&
"tf_clip_by_value_8_clip_by_value_y
dense_18_10264230
dense_18_10264232
dense_21_10264235
dense_21_10264237
dense_19_10264240
dense_19_10264242
dense_20_10264245
dense_20_10264247
dense_22_10264250
dense_22_10264252
dense_23_10264257
dense_23_10264259
dense_24_10264263
dense_24_10264265
dense_25_10264270
dense_25_10264272
normalize_2_10264277
normalize_2_10264279
dense_26_10264282
dense_26_10264284
identity?? dense_18/StatefulPartitionedCall? dense_19/StatefulPartitionedCall? dense_20/StatefulPartitionedCall? dense_21/StatefulPartitionedCall? dense_22/StatefulPartitionedCall? dense_23/StatefulPartitionedCall? dense_24/StatefulPartitionedCall? dense_25/StatefulPartitionedCall? dense_26/StatefulPartitionedCall?model_4/StatefulPartitionedCall?model_5/StatefulPartitionedCall?#normalize_2/StatefulPartitionedCall?
$tf.math.greater_equal_8/GreaterEqualGreaterEqualbets&tf_math_greater_equal_8_greaterequal_y*
T0*'
_output_shapes
:?????????
2&
$tf.math.greater_equal_8/GreaterEqual?
model_4/StatefulPartitionedCallStatefulPartitionedCallcards0model_4_10264203model_4_10264205model_4_10264207model_4_10264209*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_model_4_layer_call_and_return_conditional_losses_102635942!
model_4/StatefulPartitionedCall?
model_5/StatefulPartitionedCallStatefulPartitionedCallcards1model_5_10264212model_5_10264214model_5_10264216model_5_10264218*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_model_5_layer_call_and_return_conditional_losses_102638202!
model_5/StatefulPartitionedCall?
(tf.clip_by_value_8/clip_by_value/MinimumMinimumbets*tf_clip_by_value_8_clip_by_value_minimum_y*
T0*'
_output_shapes
:?????????
2*
(tf.clip_by_value_8/clip_by_value/Minimum?
 tf.clip_by_value_8/clip_by_valueMaximum,tf.clip_by_value_8/clip_by_value/Minimum:z:0"tf_clip_by_value_8_clip_by_value_y*
T0*'
_output_shapes
:?????????
2"
 tf.clip_by_value_8/clip_by_value?
tf.cast_8/CastCast(tf.math.greater_equal_8/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
2
tf.cast_8/Castt
tf.concat_6/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_6/concat/axis?
tf.concat_6/concatConcatV2(model_4/StatefulPartitionedCall:output:0(model_5/StatefulPartitionedCall:output:0 tf.concat_6/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_6/concat}
tf.concat_7/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_7/concat/axis?
tf.concat_7/concatConcatV2$tf.clip_by_value_8/clip_by_value:z:0tf.cast_8/Cast:y:0 tf.concat_7/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
tf.concat_7/concat?
 dense_18/StatefulPartitionedCallStatefulPartitionedCalltf.concat_6/concat:output:0dense_18_10264230dense_18_10264232*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_18_layer_call_and_return_conditional_losses_102639292"
 dense_18/StatefulPartitionedCall?
 dense_21/StatefulPartitionedCallStatefulPartitionedCalltf.concat_7/concat:output:0dense_21_10264235dense_21_10264237*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_21_layer_call_and_return_conditional_losses_102639552"
 dense_21/StatefulPartitionedCall?
 dense_19/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0dense_19_10264240dense_19_10264242*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_19_layer_call_and_return_conditional_losses_102639822"
 dense_19/StatefulPartitionedCall?
 dense_20/StatefulPartitionedCallStatefulPartitionedCall)dense_19/StatefulPartitionedCall:output:0dense_20_10264245dense_20_10264247*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_20_layer_call_and_return_conditional_losses_102640092"
 dense_20/StatefulPartitionedCall?
 dense_22/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0dense_22_10264250dense_22_10264252*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_22_layer_call_and_return_conditional_losses_102640352"
 dense_22/StatefulPartitionedCall}
tf.concat_8/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_8/concat/axis?
tf.concat_8/concatConcatV2)dense_20/StatefulPartitionedCall:output:0)dense_22/StatefulPartitionedCall:output:0 tf.concat_8/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_8/concat?
 dense_23/StatefulPartitionedCallStatefulPartitionedCalltf.concat_8/concat:output:0dense_23_10264257dense_23_10264259*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_23_layer_call_and_return_conditional_losses_102640632"
 dense_23/StatefulPartitionedCall?
tf.nn.relu_6/ReluRelu)dense_23/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_6/Relu?
 dense_24/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_6/Relu:activations:0dense_24_10264263dense_24_10264265*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_24_layer_call_and_return_conditional_losses_102640902"
 dense_24/StatefulPartitionedCall?
tf.__operators__.add_16/AddV2AddV2)dense_24/StatefulPartitionedCall:output:0tf.nn.relu_6/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_16/AddV2?
tf.nn.relu_7/ReluRelu!tf.__operators__.add_16/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_7/Relu?
 dense_25/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_7/Relu:activations:0dense_25_10264270dense_25_10264272*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_25_layer_call_and_return_conditional_losses_102641182"
 dense_25/StatefulPartitionedCall?
tf.__operators__.add_17/AddV2AddV2)dense_25/StatefulPartitionedCall:output:0tf.nn.relu_7/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_17/AddV2?
tf.nn.relu_8/ReluRelu!tf.__operators__.add_17/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_8/Relu?
#normalize_2/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_8/Relu:activations:0normalize_2_10264277normalize_2_10264279*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_normalize_2_layer_call_and_return_conditional_losses_102641532%
#normalize_2/StatefulPartitionedCall?
 dense_26/StatefulPartitionedCallStatefulPartitionedCall,normalize_2/StatefulPartitionedCall:output:0dense_26_10264282dense_26_10264284*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_26_layer_call_and_return_conditional_losses_102641792"
 dense_26/StatefulPartitionedCall?
IdentityIdentity)dense_26/StatefulPartitionedCall:output:0!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall ^model_4/StatefulPartitionedCall ^model_5/StatefulPartitionedCall$^normalize_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
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
:?????????
 
_user_specified_namecards0:OK
'
_output_shapes
:?????????
 
_user_specified_namecards1:MI
'
_output_shapes
:?????????


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
?
u
/__inference_embedding_16_layer_call_fn_10265712

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_16_layer_call_and_return_conditional_losses_102636892
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
F__inference_dense_26_layer_call_and_return_conditional_losses_10264179

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
G__inference_flatten_5_layer_call_and_return_conditional_losses_10263615

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
F__inference_dense_23_layer_call_and_return_conditional_losses_10264063

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
I__inference_normalize_2_layer_call_and_return_conditional_losses_10264153
x3
/normalization_2_reshape_readvariableop_resource5
1normalization_2_reshape_1_readvariableop_resource
identity??&normalization_2/Reshape/ReadVariableOp?(normalization_2/Reshape_1/ReadVariableOp?
&normalization_2/Reshape/ReadVariableOpReadVariableOp/normalization_2_reshape_readvariableop_resource*
_output_shapes	
:?*
dtype02(
&normalization_2/Reshape/ReadVariableOp?
normalization_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_2/Reshape/shape?
normalization_2/ReshapeReshape.normalization_2/Reshape/ReadVariableOp:value:0&normalization_2/Reshape/shape:output:0*
T0*
_output_shapes
:	?2
normalization_2/Reshape?
(normalization_2/Reshape_1/ReadVariableOpReadVariableOp1normalization_2_reshape_1_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(normalization_2/Reshape_1/ReadVariableOp?
normalization_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_2/Reshape_1/shape?
normalization_2/Reshape_1Reshape0normalization_2/Reshape_1/ReadVariableOp:value:0(normalization_2/Reshape_1/shape:output:0*
T0*
_output_shapes
:	?2
normalization_2/Reshape_1?
normalization_2/subSubx normalization_2/Reshape:output:0*
T0*(
_output_shapes
:??????????2
normalization_2/sub?
normalization_2/SqrtSqrt"normalization_2/Reshape_1:output:0*
T0*
_output_shapes
:	?2
normalization_2/Sqrt{
normalization_2/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_2/Maximum/y?
normalization_2/MaximumMaximumnormalization_2/Sqrt:y:0"normalization_2/Maximum/y:output:0*
T0*
_output_shapes
:	?2
normalization_2/Maximum?
normalization_2/truedivRealDivnormalization_2/sub:z:0normalization_2/Maximum:z:0*
T0*(
_output_shapes
:??????????2
normalization_2/truediv?
IdentityIdentitynormalization_2/truediv:z:0'^normalization_2/Reshape/ReadVariableOp)^normalization_2/Reshape_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::2P
&normalization_2/Reshape/ReadVariableOp&normalization_2/Reshape/ReadVariableOp2T
(normalization_2/Reshape_1/ReadVariableOp(normalization_2/Reshape_1/ReadVariableOp:K G
(
_output_shapes
:??????????

_user_specified_namex
?	
?
F__inference_dense_22_layer_call_and_return_conditional_losses_10265477

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?,
$__inference__traced_restore_10266244
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
identity_83??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_9?-
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*?,
value?,B?,SB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*?
value?B?SB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*a
dtypesW
U2S		2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_dense_18_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_18_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_19_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_19_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_21_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_21_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_20_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_20_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_22_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_22_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp#assignvariableop_10_dense_23_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp!assignvariableop_11_dense_23_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_24_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_24_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp#assignvariableop_14_dense_25_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp!assignvariableop_15_dense_25_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp#assignvariableop_16_dense_26_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp!assignvariableop_17_dense_26_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_iterIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_adam_beta_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpassignvariableop_20_adam_beta_2Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpassignvariableop_21_adam_decayIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp&assignvariableop_22_adam_learning_rateIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp+assignvariableop_23_embedding_14_embeddingsIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp+assignvariableop_24_embedding_12_embeddingsIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp+assignvariableop_25_embedding_13_embeddingsIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp+assignvariableop_26_embedding_17_embeddingsIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp+assignvariableop_27_embedding_15_embeddingsIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp+assignvariableop_28_embedding_16_embeddingsIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp4assignvariableop_29_normalize_2_normalization_2_meanIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp8assignvariableop_30_normalize_2_normalization_2_varianceIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp5assignvariableop_31_normalize_2_normalization_2_countIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOpassignvariableop_32_totalIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOpassignvariableop_33_countIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp*assignvariableop_34_adam_dense_18_kernel_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp(assignvariableop_35_adam_dense_18_bias_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp*assignvariableop_36_adam_dense_19_kernel_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp(assignvariableop_37_adam_dense_19_bias_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp*assignvariableop_38_adam_dense_21_kernel_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp(assignvariableop_39_adam_dense_21_bias_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp*assignvariableop_40_adam_dense_20_kernel_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp(assignvariableop_41_adam_dense_20_bias_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp*assignvariableop_42_adam_dense_22_kernel_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp(assignvariableop_43_adam_dense_22_bias_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp*assignvariableop_44_adam_dense_23_kernel_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp(assignvariableop_45_adam_dense_23_bias_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp*assignvariableop_46_adam_dense_24_kernel_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp(assignvariableop_47_adam_dense_24_bias_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp*assignvariableop_48_adam_dense_25_kernel_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp(assignvariableop_49_adam_dense_25_bias_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp*assignvariableop_50_adam_dense_26_kernel_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp(assignvariableop_51_adam_dense_26_bias_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp2assignvariableop_52_adam_embedding_14_embeddings_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp2assignvariableop_53_adam_embedding_12_embeddings_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp2assignvariableop_54_adam_embedding_13_embeddings_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp2assignvariableop_55_adam_embedding_17_embeddings_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp2assignvariableop_56_adam_embedding_15_embeddings_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp2assignvariableop_57_adam_embedding_16_embeddings_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp*assignvariableop_58_adam_dense_18_kernel_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp(assignvariableop_59_adam_dense_18_bias_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp*assignvariableop_60_adam_dense_19_kernel_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp(assignvariableop_61_adam_dense_19_bias_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp*assignvariableop_62_adam_dense_21_kernel_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp(assignvariableop_63_adam_dense_21_bias_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp*assignvariableop_64_adam_dense_20_kernel_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp(assignvariableop_65_adam_dense_20_bias_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp*assignvariableop_66_adam_dense_22_kernel_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp(assignvariableop_67_adam_dense_22_bias_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp*assignvariableop_68_adam_dense_23_kernel_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp(assignvariableop_69_adam_dense_23_bias_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp*assignvariableop_70_adam_dense_24_kernel_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOp(assignvariableop_71_adam_dense_24_bias_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOp*assignvariableop_72_adam_dense_25_kernel_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOp(assignvariableop_73_adam_dense_25_bias_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOp*assignvariableop_74_adam_dense_26_kernel_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75?
AssignVariableOp_75AssignVariableOp(assignvariableop_75_adam_dense_26_bias_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76?
AssignVariableOp_76AssignVariableOp2assignvariableop_76_adam_embedding_14_embeddings_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77?
AssignVariableOp_77AssignVariableOp2assignvariableop_77_adam_embedding_12_embeddings_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78?
AssignVariableOp_78AssignVariableOp2assignvariableop_78_adam_embedding_13_embeddings_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79?
AssignVariableOp_79AssignVariableOp2assignvariableop_79_adam_embedding_17_embeddings_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80?
AssignVariableOp_80AssignVariableOp2assignvariableop_80_adam_embedding_15_embeddings_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81?
AssignVariableOp_81AssignVariableOp2assignvariableop_81_adam_embedding_16_embeddings_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_819
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_82Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_82?
Identity_83IdentityIdentity_82:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_83"#
identity_83Identity_83:output:0*?
_input_shapes?
?: ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
?8
?
E__inference_model_5_layer_call_and_return_conditional_losses_10265362

inputs*
&tf_math_greater_equal_7_greaterequal_y*
&embedding_17_embedding_lookup_10265336*
&embedding_15_embedding_lookup_10265342*
&embedding_16_embedding_lookup_10265350
identity??embedding_15/embedding_lookup?embedding_16/embedding_lookup?embedding_17/embedding_lookups
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_5/Const?
flatten_5/ReshapeReshapeinputsflatten_5/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_5/Reshape?
*tf.clip_by_value_7/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2,
*tf.clip_by_value_7/clip_by_value/Minimum/y?
(tf.clip_by_value_7/clip_by_value/MinimumMinimumflatten_5/Reshape:output:03tf.clip_by_value_7/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2*
(tf.clip_by_value_7/clip_by_value/Minimum?
"tf.clip_by_value_7/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_7/clip_by_value/y?
 tf.clip_by_value_7/clip_by_valueMaximum,tf.clip_by_value_7/clip_by_value/Minimum:z:0+tf.clip_by_value_7/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2"
 tf.clip_by_value_7/clip_by_value?
#tf.compat.v1.floor_div_5/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2%
#tf.compat.v1.floor_div_5/FloorDiv/y?
!tf.compat.v1.floor_div_5/FloorDivFloorDiv$tf.clip_by_value_7/clip_by_value:z:0,tf.compat.v1.floor_div_5/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.compat.v1.floor_div_5/FloorDiv?
$tf.math.greater_equal_7/GreaterEqualGreaterEqualflatten_5/Reshape:output:0&tf_math_greater_equal_7_greaterequal_y*
T0*'
_output_shapes
:?????????2&
$tf.math.greater_equal_7/GreaterEqual?
tf.math.floormod_5/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod_5/FloorMod/y?
tf.math.floormod_5/FloorModFloorMod$tf.clip_by_value_7/clip_by_value:z:0&tf.math.floormod_5/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_5/FloorMod?
embedding_17/CastCast$tf.clip_by_value_7/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_17/Cast?
embedding_17/embedding_lookupResourceGather&embedding_17_embedding_lookup_10265336embedding_17/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*9
_class/
-+loc:@embedding_17/embedding_lookup/10265336*,
_output_shapes
:??????????*
dtype02
embedding_17/embedding_lookup?
&embedding_17/embedding_lookup/IdentityIdentity&embedding_17/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*9
_class/
-+loc:@embedding_17/embedding_lookup/10265336*,
_output_shapes
:??????????2(
&embedding_17/embedding_lookup/Identity?
(embedding_17/embedding_lookup/Identity_1Identity/embedding_17/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_17/embedding_lookup/Identity_1?
embedding_15/CastCast%tf.compat.v1.floor_div_5/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_15/Cast?
embedding_15/embedding_lookupResourceGather&embedding_15_embedding_lookup_10265342embedding_15/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*9
_class/
-+loc:@embedding_15/embedding_lookup/10265342*,
_output_shapes
:??????????*
dtype02
embedding_15/embedding_lookup?
&embedding_15/embedding_lookup/IdentityIdentity&embedding_15/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*9
_class/
-+loc:@embedding_15/embedding_lookup/10265342*,
_output_shapes
:??????????2(
&embedding_15/embedding_lookup/Identity?
(embedding_15/embedding_lookup/Identity_1Identity/embedding_15/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_15/embedding_lookup/Identity_1?
tf.cast_7/CastCast(tf.math.greater_equal_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_7/Cast?
tf.__operators__.add_14/AddV2AddV21embedding_17/embedding_lookup/Identity_1:output:01embedding_15/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_14/AddV2?
embedding_16/CastCasttf.math.floormod_5/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_16/Cast?
embedding_16/embedding_lookupResourceGather&embedding_16_embedding_lookup_10265350embedding_16/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*9
_class/
-+loc:@embedding_16/embedding_lookup/10265350*,
_output_shapes
:??????????*
dtype02
embedding_16/embedding_lookup?
&embedding_16/embedding_lookup/IdentityIdentity&embedding_16/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*9
_class/
-+loc:@embedding_16/embedding_lookup/10265350*,
_output_shapes
:??????????2(
&embedding_16/embedding_lookup/Identity?
(embedding_16/embedding_lookup/Identity_1Identity/embedding_16/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_16/embedding_lookup/Identity_1?
tf.__operators__.add_15/AddV2AddV2!tf.__operators__.add_14/AddV2:z:01embedding_16/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_15/AddV2?
tf.expand_dims_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_5/ExpandDims/dim?
tf.expand_dims_5/ExpandDims
ExpandDimstf.cast_7/Cast:y:0(tf.expand_dims_5/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_5/ExpandDims?
tf.math.multiply_5/MulMul!tf.__operators__.add_15/AddV2:z:0$tf.expand_dims_5/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_5/Mul?
*tf.math.reduce_sum_5/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_5/Sum/reduction_indices?
tf.math.reduce_sum_5/SumSumtf.math.multiply_5/Mul:z:03tf.math.reduce_sum_5/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_5/Sum?
IdentityIdentity!tf.math.reduce_sum_5/Sum:output:0^embedding_15/embedding_lookup^embedding_16/embedding_lookup^embedding_17/embedding_lookup*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2>
embedding_15/embedding_lookupembedding_15/embedding_lookup2>
embedding_16/embedding_lookupembedding_16/embedding_lookup2>
embedding_17/embedding_lookupembedding_17/embedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?	
?
F__inference_dense_21_layer_call_and_return_conditional_losses_10263955

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
F__inference_dense_19_layer_call_and_return_conditional_losses_10265419

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
*__inference_model_4_layer_call_fn_10263560
input_5
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_model_4_layer_call_and_return_conditional_losses_102635492
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_5:

_output_shapes
: 
?	
?
F__inference_dense_25_layer_call_and_return_conditional_losses_10265534

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
F__inference_dense_26_layer_call_and_return_conditional_losses_10265579

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?-
?
E__inference_model_5_layer_call_and_return_conditional_losses_10263775

inputs*
&tf_math_greater_equal_7_greaterequal_y
embedding_17_10263757
embedding_15_10263760
embedding_16_10263765
identity??$embedding_15/StatefulPartitionedCall?$embedding_16/StatefulPartitionedCall?$embedding_17/StatefulPartitionedCall?
flatten_5/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_flatten_5_layer_call_and_return_conditional_losses_102636152
flatten_5/PartitionedCall?
*tf.clip_by_value_7/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2,
*tf.clip_by_value_7/clip_by_value/Minimum/y?
(tf.clip_by_value_7/clip_by_value/MinimumMinimum"flatten_5/PartitionedCall:output:03tf.clip_by_value_7/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2*
(tf.clip_by_value_7/clip_by_value/Minimum?
"tf.clip_by_value_7/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_7/clip_by_value/y?
 tf.clip_by_value_7/clip_by_valueMaximum,tf.clip_by_value_7/clip_by_value/Minimum:z:0+tf.clip_by_value_7/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2"
 tf.clip_by_value_7/clip_by_value?
#tf.compat.v1.floor_div_5/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2%
#tf.compat.v1.floor_div_5/FloorDiv/y?
!tf.compat.v1.floor_div_5/FloorDivFloorDiv$tf.clip_by_value_7/clip_by_value:z:0,tf.compat.v1.floor_div_5/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.compat.v1.floor_div_5/FloorDiv?
$tf.math.greater_equal_7/GreaterEqualGreaterEqual"flatten_5/PartitionedCall:output:0&tf_math_greater_equal_7_greaterequal_y*
T0*'
_output_shapes
:?????????2&
$tf.math.greater_equal_7/GreaterEqual?
tf.math.floormod_5/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod_5/FloorMod/y?
tf.math.floormod_5/FloorModFloorMod$tf.clip_by_value_7/clip_by_value:z:0&tf.math.floormod_5/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_5/FloorMod?
$embedding_17/StatefulPartitionedCallStatefulPartitionedCall$tf.clip_by_value_7/clip_by_value:z:0embedding_17_10263757*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_17_layer_call_and_return_conditional_losses_102636432&
$embedding_17/StatefulPartitionedCall?
$embedding_15/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.floor_div_5/FloorDiv:z:0embedding_15_10263760*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_15_layer_call_and_return_conditional_losses_102636652&
$embedding_15/StatefulPartitionedCall?
tf.cast_7/CastCast(tf.math.greater_equal_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_7/Cast?
tf.__operators__.add_14/AddV2AddV2-embedding_17/StatefulPartitionedCall:output:0-embedding_15/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_14/AddV2?
$embedding_16/StatefulPartitionedCallStatefulPartitionedCalltf.math.floormod_5/FloorMod:z:0embedding_16_10263765*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_16_layer_call_and_return_conditional_losses_102636892&
$embedding_16/StatefulPartitionedCall?
tf.__operators__.add_15/AddV2AddV2!tf.__operators__.add_14/AddV2:z:0-embedding_16/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_15/AddV2?
tf.expand_dims_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_5/ExpandDims/dim?
tf.expand_dims_5/ExpandDims
ExpandDimstf.cast_7/Cast:y:0(tf.expand_dims_5/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_5/ExpandDims?
tf.math.multiply_5/MulMul!tf.__operators__.add_15/AddV2:z:0$tf.expand_dims_5/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_5/Mul?
*tf.math.reduce_sum_5/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_5/Sum/reduction_indices?
tf.math.reduce_sum_5/SumSumtf.math.multiply_5/Mul:z:03tf.math.reduce_sum_5/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_5/Sum?
IdentityIdentity!tf.math.reduce_sum_5/Sum:output:0%^embedding_15/StatefulPartitionedCall%^embedding_16/StatefulPartitionedCall%^embedding_17/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2L
$embedding_15/StatefulPartitionedCall$embedding_15/StatefulPartitionedCall2L
$embedding_16/StatefulPartitionedCall$embedding_16/StatefulPartitionedCall2L
$embedding_17/StatefulPartitionedCall$embedding_17/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?	
?
J__inference_embedding_14_layer_call_and_return_conditional_losses_10265609

inputs
embedding_lookup_10265603
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_10265603Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*,
_class"
 loc:@embedding_lookup/10265603*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@embedding_lookup/10265603*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?8
?
E__inference_model_5_layer_call_and_return_conditional_losses_10265320

inputs*
&tf_math_greater_equal_7_greaterequal_y*
&embedding_17_embedding_lookup_10265294*
&embedding_15_embedding_lookup_10265300*
&embedding_16_embedding_lookup_10265308
identity??embedding_15/embedding_lookup?embedding_16/embedding_lookup?embedding_17/embedding_lookups
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_5/Const?
flatten_5/ReshapeReshapeinputsflatten_5/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_5/Reshape?
*tf.clip_by_value_7/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2,
*tf.clip_by_value_7/clip_by_value/Minimum/y?
(tf.clip_by_value_7/clip_by_value/MinimumMinimumflatten_5/Reshape:output:03tf.clip_by_value_7/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2*
(tf.clip_by_value_7/clip_by_value/Minimum?
"tf.clip_by_value_7/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_7/clip_by_value/y?
 tf.clip_by_value_7/clip_by_valueMaximum,tf.clip_by_value_7/clip_by_value/Minimum:z:0+tf.clip_by_value_7/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2"
 tf.clip_by_value_7/clip_by_value?
#tf.compat.v1.floor_div_5/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2%
#tf.compat.v1.floor_div_5/FloorDiv/y?
!tf.compat.v1.floor_div_5/FloorDivFloorDiv$tf.clip_by_value_7/clip_by_value:z:0,tf.compat.v1.floor_div_5/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.compat.v1.floor_div_5/FloorDiv?
$tf.math.greater_equal_7/GreaterEqualGreaterEqualflatten_5/Reshape:output:0&tf_math_greater_equal_7_greaterequal_y*
T0*'
_output_shapes
:?????????2&
$tf.math.greater_equal_7/GreaterEqual?
tf.math.floormod_5/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod_5/FloorMod/y?
tf.math.floormod_5/FloorModFloorMod$tf.clip_by_value_7/clip_by_value:z:0&tf.math.floormod_5/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_5/FloorMod?
embedding_17/CastCast$tf.clip_by_value_7/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_17/Cast?
embedding_17/embedding_lookupResourceGather&embedding_17_embedding_lookup_10265294embedding_17/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*9
_class/
-+loc:@embedding_17/embedding_lookup/10265294*,
_output_shapes
:??????????*
dtype02
embedding_17/embedding_lookup?
&embedding_17/embedding_lookup/IdentityIdentity&embedding_17/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*9
_class/
-+loc:@embedding_17/embedding_lookup/10265294*,
_output_shapes
:??????????2(
&embedding_17/embedding_lookup/Identity?
(embedding_17/embedding_lookup/Identity_1Identity/embedding_17/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_17/embedding_lookup/Identity_1?
embedding_15/CastCast%tf.compat.v1.floor_div_5/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_15/Cast?
embedding_15/embedding_lookupResourceGather&embedding_15_embedding_lookup_10265300embedding_15/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*9
_class/
-+loc:@embedding_15/embedding_lookup/10265300*,
_output_shapes
:??????????*
dtype02
embedding_15/embedding_lookup?
&embedding_15/embedding_lookup/IdentityIdentity&embedding_15/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*9
_class/
-+loc:@embedding_15/embedding_lookup/10265300*,
_output_shapes
:??????????2(
&embedding_15/embedding_lookup/Identity?
(embedding_15/embedding_lookup/Identity_1Identity/embedding_15/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_15/embedding_lookup/Identity_1?
tf.cast_7/CastCast(tf.math.greater_equal_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_7/Cast?
tf.__operators__.add_14/AddV2AddV21embedding_17/embedding_lookup/Identity_1:output:01embedding_15/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_14/AddV2?
embedding_16/CastCasttf.math.floormod_5/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_16/Cast?
embedding_16/embedding_lookupResourceGather&embedding_16_embedding_lookup_10265308embedding_16/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*9
_class/
-+loc:@embedding_16/embedding_lookup/10265308*,
_output_shapes
:??????????*
dtype02
embedding_16/embedding_lookup?
&embedding_16/embedding_lookup/IdentityIdentity&embedding_16/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*9
_class/
-+loc:@embedding_16/embedding_lookup/10265308*,
_output_shapes
:??????????2(
&embedding_16/embedding_lookup/Identity?
(embedding_16/embedding_lookup/Identity_1Identity/embedding_16/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_16/embedding_lookup/Identity_1?
tf.__operators__.add_15/AddV2AddV2!tf.__operators__.add_14/AddV2:z:01embedding_16/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_15/AddV2?
tf.expand_dims_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_5/ExpandDims/dim?
tf.expand_dims_5/ExpandDims
ExpandDimstf.cast_7/Cast:y:0(tf.expand_dims_5/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_5/ExpandDims?
tf.math.multiply_5/MulMul!tf.__operators__.add_15/AddV2:z:0$tf.expand_dims_5/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_5/Mul?
*tf.math.reduce_sum_5/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_5/Sum/reduction_indices?
tf.math.reduce_sum_5/SumSumtf.math.multiply_5/Mul:z:03tf.math.reduce_sum_5/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_5/Sum?
IdentityIdentity!tf.math.reduce_sum_5/Sum:output:0^embedding_15/embedding_lookup^embedding_16/embedding_lookup^embedding_17/embedding_lookup*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2>
embedding_15/embedding_lookupembedding_15/embedding_lookup2>
embedding_16/embedding_lookupembedding_16/embedding_lookup2>
embedding_17/embedding_lookupembedding_17/embedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
*__inference_model_5_layer_call_fn_10265388

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_model_5_layer_call_and_return_conditional_losses_102638202
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?X
?

L__inference_custom_model_2_layer_call_and_return_conditional_losses_10264546

inputs
inputs_1
inputs_2*
&tf_math_greater_equal_8_greaterequal_y
model_4_10264461
model_4_10264463
model_4_10264465
model_4_10264467
model_5_10264470
model_5_10264472
model_5_10264474
model_5_10264476.
*tf_clip_by_value_8_clip_by_value_minimum_y&
"tf_clip_by_value_8_clip_by_value_y
dense_18_10264488
dense_18_10264490
dense_21_10264493
dense_21_10264495
dense_19_10264498
dense_19_10264500
dense_20_10264503
dense_20_10264505
dense_22_10264508
dense_22_10264510
dense_23_10264515
dense_23_10264517
dense_24_10264521
dense_24_10264523
dense_25_10264528
dense_25_10264530
normalize_2_10264535
normalize_2_10264537
dense_26_10264540
dense_26_10264542
identity?? dense_18/StatefulPartitionedCall? dense_19/StatefulPartitionedCall? dense_20/StatefulPartitionedCall? dense_21/StatefulPartitionedCall? dense_22/StatefulPartitionedCall? dense_23/StatefulPartitionedCall? dense_24/StatefulPartitionedCall? dense_25/StatefulPartitionedCall? dense_26/StatefulPartitionedCall?model_4/StatefulPartitionedCall?model_5/StatefulPartitionedCall?#normalize_2/StatefulPartitionedCall?
$tf.math.greater_equal_8/GreaterEqualGreaterEqualinputs_2&tf_math_greater_equal_8_greaterequal_y*
T0*'
_output_shapes
:?????????
2&
$tf.math.greater_equal_8/GreaterEqual?
model_4/StatefulPartitionedCallStatefulPartitionedCallinputsmodel_4_10264461model_4_10264463model_4_10264465model_4_10264467*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_model_4_layer_call_and_return_conditional_losses_102635942!
model_4/StatefulPartitionedCall?
model_5/StatefulPartitionedCallStatefulPartitionedCallinputs_1model_5_10264470model_5_10264472model_5_10264474model_5_10264476*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_model_5_layer_call_and_return_conditional_losses_102638202!
model_5/StatefulPartitionedCall?
(tf.clip_by_value_8/clip_by_value/MinimumMinimuminputs_2*tf_clip_by_value_8_clip_by_value_minimum_y*
T0*'
_output_shapes
:?????????
2*
(tf.clip_by_value_8/clip_by_value/Minimum?
 tf.clip_by_value_8/clip_by_valueMaximum,tf.clip_by_value_8/clip_by_value/Minimum:z:0"tf_clip_by_value_8_clip_by_value_y*
T0*'
_output_shapes
:?????????
2"
 tf.clip_by_value_8/clip_by_value?
tf.cast_8/CastCast(tf.math.greater_equal_8/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
2
tf.cast_8/Castt
tf.concat_6/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_6/concat/axis?
tf.concat_6/concatConcatV2(model_4/StatefulPartitionedCall:output:0(model_5/StatefulPartitionedCall:output:0 tf.concat_6/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_6/concat}
tf.concat_7/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_7/concat/axis?
tf.concat_7/concatConcatV2$tf.clip_by_value_8/clip_by_value:z:0tf.cast_8/Cast:y:0 tf.concat_7/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
tf.concat_7/concat?
 dense_18/StatefulPartitionedCallStatefulPartitionedCalltf.concat_6/concat:output:0dense_18_10264488dense_18_10264490*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_18_layer_call_and_return_conditional_losses_102639292"
 dense_18/StatefulPartitionedCall?
 dense_21/StatefulPartitionedCallStatefulPartitionedCalltf.concat_7/concat:output:0dense_21_10264493dense_21_10264495*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_21_layer_call_and_return_conditional_losses_102639552"
 dense_21/StatefulPartitionedCall?
 dense_19/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0dense_19_10264498dense_19_10264500*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_19_layer_call_and_return_conditional_losses_102639822"
 dense_19/StatefulPartitionedCall?
 dense_20/StatefulPartitionedCallStatefulPartitionedCall)dense_19/StatefulPartitionedCall:output:0dense_20_10264503dense_20_10264505*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_20_layer_call_and_return_conditional_losses_102640092"
 dense_20/StatefulPartitionedCall?
 dense_22/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0dense_22_10264508dense_22_10264510*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_22_layer_call_and_return_conditional_losses_102640352"
 dense_22/StatefulPartitionedCall}
tf.concat_8/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_8/concat/axis?
tf.concat_8/concatConcatV2)dense_20/StatefulPartitionedCall:output:0)dense_22/StatefulPartitionedCall:output:0 tf.concat_8/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_8/concat?
 dense_23/StatefulPartitionedCallStatefulPartitionedCalltf.concat_8/concat:output:0dense_23_10264515dense_23_10264517*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_23_layer_call_and_return_conditional_losses_102640632"
 dense_23/StatefulPartitionedCall?
tf.nn.relu_6/ReluRelu)dense_23/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_6/Relu?
 dense_24/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_6/Relu:activations:0dense_24_10264521dense_24_10264523*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_24_layer_call_and_return_conditional_losses_102640902"
 dense_24/StatefulPartitionedCall?
tf.__operators__.add_16/AddV2AddV2)dense_24/StatefulPartitionedCall:output:0tf.nn.relu_6/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_16/AddV2?
tf.nn.relu_7/ReluRelu!tf.__operators__.add_16/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_7/Relu?
 dense_25/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_7/Relu:activations:0dense_25_10264528dense_25_10264530*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_25_layer_call_and_return_conditional_losses_102641182"
 dense_25/StatefulPartitionedCall?
tf.__operators__.add_17/AddV2AddV2)dense_25/StatefulPartitionedCall:output:0tf.nn.relu_7/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_17/AddV2?
tf.nn.relu_8/ReluRelu!tf.__operators__.add_17/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_8/Relu?
#normalize_2/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_8/Relu:activations:0normalize_2_10264535normalize_2_10264537*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_normalize_2_layer_call_and_return_conditional_losses_102641532%
#normalize_2/StatefulPartitionedCall?
 dense_26/StatefulPartitionedCallStatefulPartitionedCall,normalize_2/StatefulPartitionedCall:output:0dense_26_10264540dense_26_10264542*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_26_layer_call_and_return_conditional_losses_102641792"
 dense_26/StatefulPartitionedCall?
IdentityIdentity)dense_26/StatefulPartitionedCall:output:0!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall ^model_4/StatefulPartitionedCall ^model_5/StatefulPartitionedCall$^normalize_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
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
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????

 
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
?
?
1__inference_custom_model_2_layer_call_fn_10265099

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
identity??StatefulPartitionedCall?
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
:?????????*<
_read_only_resource_inputs
	
 !*2
config_proto" 

CPU

GPU2 *0J 8? *U
fPRN
L__inference_custom_model_2_layer_call_and_return_conditional_losses_102643852
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
: : :::: :::: : ::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:?????????
$
_user_specified_name
inputs/0/0:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/0/1:QM
'
_output_shapes
:?????????

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
?	
?
J__inference_embedding_12_layer_call_and_return_conditional_losses_10263439

inputs
embedding_lookup_10263433
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_10263433Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*,
_class"
 loc:@embedding_lookup/10263433*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@embedding_lookup/10263433*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
H
,__inference_flatten_5_layer_call_fn_10265661

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_flatten_5_layer_call_and_return_conditional_losses_102636152
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?-
?
E__inference_model_4_layer_call_and_return_conditional_losses_10263594

inputs*
&tf_math_greater_equal_6_greaterequal_y
embedding_14_10263576
embedding_12_10263579
embedding_13_10263584
identity??$embedding_12/StatefulPartitionedCall?$embedding_13/StatefulPartitionedCall?$embedding_14/StatefulPartitionedCall?
flatten_4/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_flatten_4_layer_call_and_return_conditional_losses_102633892
flatten_4/PartitionedCall?
*tf.clip_by_value_6/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2,
*tf.clip_by_value_6/clip_by_value/Minimum/y?
(tf.clip_by_value_6/clip_by_value/MinimumMinimum"flatten_4/PartitionedCall:output:03tf.clip_by_value_6/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2*
(tf.clip_by_value_6/clip_by_value/Minimum?
"tf.clip_by_value_6/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_6/clip_by_value/y?
 tf.clip_by_value_6/clip_by_valueMaximum,tf.clip_by_value_6/clip_by_value/Minimum:z:0+tf.clip_by_value_6/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2"
 tf.clip_by_value_6/clip_by_value?
#tf.compat.v1.floor_div_4/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2%
#tf.compat.v1.floor_div_4/FloorDiv/y?
!tf.compat.v1.floor_div_4/FloorDivFloorDiv$tf.clip_by_value_6/clip_by_value:z:0,tf.compat.v1.floor_div_4/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.compat.v1.floor_div_4/FloorDiv?
$tf.math.greater_equal_6/GreaterEqualGreaterEqual"flatten_4/PartitionedCall:output:0&tf_math_greater_equal_6_greaterequal_y*
T0*'
_output_shapes
:?????????2&
$tf.math.greater_equal_6/GreaterEqual?
tf.math.floormod_4/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod_4/FloorMod/y?
tf.math.floormod_4/FloorModFloorMod$tf.clip_by_value_6/clip_by_value:z:0&tf.math.floormod_4/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_4/FloorMod?
$embedding_14/StatefulPartitionedCallStatefulPartitionedCall$tf.clip_by_value_6/clip_by_value:z:0embedding_14_10263576*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_14_layer_call_and_return_conditional_losses_102634172&
$embedding_14/StatefulPartitionedCall?
$embedding_12/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.floor_div_4/FloorDiv:z:0embedding_12_10263579*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_12_layer_call_and_return_conditional_losses_102634392&
$embedding_12/StatefulPartitionedCall?
tf.cast_6/CastCast(tf.math.greater_equal_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_6/Cast?
tf.__operators__.add_12/AddV2AddV2-embedding_14/StatefulPartitionedCall:output:0-embedding_12/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_12/AddV2?
$embedding_13/StatefulPartitionedCallStatefulPartitionedCalltf.math.floormod_4/FloorMod:z:0embedding_13_10263584*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_13_layer_call_and_return_conditional_losses_102634632&
$embedding_13/StatefulPartitionedCall?
tf.__operators__.add_13/AddV2AddV2!tf.__operators__.add_12/AddV2:z:0-embedding_13/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_13/AddV2?
tf.expand_dims_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_4/ExpandDims/dim?
tf.expand_dims_4/ExpandDims
ExpandDimstf.cast_6/Cast:y:0(tf.expand_dims_4/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_4/ExpandDims?
tf.math.multiply_4/MulMul!tf.__operators__.add_13/AddV2:z:0$tf.expand_dims_4/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_4/Mul?
*tf.math.reduce_sum_4/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_4/Sum/reduction_indices?
tf.math.reduce_sum_4/SumSumtf.math.multiply_4/Mul:z:03tf.math.reduce_sum_4/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_4/Sum?
IdentityIdentity!tf.math.reduce_sum_4/Sum:output:0%^embedding_12/StatefulPartitionedCall%^embedding_13/StatefulPartitionedCall%^embedding_14/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2L
$embedding_12/StatefulPartitionedCall$embedding_12/StatefulPartitionedCall2L
$embedding_13/StatefulPartitionedCall$embedding_13/StatefulPartitionedCall2L
$embedding_14/StatefulPartitionedCall$embedding_14/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?-
?
E__inference_model_4_layer_call_and_return_conditional_losses_10263549

inputs*
&tf_math_greater_equal_6_greaterequal_y
embedding_14_10263531
embedding_12_10263534
embedding_13_10263539
identity??$embedding_12/StatefulPartitionedCall?$embedding_13/StatefulPartitionedCall?$embedding_14/StatefulPartitionedCall?
flatten_4/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_flatten_4_layer_call_and_return_conditional_losses_102633892
flatten_4/PartitionedCall?
*tf.clip_by_value_6/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2,
*tf.clip_by_value_6/clip_by_value/Minimum/y?
(tf.clip_by_value_6/clip_by_value/MinimumMinimum"flatten_4/PartitionedCall:output:03tf.clip_by_value_6/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2*
(tf.clip_by_value_6/clip_by_value/Minimum?
"tf.clip_by_value_6/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_6/clip_by_value/y?
 tf.clip_by_value_6/clip_by_valueMaximum,tf.clip_by_value_6/clip_by_value/Minimum:z:0+tf.clip_by_value_6/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2"
 tf.clip_by_value_6/clip_by_value?
#tf.compat.v1.floor_div_4/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2%
#tf.compat.v1.floor_div_4/FloorDiv/y?
!tf.compat.v1.floor_div_4/FloorDivFloorDiv$tf.clip_by_value_6/clip_by_value:z:0,tf.compat.v1.floor_div_4/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.compat.v1.floor_div_4/FloorDiv?
$tf.math.greater_equal_6/GreaterEqualGreaterEqual"flatten_4/PartitionedCall:output:0&tf_math_greater_equal_6_greaterequal_y*
T0*'
_output_shapes
:?????????2&
$tf.math.greater_equal_6/GreaterEqual?
tf.math.floormod_4/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod_4/FloorMod/y?
tf.math.floormod_4/FloorModFloorMod$tf.clip_by_value_6/clip_by_value:z:0&tf.math.floormod_4/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_4/FloorMod?
$embedding_14/StatefulPartitionedCallStatefulPartitionedCall$tf.clip_by_value_6/clip_by_value:z:0embedding_14_10263531*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_14_layer_call_and_return_conditional_losses_102634172&
$embedding_14/StatefulPartitionedCall?
$embedding_12/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.floor_div_4/FloorDiv:z:0embedding_12_10263534*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_12_layer_call_and_return_conditional_losses_102634392&
$embedding_12/StatefulPartitionedCall?
tf.cast_6/CastCast(tf.math.greater_equal_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_6/Cast?
tf.__operators__.add_12/AddV2AddV2-embedding_14/StatefulPartitionedCall:output:0-embedding_12/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_12/AddV2?
$embedding_13/StatefulPartitionedCallStatefulPartitionedCalltf.math.floormod_4/FloorMod:z:0embedding_13_10263539*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_13_layer_call_and_return_conditional_losses_102634632&
$embedding_13/StatefulPartitionedCall?
tf.__operators__.add_13/AddV2AddV2!tf.__operators__.add_12/AddV2:z:0-embedding_13/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_13/AddV2?
tf.expand_dims_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_4/ExpandDims/dim?
tf.expand_dims_4/ExpandDims
ExpandDimstf.cast_6/Cast:y:0(tf.expand_dims_4/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_4/ExpandDims?
tf.math.multiply_4/MulMul!tf.__operators__.add_13/AddV2:z:0$tf.expand_dims_4/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_4/Mul?
*tf.math.reduce_sum_4/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_4/Sum/reduction_indices?
tf.math.reduce_sum_4/SumSumtf.math.multiply_4/Mul:z:03tf.math.reduce_sum_4/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_4/Sum?
IdentityIdentity!tf.math.reduce_sum_4/Sum:output:0%^embedding_12/StatefulPartitionedCall%^embedding_13/StatefulPartitionedCall%^embedding_14/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2L
$embedding_12/StatefulPartitionedCall$embedding_12/StatefulPartitionedCall2L
$embedding_13/StatefulPartitionedCall$embedding_13/StatefulPartitionedCall2L
$embedding_14/StatefulPartitionedCall$embedding_14/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?	
?
J__inference_embedding_15_layer_call_and_return_conditional_losses_10265688

inputs
embedding_lookup_10265682
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_10265682Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*,
_class"
 loc:@embedding_lookup/10265682*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@embedding_lookup/10265682*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?X
?

L__inference_custom_model_2_layer_call_and_return_conditional_losses_10264385

inputs
inputs_1
inputs_2*
&tf_math_greater_equal_8_greaterequal_y
model_4_10264300
model_4_10264302
model_4_10264304
model_4_10264306
model_5_10264309
model_5_10264311
model_5_10264313
model_5_10264315.
*tf_clip_by_value_8_clip_by_value_minimum_y&
"tf_clip_by_value_8_clip_by_value_y
dense_18_10264327
dense_18_10264329
dense_21_10264332
dense_21_10264334
dense_19_10264337
dense_19_10264339
dense_20_10264342
dense_20_10264344
dense_22_10264347
dense_22_10264349
dense_23_10264354
dense_23_10264356
dense_24_10264360
dense_24_10264362
dense_25_10264367
dense_25_10264369
normalize_2_10264374
normalize_2_10264376
dense_26_10264379
dense_26_10264381
identity?? dense_18/StatefulPartitionedCall? dense_19/StatefulPartitionedCall? dense_20/StatefulPartitionedCall? dense_21/StatefulPartitionedCall? dense_22/StatefulPartitionedCall? dense_23/StatefulPartitionedCall? dense_24/StatefulPartitionedCall? dense_25/StatefulPartitionedCall? dense_26/StatefulPartitionedCall?model_4/StatefulPartitionedCall?model_5/StatefulPartitionedCall?#normalize_2/StatefulPartitionedCall?
$tf.math.greater_equal_8/GreaterEqualGreaterEqualinputs_2&tf_math_greater_equal_8_greaterequal_y*
T0*'
_output_shapes
:?????????
2&
$tf.math.greater_equal_8/GreaterEqual?
model_4/StatefulPartitionedCallStatefulPartitionedCallinputsmodel_4_10264300model_4_10264302model_4_10264304model_4_10264306*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_model_4_layer_call_and_return_conditional_losses_102635492!
model_4/StatefulPartitionedCall?
model_5/StatefulPartitionedCallStatefulPartitionedCallinputs_1model_5_10264309model_5_10264311model_5_10264313model_5_10264315*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_model_5_layer_call_and_return_conditional_losses_102637752!
model_5/StatefulPartitionedCall?
(tf.clip_by_value_8/clip_by_value/MinimumMinimuminputs_2*tf_clip_by_value_8_clip_by_value_minimum_y*
T0*'
_output_shapes
:?????????
2*
(tf.clip_by_value_8/clip_by_value/Minimum?
 tf.clip_by_value_8/clip_by_valueMaximum,tf.clip_by_value_8/clip_by_value/Minimum:z:0"tf_clip_by_value_8_clip_by_value_y*
T0*'
_output_shapes
:?????????
2"
 tf.clip_by_value_8/clip_by_value?
tf.cast_8/CastCast(tf.math.greater_equal_8/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
2
tf.cast_8/Castt
tf.concat_6/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_6/concat/axis?
tf.concat_6/concatConcatV2(model_4/StatefulPartitionedCall:output:0(model_5/StatefulPartitionedCall:output:0 tf.concat_6/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_6/concat}
tf.concat_7/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_7/concat/axis?
tf.concat_7/concatConcatV2$tf.clip_by_value_8/clip_by_value:z:0tf.cast_8/Cast:y:0 tf.concat_7/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
tf.concat_7/concat?
 dense_18/StatefulPartitionedCallStatefulPartitionedCalltf.concat_6/concat:output:0dense_18_10264327dense_18_10264329*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_18_layer_call_and_return_conditional_losses_102639292"
 dense_18/StatefulPartitionedCall?
 dense_21/StatefulPartitionedCallStatefulPartitionedCalltf.concat_7/concat:output:0dense_21_10264332dense_21_10264334*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_21_layer_call_and_return_conditional_losses_102639552"
 dense_21/StatefulPartitionedCall?
 dense_19/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0dense_19_10264337dense_19_10264339*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_19_layer_call_and_return_conditional_losses_102639822"
 dense_19/StatefulPartitionedCall?
 dense_20/StatefulPartitionedCallStatefulPartitionedCall)dense_19/StatefulPartitionedCall:output:0dense_20_10264342dense_20_10264344*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_20_layer_call_and_return_conditional_losses_102640092"
 dense_20/StatefulPartitionedCall?
 dense_22/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0dense_22_10264347dense_22_10264349*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_22_layer_call_and_return_conditional_losses_102640352"
 dense_22/StatefulPartitionedCall}
tf.concat_8/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_8/concat/axis?
tf.concat_8/concatConcatV2)dense_20/StatefulPartitionedCall:output:0)dense_22/StatefulPartitionedCall:output:0 tf.concat_8/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_8/concat?
 dense_23/StatefulPartitionedCallStatefulPartitionedCalltf.concat_8/concat:output:0dense_23_10264354dense_23_10264356*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_23_layer_call_and_return_conditional_losses_102640632"
 dense_23/StatefulPartitionedCall?
tf.nn.relu_6/ReluRelu)dense_23/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_6/Relu?
 dense_24/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_6/Relu:activations:0dense_24_10264360dense_24_10264362*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_24_layer_call_and_return_conditional_losses_102640902"
 dense_24/StatefulPartitionedCall?
tf.__operators__.add_16/AddV2AddV2)dense_24/StatefulPartitionedCall:output:0tf.nn.relu_6/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_16/AddV2?
tf.nn.relu_7/ReluRelu!tf.__operators__.add_16/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_7/Relu?
 dense_25/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_7/Relu:activations:0dense_25_10264367dense_25_10264369*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_25_layer_call_and_return_conditional_losses_102641182"
 dense_25/StatefulPartitionedCall?
tf.__operators__.add_17/AddV2AddV2)dense_25/StatefulPartitionedCall:output:0tf.nn.relu_7/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_17/AddV2?
tf.nn.relu_8/ReluRelu!tf.__operators__.add_17/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_8/Relu?
#normalize_2/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_8/Relu:activations:0normalize_2_10264374normalize_2_10264376*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_normalize_2_layer_call_and_return_conditional_losses_102641532%
#normalize_2/StatefulPartitionedCall?
 dense_26/StatefulPartitionedCallStatefulPartitionedCall,normalize_2/StatefulPartitionedCall:output:0dense_26_10264379dense_26_10264381*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_26_layer_call_and_return_conditional_losses_102641792"
 dense_26/StatefulPartitionedCall?
IdentityIdentity)dense_26/StatefulPartitionedCall:output:0!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall ^model_4/StatefulPartitionedCall ^model_5/StatefulPartitionedCall$^normalize_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
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
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????

 
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
?	
?
F__inference_dense_25_layer_call_and_return_conditional_losses_10264118

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?-
?
E__inference_model_5_layer_call_and_return_conditional_losses_10263740
input_6*
&tf_math_greater_equal_7_greaterequal_y
embedding_17_10263722
embedding_15_10263725
embedding_16_10263730
identity??$embedding_15/StatefulPartitionedCall?$embedding_16/StatefulPartitionedCall?$embedding_17/StatefulPartitionedCall?
flatten_5/PartitionedCallPartitionedCallinput_6*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_flatten_5_layer_call_and_return_conditional_losses_102636152
flatten_5/PartitionedCall?
*tf.clip_by_value_7/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2,
*tf.clip_by_value_7/clip_by_value/Minimum/y?
(tf.clip_by_value_7/clip_by_value/MinimumMinimum"flatten_5/PartitionedCall:output:03tf.clip_by_value_7/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2*
(tf.clip_by_value_7/clip_by_value/Minimum?
"tf.clip_by_value_7/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_7/clip_by_value/y?
 tf.clip_by_value_7/clip_by_valueMaximum,tf.clip_by_value_7/clip_by_value/Minimum:z:0+tf.clip_by_value_7/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2"
 tf.clip_by_value_7/clip_by_value?
#tf.compat.v1.floor_div_5/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2%
#tf.compat.v1.floor_div_5/FloorDiv/y?
!tf.compat.v1.floor_div_5/FloorDivFloorDiv$tf.clip_by_value_7/clip_by_value:z:0,tf.compat.v1.floor_div_5/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.compat.v1.floor_div_5/FloorDiv?
$tf.math.greater_equal_7/GreaterEqualGreaterEqual"flatten_5/PartitionedCall:output:0&tf_math_greater_equal_7_greaterequal_y*
T0*'
_output_shapes
:?????????2&
$tf.math.greater_equal_7/GreaterEqual?
tf.math.floormod_5/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod_5/FloorMod/y?
tf.math.floormod_5/FloorModFloorMod$tf.clip_by_value_7/clip_by_value:z:0&tf.math.floormod_5/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_5/FloorMod?
$embedding_17/StatefulPartitionedCallStatefulPartitionedCall$tf.clip_by_value_7/clip_by_value:z:0embedding_17_10263722*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_17_layer_call_and_return_conditional_losses_102636432&
$embedding_17/StatefulPartitionedCall?
$embedding_15/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.floor_div_5/FloorDiv:z:0embedding_15_10263725*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_15_layer_call_and_return_conditional_losses_102636652&
$embedding_15/StatefulPartitionedCall?
tf.cast_7/CastCast(tf.math.greater_equal_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_7/Cast?
tf.__operators__.add_14/AddV2AddV2-embedding_17/StatefulPartitionedCall:output:0-embedding_15/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_14/AddV2?
$embedding_16/StatefulPartitionedCallStatefulPartitionedCalltf.math.floormod_5/FloorMod:z:0embedding_16_10263730*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_16_layer_call_and_return_conditional_losses_102636892&
$embedding_16/StatefulPartitionedCall?
tf.__operators__.add_15/AddV2AddV2!tf.__operators__.add_14/AddV2:z:0-embedding_16/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_15/AddV2?
tf.expand_dims_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_5/ExpandDims/dim?
tf.expand_dims_5/ExpandDims
ExpandDimstf.cast_7/Cast:y:0(tf.expand_dims_5/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_5/ExpandDims?
tf.math.multiply_5/MulMul!tf.__operators__.add_15/AddV2:z:0$tf.expand_dims_5/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_5/Mul?
*tf.math.reduce_sum_5/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_5/Sum/reduction_indices?
tf.math.reduce_sum_5/SumSumtf.math.multiply_5/Mul:z:03tf.math.reduce_sum_5/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_5/Sum?
IdentityIdentity!tf.math.reduce_sum_5/Sum:output:0%^embedding_15/StatefulPartitionedCall%^embedding_16/StatefulPartitionedCall%^embedding_17/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2L
$embedding_15/StatefulPartitionedCall$embedding_15/StatefulPartitionedCall2L
$embedding_16/StatefulPartitionedCall$embedding_16/StatefulPartitionedCall2L
$embedding_17/StatefulPartitionedCall$embedding_17/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_6:

_output_shapes
: 
?	
?
J__inference_embedding_12_layer_call_and_return_conditional_losses_10265626

inputs
embedding_lookup_10265620
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_10265620Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*,
_class"
 loc:@embedding_lookup/10265620*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@embedding_lookup/10265620*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
*__inference_model_5_layer_call_fn_10263786
input_6
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_6unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_model_5_layer_call_and_return_conditional_losses_102637752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_6:

_output_shapes
: 
?-
?
E__inference_model_4_layer_call_and_return_conditional_losses_10263482
input_5*
&tf_math_greater_equal_6_greaterequal_y
embedding_14_10263426
embedding_12_10263448
embedding_13_10263472
identity??$embedding_12/StatefulPartitionedCall?$embedding_13/StatefulPartitionedCall?$embedding_14/StatefulPartitionedCall?
flatten_4/PartitionedCallPartitionedCallinput_5*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_flatten_4_layer_call_and_return_conditional_losses_102633892
flatten_4/PartitionedCall?
*tf.clip_by_value_6/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2,
*tf.clip_by_value_6/clip_by_value/Minimum/y?
(tf.clip_by_value_6/clip_by_value/MinimumMinimum"flatten_4/PartitionedCall:output:03tf.clip_by_value_6/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2*
(tf.clip_by_value_6/clip_by_value/Minimum?
"tf.clip_by_value_6/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_6/clip_by_value/y?
 tf.clip_by_value_6/clip_by_valueMaximum,tf.clip_by_value_6/clip_by_value/Minimum:z:0+tf.clip_by_value_6/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2"
 tf.clip_by_value_6/clip_by_value?
#tf.compat.v1.floor_div_4/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2%
#tf.compat.v1.floor_div_4/FloorDiv/y?
!tf.compat.v1.floor_div_4/FloorDivFloorDiv$tf.clip_by_value_6/clip_by_value:z:0,tf.compat.v1.floor_div_4/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.compat.v1.floor_div_4/FloorDiv?
$tf.math.greater_equal_6/GreaterEqualGreaterEqual"flatten_4/PartitionedCall:output:0&tf_math_greater_equal_6_greaterequal_y*
T0*'
_output_shapes
:?????????2&
$tf.math.greater_equal_6/GreaterEqual?
tf.math.floormod_4/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod_4/FloorMod/y?
tf.math.floormod_4/FloorModFloorMod$tf.clip_by_value_6/clip_by_value:z:0&tf.math.floormod_4/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_4/FloorMod?
$embedding_14/StatefulPartitionedCallStatefulPartitionedCall$tf.clip_by_value_6/clip_by_value:z:0embedding_14_10263426*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_14_layer_call_and_return_conditional_losses_102634172&
$embedding_14/StatefulPartitionedCall?
$embedding_12/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.floor_div_4/FloorDiv:z:0embedding_12_10263448*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_12_layer_call_and_return_conditional_losses_102634392&
$embedding_12/StatefulPartitionedCall?
tf.cast_6/CastCast(tf.math.greater_equal_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_6/Cast?
tf.__operators__.add_12/AddV2AddV2-embedding_14/StatefulPartitionedCall:output:0-embedding_12/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_12/AddV2?
$embedding_13/StatefulPartitionedCallStatefulPartitionedCalltf.math.floormod_4/FloorMod:z:0embedding_13_10263472*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_13_layer_call_and_return_conditional_losses_102634632&
$embedding_13/StatefulPartitionedCall?
tf.__operators__.add_13/AddV2AddV2!tf.__operators__.add_12/AddV2:z:0-embedding_13/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_13/AddV2?
tf.expand_dims_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_4/ExpandDims/dim?
tf.expand_dims_4/ExpandDims
ExpandDimstf.cast_6/Cast:y:0(tf.expand_dims_4/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_4/ExpandDims?
tf.math.multiply_4/MulMul!tf.__operators__.add_13/AddV2:z:0$tf.expand_dims_4/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_4/Mul?
*tf.math.reduce_sum_4/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_4/Sum/reduction_indices?
tf.math.reduce_sum_4/SumSumtf.math.multiply_4/Mul:z:03tf.math.reduce_sum_4/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_4/Sum?
IdentityIdentity!tf.math.reduce_sum_4/Sum:output:0%^embedding_12/StatefulPartitionedCall%^embedding_13/StatefulPartitionedCall%^embedding_14/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2L
$embedding_12/StatefulPartitionedCall$embedding_12/StatefulPartitionedCall2L
$embedding_13/StatefulPartitionedCall$embedding_13/StatefulPartitionedCall2L
$embedding_14/StatefulPartitionedCall$embedding_14/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_5:

_output_shapes
: 
?8
?
E__inference_model_4_layer_call_and_return_conditional_losses_10265210

inputs*
&tf_math_greater_equal_6_greaterequal_y*
&embedding_14_embedding_lookup_10265184*
&embedding_12_embedding_lookup_10265190*
&embedding_13_embedding_lookup_10265198
identity??embedding_12/embedding_lookup?embedding_13/embedding_lookup?embedding_14/embedding_lookups
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_4/Const?
flatten_4/ReshapeReshapeinputsflatten_4/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_4/Reshape?
*tf.clip_by_value_6/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2,
*tf.clip_by_value_6/clip_by_value/Minimum/y?
(tf.clip_by_value_6/clip_by_value/MinimumMinimumflatten_4/Reshape:output:03tf.clip_by_value_6/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2*
(tf.clip_by_value_6/clip_by_value/Minimum?
"tf.clip_by_value_6/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_6/clip_by_value/y?
 tf.clip_by_value_6/clip_by_valueMaximum,tf.clip_by_value_6/clip_by_value/Minimum:z:0+tf.clip_by_value_6/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2"
 tf.clip_by_value_6/clip_by_value?
#tf.compat.v1.floor_div_4/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2%
#tf.compat.v1.floor_div_4/FloorDiv/y?
!tf.compat.v1.floor_div_4/FloorDivFloorDiv$tf.clip_by_value_6/clip_by_value:z:0,tf.compat.v1.floor_div_4/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.compat.v1.floor_div_4/FloorDiv?
$tf.math.greater_equal_6/GreaterEqualGreaterEqualflatten_4/Reshape:output:0&tf_math_greater_equal_6_greaterequal_y*
T0*'
_output_shapes
:?????????2&
$tf.math.greater_equal_6/GreaterEqual?
tf.math.floormod_4/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod_4/FloorMod/y?
tf.math.floormod_4/FloorModFloorMod$tf.clip_by_value_6/clip_by_value:z:0&tf.math.floormod_4/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_4/FloorMod?
embedding_14/CastCast$tf.clip_by_value_6/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_14/Cast?
embedding_14/embedding_lookupResourceGather&embedding_14_embedding_lookup_10265184embedding_14/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*9
_class/
-+loc:@embedding_14/embedding_lookup/10265184*,
_output_shapes
:??????????*
dtype02
embedding_14/embedding_lookup?
&embedding_14/embedding_lookup/IdentityIdentity&embedding_14/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*9
_class/
-+loc:@embedding_14/embedding_lookup/10265184*,
_output_shapes
:??????????2(
&embedding_14/embedding_lookup/Identity?
(embedding_14/embedding_lookup/Identity_1Identity/embedding_14/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_14/embedding_lookup/Identity_1?
embedding_12/CastCast%tf.compat.v1.floor_div_4/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_12/Cast?
embedding_12/embedding_lookupResourceGather&embedding_12_embedding_lookup_10265190embedding_12/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*9
_class/
-+loc:@embedding_12/embedding_lookup/10265190*,
_output_shapes
:??????????*
dtype02
embedding_12/embedding_lookup?
&embedding_12/embedding_lookup/IdentityIdentity&embedding_12/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*9
_class/
-+loc:@embedding_12/embedding_lookup/10265190*,
_output_shapes
:??????????2(
&embedding_12/embedding_lookup/Identity?
(embedding_12/embedding_lookup/Identity_1Identity/embedding_12/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_12/embedding_lookup/Identity_1?
tf.cast_6/CastCast(tf.math.greater_equal_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_6/Cast?
tf.__operators__.add_12/AddV2AddV21embedding_14/embedding_lookup/Identity_1:output:01embedding_12/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_12/AddV2?
embedding_13/CastCasttf.math.floormod_4/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_13/Cast?
embedding_13/embedding_lookupResourceGather&embedding_13_embedding_lookup_10265198embedding_13/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*9
_class/
-+loc:@embedding_13/embedding_lookup/10265198*,
_output_shapes
:??????????*
dtype02
embedding_13/embedding_lookup?
&embedding_13/embedding_lookup/IdentityIdentity&embedding_13/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*9
_class/
-+loc:@embedding_13/embedding_lookup/10265198*,
_output_shapes
:??????????2(
&embedding_13/embedding_lookup/Identity?
(embedding_13/embedding_lookup/Identity_1Identity/embedding_13/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_13/embedding_lookup/Identity_1?
tf.__operators__.add_13/AddV2AddV2!tf.__operators__.add_12/AddV2:z:01embedding_13/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_13/AddV2?
tf.expand_dims_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_4/ExpandDims/dim?
tf.expand_dims_4/ExpandDims
ExpandDimstf.cast_6/Cast:y:0(tf.expand_dims_4/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_4/ExpandDims?
tf.math.multiply_4/MulMul!tf.__operators__.add_13/AddV2:z:0$tf.expand_dims_4/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_4/Mul?
*tf.math.reduce_sum_4/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_4/Sum/reduction_indices?
tf.math.reduce_sum_4/SumSumtf.math.multiply_4/Mul:z:03tf.math.reduce_sum_4/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_4/Sum?
IdentityIdentity!tf.math.reduce_sum_4/Sum:output:0^embedding_12/embedding_lookup^embedding_13/embedding_lookup^embedding_14/embedding_lookup*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2>
embedding_12/embedding_lookupembedding_12/embedding_lookup2>
embedding_13/embedding_lookupembedding_13/embedding_lookup2>
embedding_14/embedding_lookupembedding_14/embedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
*__inference_model_5_layer_call_fn_10263831
input_6
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_6unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_model_5_layer_call_and_return_conditional_losses_102638202
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_6:

_output_shapes
: 
?	
?
J__inference_embedding_16_layer_call_and_return_conditional_losses_10263689

inputs
embedding_lookup_10263683
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_10263683Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*,
_class"
 loc:@embedding_lookup/10263683*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@embedding_lookup/10263683*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
*__inference_model_4_layer_call_fn_10265265

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_model_4_layer_call_and_return_conditional_losses_102635492
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
+__inference_dense_22_layer_call_fn_10265486

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_22_layer_call_and_return_conditional_losses_102640352
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
u
/__inference_embedding_13_layer_call_fn_10265650

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_13_layer_call_and_return_conditional_losses_102634632
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
F__inference_dense_24_layer_call_and_return_conditional_losses_10264090

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_dense_21_layer_call_fn_10265447

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_21_layer_call_and_return_conditional_losses_102639552
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
+__inference_dense_26_layer_call_fn_10265588

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_26_layer_call_and_return_conditional_losses_102641792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?8
?
E__inference_model_4_layer_call_and_return_conditional_losses_10265252

inputs*
&tf_math_greater_equal_6_greaterequal_y*
&embedding_14_embedding_lookup_10265226*
&embedding_12_embedding_lookup_10265232*
&embedding_13_embedding_lookup_10265240
identity??embedding_12/embedding_lookup?embedding_13/embedding_lookup?embedding_14/embedding_lookups
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_4/Const?
flatten_4/ReshapeReshapeinputsflatten_4/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_4/Reshape?
*tf.clip_by_value_6/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2,
*tf.clip_by_value_6/clip_by_value/Minimum/y?
(tf.clip_by_value_6/clip_by_value/MinimumMinimumflatten_4/Reshape:output:03tf.clip_by_value_6/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2*
(tf.clip_by_value_6/clip_by_value/Minimum?
"tf.clip_by_value_6/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_6/clip_by_value/y?
 tf.clip_by_value_6/clip_by_valueMaximum,tf.clip_by_value_6/clip_by_value/Minimum:z:0+tf.clip_by_value_6/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2"
 tf.clip_by_value_6/clip_by_value?
#tf.compat.v1.floor_div_4/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2%
#tf.compat.v1.floor_div_4/FloorDiv/y?
!tf.compat.v1.floor_div_4/FloorDivFloorDiv$tf.clip_by_value_6/clip_by_value:z:0,tf.compat.v1.floor_div_4/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.compat.v1.floor_div_4/FloorDiv?
$tf.math.greater_equal_6/GreaterEqualGreaterEqualflatten_4/Reshape:output:0&tf_math_greater_equal_6_greaterequal_y*
T0*'
_output_shapes
:?????????2&
$tf.math.greater_equal_6/GreaterEqual?
tf.math.floormod_4/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod_4/FloorMod/y?
tf.math.floormod_4/FloorModFloorMod$tf.clip_by_value_6/clip_by_value:z:0&tf.math.floormod_4/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_4/FloorMod?
embedding_14/CastCast$tf.clip_by_value_6/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_14/Cast?
embedding_14/embedding_lookupResourceGather&embedding_14_embedding_lookup_10265226embedding_14/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*9
_class/
-+loc:@embedding_14/embedding_lookup/10265226*,
_output_shapes
:??????????*
dtype02
embedding_14/embedding_lookup?
&embedding_14/embedding_lookup/IdentityIdentity&embedding_14/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*9
_class/
-+loc:@embedding_14/embedding_lookup/10265226*,
_output_shapes
:??????????2(
&embedding_14/embedding_lookup/Identity?
(embedding_14/embedding_lookup/Identity_1Identity/embedding_14/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_14/embedding_lookup/Identity_1?
embedding_12/CastCast%tf.compat.v1.floor_div_4/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_12/Cast?
embedding_12/embedding_lookupResourceGather&embedding_12_embedding_lookup_10265232embedding_12/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*9
_class/
-+loc:@embedding_12/embedding_lookup/10265232*,
_output_shapes
:??????????*
dtype02
embedding_12/embedding_lookup?
&embedding_12/embedding_lookup/IdentityIdentity&embedding_12/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*9
_class/
-+loc:@embedding_12/embedding_lookup/10265232*,
_output_shapes
:??????????2(
&embedding_12/embedding_lookup/Identity?
(embedding_12/embedding_lookup/Identity_1Identity/embedding_12/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_12/embedding_lookup/Identity_1?
tf.cast_6/CastCast(tf.math.greater_equal_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_6/Cast?
tf.__operators__.add_12/AddV2AddV21embedding_14/embedding_lookup/Identity_1:output:01embedding_12/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_12/AddV2?
embedding_13/CastCasttf.math.floormod_4/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_13/Cast?
embedding_13/embedding_lookupResourceGather&embedding_13_embedding_lookup_10265240embedding_13/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*9
_class/
-+loc:@embedding_13/embedding_lookup/10265240*,
_output_shapes
:??????????*
dtype02
embedding_13/embedding_lookup?
&embedding_13/embedding_lookup/IdentityIdentity&embedding_13/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*9
_class/
-+loc:@embedding_13/embedding_lookup/10265240*,
_output_shapes
:??????????2(
&embedding_13/embedding_lookup/Identity?
(embedding_13/embedding_lookup/Identity_1Identity/embedding_13/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_13/embedding_lookup/Identity_1?
tf.__operators__.add_13/AddV2AddV2!tf.__operators__.add_12/AddV2:z:01embedding_13/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_13/AddV2?
tf.expand_dims_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_4/ExpandDims/dim?
tf.expand_dims_4/ExpandDims
ExpandDimstf.cast_6/Cast:y:0(tf.expand_dims_4/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_4/ExpandDims?
tf.math.multiply_4/MulMul!tf.__operators__.add_13/AddV2:z:0$tf.expand_dims_4/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_4/Mul?
*tf.math.reduce_sum_4/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_4/Sum/reduction_indices?
tf.math.reduce_sum_4/SumSumtf.math.multiply_4/Mul:z:03tf.math.reduce_sum_4/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_4/Sum?
IdentityIdentity!tf.math.reduce_sum_4/Sum:output:0^embedding_12/embedding_lookup^embedding_13/embedding_lookup^embedding_14/embedding_lookup*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2>
embedding_12/embedding_lookupembedding_12/embedding_lookup2>
embedding_13/embedding_lookupembedding_13/embedding_lookup2>
embedding_14/embedding_lookupembedding_14/embedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
~
.__inference_normalize_2_layer_call_fn_10265569
x
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_normalize_2_layer_call_and_return_conditional_losses_102641532
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:K G
(
_output_shapes
:??????????

_user_specified_namex
?
c
G__inference_flatten_4_layer_call_and_return_conditional_losses_10265594

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
F__inference_dense_18_layer_call_and_return_conditional_losses_10265399

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
u
/__inference_embedding_15_layer_call_fn_10265695

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_15_layer_call_and_return_conditional_losses_102636652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
+__inference_dense_23_layer_call_fn_10265505

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_23_layer_call_and_return_conditional_losses_102640632
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?-
?
E__inference_model_5_layer_call_and_return_conditional_losses_10263708
input_6*
&tf_math_greater_equal_7_greaterequal_y
embedding_17_10263652
embedding_15_10263674
embedding_16_10263698
identity??$embedding_15/StatefulPartitionedCall?$embedding_16/StatefulPartitionedCall?$embedding_17/StatefulPartitionedCall?
flatten_5/PartitionedCallPartitionedCallinput_6*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_flatten_5_layer_call_and_return_conditional_losses_102636152
flatten_5/PartitionedCall?
*tf.clip_by_value_7/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2,
*tf.clip_by_value_7/clip_by_value/Minimum/y?
(tf.clip_by_value_7/clip_by_value/MinimumMinimum"flatten_5/PartitionedCall:output:03tf.clip_by_value_7/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2*
(tf.clip_by_value_7/clip_by_value/Minimum?
"tf.clip_by_value_7/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_7/clip_by_value/y?
 tf.clip_by_value_7/clip_by_valueMaximum,tf.clip_by_value_7/clip_by_value/Minimum:z:0+tf.clip_by_value_7/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2"
 tf.clip_by_value_7/clip_by_value?
#tf.compat.v1.floor_div_5/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2%
#tf.compat.v1.floor_div_5/FloorDiv/y?
!tf.compat.v1.floor_div_5/FloorDivFloorDiv$tf.clip_by_value_7/clip_by_value:z:0,tf.compat.v1.floor_div_5/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.compat.v1.floor_div_5/FloorDiv?
$tf.math.greater_equal_7/GreaterEqualGreaterEqual"flatten_5/PartitionedCall:output:0&tf_math_greater_equal_7_greaterequal_y*
T0*'
_output_shapes
:?????????2&
$tf.math.greater_equal_7/GreaterEqual?
tf.math.floormod_5/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod_5/FloorMod/y?
tf.math.floormod_5/FloorModFloorMod$tf.clip_by_value_7/clip_by_value:z:0&tf.math.floormod_5/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_5/FloorMod?
$embedding_17/StatefulPartitionedCallStatefulPartitionedCall$tf.clip_by_value_7/clip_by_value:z:0embedding_17_10263652*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_17_layer_call_and_return_conditional_losses_102636432&
$embedding_17/StatefulPartitionedCall?
$embedding_15/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.floor_div_5/FloorDiv:z:0embedding_15_10263674*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_15_layer_call_and_return_conditional_losses_102636652&
$embedding_15/StatefulPartitionedCall?
tf.cast_7/CastCast(tf.math.greater_equal_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_7/Cast?
tf.__operators__.add_14/AddV2AddV2-embedding_17/StatefulPartitionedCall:output:0-embedding_15/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_14/AddV2?
$embedding_16/StatefulPartitionedCallStatefulPartitionedCalltf.math.floormod_5/FloorMod:z:0embedding_16_10263698*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_16_layer_call_and_return_conditional_losses_102636892&
$embedding_16/StatefulPartitionedCall?
tf.__operators__.add_15/AddV2AddV2!tf.__operators__.add_14/AddV2:z:0-embedding_16/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_15/AddV2?
tf.expand_dims_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_5/ExpandDims/dim?
tf.expand_dims_5/ExpandDims
ExpandDimstf.cast_7/Cast:y:0(tf.expand_dims_5/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_5/ExpandDims?
tf.math.multiply_5/MulMul!tf.__operators__.add_15/AddV2:z:0$tf.expand_dims_5/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_5/Mul?
*tf.math.reduce_sum_5/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_5/Sum/reduction_indices?
tf.math.reduce_sum_5/SumSumtf.math.multiply_5/Mul:z:03tf.math.reduce_sum_5/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_5/Sum?
IdentityIdentity!tf.math.reduce_sum_5/Sum:output:0%^embedding_15/StatefulPartitionedCall%^embedding_16/StatefulPartitionedCall%^embedding_17/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2L
$embedding_15/StatefulPartitionedCall$embedding_15/StatefulPartitionedCall2L
$embedding_16/StatefulPartitionedCall$embedding_16/StatefulPartitionedCall2L
$embedding_17/StatefulPartitionedCall$embedding_17/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_6:

_output_shapes
: 
?
?
+__inference_dense_20_layer_call_fn_10265467

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_20_layer_call_and_return_conditional_losses_102640092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
J__inference_embedding_17_layer_call_and_return_conditional_losses_10265671

inputs
embedding_lookup_10265665
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_10265665Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*,
_class"
 loc:@embedding_lookup/10265665*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@embedding_lookup/10265665*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
J__inference_embedding_13_layer_call_and_return_conditional_losses_10263463

inputs
embedding_lookup_10263457
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_10263457Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*,
_class"
 loc:@embedding_lookup/10263457*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@embedding_lookup/10263457*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
F__inference_dense_19_layer_call_and_return_conditional_losses_10263982

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
*__inference_model_4_layer_call_fn_10263605
input_5
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_model_4_layer_call_and_return_conditional_losses_102635942
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_5:

_output_shapes
: 
?
H
,__inference_flatten_4_layer_call_fn_10265599

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_flatten_4_layer_call_and_return_conditional_losses_102633892
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
F__inference_dense_20_layer_call_and_return_conditional_losses_10264009

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_dense_19_layer_call_fn_10265428

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_19_layer_call_and_return_conditional_losses_102639822
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
*__inference_model_4_layer_call_fn_10265278

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_model_4_layer_call_and_return_conditional_losses_102635942
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?-
?
E__inference_model_4_layer_call_and_return_conditional_losses_10263514
input_5*
&tf_math_greater_equal_6_greaterequal_y
embedding_14_10263496
embedding_12_10263499
embedding_13_10263504
identity??$embedding_12/StatefulPartitionedCall?$embedding_13/StatefulPartitionedCall?$embedding_14/StatefulPartitionedCall?
flatten_4/PartitionedCallPartitionedCallinput_5*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_flatten_4_layer_call_and_return_conditional_losses_102633892
flatten_4/PartitionedCall?
*tf.clip_by_value_6/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2,
*tf.clip_by_value_6/clip_by_value/Minimum/y?
(tf.clip_by_value_6/clip_by_value/MinimumMinimum"flatten_4/PartitionedCall:output:03tf.clip_by_value_6/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2*
(tf.clip_by_value_6/clip_by_value/Minimum?
"tf.clip_by_value_6/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_6/clip_by_value/y?
 tf.clip_by_value_6/clip_by_valueMaximum,tf.clip_by_value_6/clip_by_value/Minimum:z:0+tf.clip_by_value_6/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2"
 tf.clip_by_value_6/clip_by_value?
#tf.compat.v1.floor_div_4/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2%
#tf.compat.v1.floor_div_4/FloorDiv/y?
!tf.compat.v1.floor_div_4/FloorDivFloorDiv$tf.clip_by_value_6/clip_by_value:z:0,tf.compat.v1.floor_div_4/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.compat.v1.floor_div_4/FloorDiv?
$tf.math.greater_equal_6/GreaterEqualGreaterEqual"flatten_4/PartitionedCall:output:0&tf_math_greater_equal_6_greaterequal_y*
T0*'
_output_shapes
:?????????2&
$tf.math.greater_equal_6/GreaterEqual?
tf.math.floormod_4/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod_4/FloorMod/y?
tf.math.floormod_4/FloorModFloorMod$tf.clip_by_value_6/clip_by_value:z:0&tf.math.floormod_4/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_4/FloorMod?
$embedding_14/StatefulPartitionedCallStatefulPartitionedCall$tf.clip_by_value_6/clip_by_value:z:0embedding_14_10263496*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_14_layer_call_and_return_conditional_losses_102634172&
$embedding_14/StatefulPartitionedCall?
$embedding_12/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.floor_div_4/FloorDiv:z:0embedding_12_10263499*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_12_layer_call_and_return_conditional_losses_102634392&
$embedding_12/StatefulPartitionedCall?
tf.cast_6/CastCast(tf.math.greater_equal_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_6/Cast?
tf.__operators__.add_12/AddV2AddV2-embedding_14/StatefulPartitionedCall:output:0-embedding_12/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_12/AddV2?
$embedding_13/StatefulPartitionedCallStatefulPartitionedCalltf.math.floormod_4/FloorMod:z:0embedding_13_10263504*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_embedding_13_layer_call_and_return_conditional_losses_102634632&
$embedding_13/StatefulPartitionedCall?
tf.__operators__.add_13/AddV2AddV2!tf.__operators__.add_12/AddV2:z:0-embedding_13/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_13/AddV2?
tf.expand_dims_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_4/ExpandDims/dim?
tf.expand_dims_4/ExpandDims
ExpandDimstf.cast_6/Cast:y:0(tf.expand_dims_4/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_4/ExpandDims?
tf.math.multiply_4/MulMul!tf.__operators__.add_13/AddV2:z:0$tf.expand_dims_4/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_4/Mul?
*tf.math.reduce_sum_4/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_4/Sum/reduction_indices?
tf.math.reduce_sum_4/SumSumtf.math.multiply_4/Mul:z:03tf.math.reduce_sum_4/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_4/Sum?
IdentityIdentity!tf.math.reduce_sum_4/Sum:output:0%^embedding_12/StatefulPartitionedCall%^embedding_13/StatefulPartitionedCall%^embedding_14/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2L
$embedding_12/StatefulPartitionedCall$embedding_12/StatefulPartitionedCall2L
$embedding_13/StatefulPartitionedCall$embedding_13/StatefulPartitionedCall2L
$embedding_14/StatefulPartitionedCall$embedding_14/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_5:

_output_shapes
: 
?
?
+__inference_dense_24_layer_call_fn_10265524

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_24_layer_call_and_return_conditional_losses_102640902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
L__inference_custom_model_2_layer_call_and_return_conditional_losses_10265030

inputs_0_0

inputs_0_1
inputs_1*
&tf_math_greater_equal_8_greaterequal_y2
.model_4_tf_math_greater_equal_6_greaterequal_y2
.model_4_embedding_14_embedding_lookup_102648802
.model_4_embedding_12_embedding_lookup_102648862
.model_4_embedding_13_embedding_lookup_102648942
.model_5_tf_math_greater_equal_7_greaterequal_y2
.model_5_embedding_17_embedding_lookup_102649182
.model_5_embedding_15_embedding_lookup_102649242
.model_5_embedding_16_embedding_lookup_10264932.
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
identity??dense_18/BiasAdd/ReadVariableOp?dense_18/MatMul/ReadVariableOp?dense_19/BiasAdd/ReadVariableOp?dense_19/MatMul/ReadVariableOp?dense_20/BiasAdd/ReadVariableOp?dense_20/MatMul/ReadVariableOp?dense_21/BiasAdd/ReadVariableOp?dense_21/MatMul/ReadVariableOp?dense_22/BiasAdd/ReadVariableOp?dense_22/MatMul/ReadVariableOp?dense_23/BiasAdd/ReadVariableOp?dense_23/MatMul/ReadVariableOp?dense_24/BiasAdd/ReadVariableOp?dense_24/MatMul/ReadVariableOp?dense_25/BiasAdd/ReadVariableOp?dense_25/MatMul/ReadVariableOp?dense_26/BiasAdd/ReadVariableOp?dense_26/MatMul/ReadVariableOp?%model_4/embedding_12/embedding_lookup?%model_4/embedding_13/embedding_lookup?%model_4/embedding_14/embedding_lookup?%model_5/embedding_15/embedding_lookup?%model_5/embedding_16/embedding_lookup?%model_5/embedding_17/embedding_lookup?2normalize_2/normalization_2/Reshape/ReadVariableOp?4normalize_2/normalization_2/Reshape_1/ReadVariableOp?
$tf.math.greater_equal_8/GreaterEqualGreaterEqualinputs_1&tf_math_greater_equal_8_greaterequal_y*
T0*'
_output_shapes
:?????????
2&
$tf.math.greater_equal_8/GreaterEqual?
model_4/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
model_4/flatten_4/Const?
model_4/flatten_4/ReshapeReshape
inputs_0_0 model_4/flatten_4/Const:output:0*
T0*'
_output_shapes
:?????????2
model_4/flatten_4/Reshape?
2model_4/tf.clip_by_value_6/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI24
2model_4/tf.clip_by_value_6/clip_by_value/Minimum/y?
0model_4/tf.clip_by_value_6/clip_by_value/MinimumMinimum"model_4/flatten_4/Reshape:output:0;model_4/tf.clip_by_value_6/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????22
0model_4/tf.clip_by_value_6/clip_by_value/Minimum?
*model_4/tf.clip_by_value_6/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*model_4/tf.clip_by_value_6/clip_by_value/y?
(model_4/tf.clip_by_value_6/clip_by_valueMaximum4model_4/tf.clip_by_value_6/clip_by_value/Minimum:z:03model_4/tf.clip_by_value_6/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2*
(model_4/tf.clip_by_value_6/clip_by_value?
+model_4/tf.compat.v1.floor_div_4/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2-
+model_4/tf.compat.v1.floor_div_4/FloorDiv/y?
)model_4/tf.compat.v1.floor_div_4/FloorDivFloorDiv,model_4/tf.clip_by_value_6/clip_by_value:z:04model_4/tf.compat.v1.floor_div_4/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2+
)model_4/tf.compat.v1.floor_div_4/FloorDiv?
,model_4/tf.math.greater_equal_6/GreaterEqualGreaterEqual"model_4/flatten_4/Reshape:output:0.model_4_tf_math_greater_equal_6_greaterequal_y*
T0*'
_output_shapes
:?????????2.
,model_4/tf.math.greater_equal_6/GreaterEqual?
%model_4/tf.math.floormod_4/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2'
%model_4/tf.math.floormod_4/FloorMod/y?
#model_4/tf.math.floormod_4/FloorModFloorMod,model_4/tf.clip_by_value_6/clip_by_value:z:0.model_4/tf.math.floormod_4/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2%
#model_4/tf.math.floormod_4/FloorMod?
model_4/embedding_14/CastCast,model_4/tf.clip_by_value_6/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_4/embedding_14/Cast?
%model_4/embedding_14/embedding_lookupResourceGather.model_4_embedding_14_embedding_lookup_10264880model_4/embedding_14/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*A
_class7
53loc:@model_4/embedding_14/embedding_lookup/10264880*,
_output_shapes
:??????????*
dtype02'
%model_4/embedding_14/embedding_lookup?
.model_4/embedding_14/embedding_lookup/IdentityIdentity.model_4/embedding_14/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@model_4/embedding_14/embedding_lookup/10264880*,
_output_shapes
:??????????20
.model_4/embedding_14/embedding_lookup/Identity?
0model_4/embedding_14/embedding_lookup/Identity_1Identity7model_4/embedding_14/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????22
0model_4/embedding_14/embedding_lookup/Identity_1?
model_4/embedding_12/CastCast-model_4/tf.compat.v1.floor_div_4/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_4/embedding_12/Cast?
%model_4/embedding_12/embedding_lookupResourceGather.model_4_embedding_12_embedding_lookup_10264886model_4/embedding_12/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*A
_class7
53loc:@model_4/embedding_12/embedding_lookup/10264886*,
_output_shapes
:??????????*
dtype02'
%model_4/embedding_12/embedding_lookup?
.model_4/embedding_12/embedding_lookup/IdentityIdentity.model_4/embedding_12/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@model_4/embedding_12/embedding_lookup/10264886*,
_output_shapes
:??????????20
.model_4/embedding_12/embedding_lookup/Identity?
0model_4/embedding_12/embedding_lookup/Identity_1Identity7model_4/embedding_12/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????22
0model_4/embedding_12/embedding_lookup/Identity_1?
model_4/tf.cast_6/CastCast0model_4/tf.math.greater_equal_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
model_4/tf.cast_6/Cast?
%model_4/tf.__operators__.add_12/AddV2AddV29model_4/embedding_14/embedding_lookup/Identity_1:output:09model_4/embedding_12/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2'
%model_4/tf.__operators__.add_12/AddV2?
model_4/embedding_13/CastCast'model_4/tf.math.floormod_4/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_4/embedding_13/Cast?
%model_4/embedding_13/embedding_lookupResourceGather.model_4_embedding_13_embedding_lookup_10264894model_4/embedding_13/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*A
_class7
53loc:@model_4/embedding_13/embedding_lookup/10264894*,
_output_shapes
:??????????*
dtype02'
%model_4/embedding_13/embedding_lookup?
.model_4/embedding_13/embedding_lookup/IdentityIdentity.model_4/embedding_13/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@model_4/embedding_13/embedding_lookup/10264894*,
_output_shapes
:??????????20
.model_4/embedding_13/embedding_lookup/Identity?
0model_4/embedding_13/embedding_lookup/Identity_1Identity7model_4/embedding_13/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????22
0model_4/embedding_13/embedding_lookup/Identity_1?
%model_4/tf.__operators__.add_13/AddV2AddV2)model_4/tf.__operators__.add_12/AddV2:z:09model_4/embedding_13/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2'
%model_4/tf.__operators__.add_13/AddV2?
'model_4/tf.expand_dims_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'model_4/tf.expand_dims_4/ExpandDims/dim?
#model_4/tf.expand_dims_4/ExpandDims
ExpandDimsmodel_4/tf.cast_6/Cast:y:00model_4/tf.expand_dims_4/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2%
#model_4/tf.expand_dims_4/ExpandDims?
model_4/tf.math.multiply_4/MulMul)model_4/tf.__operators__.add_13/AddV2:z:0,model_4/tf.expand_dims_4/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2 
model_4/tf.math.multiply_4/Mul?
2model_4/tf.math.reduce_sum_4/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :24
2model_4/tf.math.reduce_sum_4/Sum/reduction_indices?
 model_4/tf.math.reduce_sum_4/SumSum"model_4/tf.math.multiply_4/Mul:z:0;model_4/tf.math.reduce_sum_4/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2"
 model_4/tf.math.reduce_sum_4/Sum?
model_5/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
model_5/flatten_5/Const?
model_5/flatten_5/ReshapeReshape
inputs_0_1 model_5/flatten_5/Const:output:0*
T0*'
_output_shapes
:?????????2
model_5/flatten_5/Reshape?
2model_5/tf.clip_by_value_7/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI24
2model_5/tf.clip_by_value_7/clip_by_value/Minimum/y?
0model_5/tf.clip_by_value_7/clip_by_value/MinimumMinimum"model_5/flatten_5/Reshape:output:0;model_5/tf.clip_by_value_7/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????22
0model_5/tf.clip_by_value_7/clip_by_value/Minimum?
*model_5/tf.clip_by_value_7/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*model_5/tf.clip_by_value_7/clip_by_value/y?
(model_5/tf.clip_by_value_7/clip_by_valueMaximum4model_5/tf.clip_by_value_7/clip_by_value/Minimum:z:03model_5/tf.clip_by_value_7/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2*
(model_5/tf.clip_by_value_7/clip_by_value?
+model_5/tf.compat.v1.floor_div_5/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2-
+model_5/tf.compat.v1.floor_div_5/FloorDiv/y?
)model_5/tf.compat.v1.floor_div_5/FloorDivFloorDiv,model_5/tf.clip_by_value_7/clip_by_value:z:04model_5/tf.compat.v1.floor_div_5/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2+
)model_5/tf.compat.v1.floor_div_5/FloorDiv?
,model_5/tf.math.greater_equal_7/GreaterEqualGreaterEqual"model_5/flatten_5/Reshape:output:0.model_5_tf_math_greater_equal_7_greaterequal_y*
T0*'
_output_shapes
:?????????2.
,model_5/tf.math.greater_equal_7/GreaterEqual?
%model_5/tf.math.floormod_5/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2'
%model_5/tf.math.floormod_5/FloorMod/y?
#model_5/tf.math.floormod_5/FloorModFloorMod,model_5/tf.clip_by_value_7/clip_by_value:z:0.model_5/tf.math.floormod_5/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2%
#model_5/tf.math.floormod_5/FloorMod?
model_5/embedding_17/CastCast,model_5/tf.clip_by_value_7/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_5/embedding_17/Cast?
%model_5/embedding_17/embedding_lookupResourceGather.model_5_embedding_17_embedding_lookup_10264918model_5/embedding_17/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*A
_class7
53loc:@model_5/embedding_17/embedding_lookup/10264918*,
_output_shapes
:??????????*
dtype02'
%model_5/embedding_17/embedding_lookup?
.model_5/embedding_17/embedding_lookup/IdentityIdentity.model_5/embedding_17/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@model_5/embedding_17/embedding_lookup/10264918*,
_output_shapes
:??????????20
.model_5/embedding_17/embedding_lookup/Identity?
0model_5/embedding_17/embedding_lookup/Identity_1Identity7model_5/embedding_17/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????22
0model_5/embedding_17/embedding_lookup/Identity_1?
model_5/embedding_15/CastCast-model_5/tf.compat.v1.floor_div_5/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_5/embedding_15/Cast?
%model_5/embedding_15/embedding_lookupResourceGather.model_5_embedding_15_embedding_lookup_10264924model_5/embedding_15/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*A
_class7
53loc:@model_5/embedding_15/embedding_lookup/10264924*,
_output_shapes
:??????????*
dtype02'
%model_5/embedding_15/embedding_lookup?
.model_5/embedding_15/embedding_lookup/IdentityIdentity.model_5/embedding_15/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@model_5/embedding_15/embedding_lookup/10264924*,
_output_shapes
:??????????20
.model_5/embedding_15/embedding_lookup/Identity?
0model_5/embedding_15/embedding_lookup/Identity_1Identity7model_5/embedding_15/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????22
0model_5/embedding_15/embedding_lookup/Identity_1?
model_5/tf.cast_7/CastCast0model_5/tf.math.greater_equal_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
model_5/tf.cast_7/Cast?
%model_5/tf.__operators__.add_14/AddV2AddV29model_5/embedding_17/embedding_lookup/Identity_1:output:09model_5/embedding_15/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2'
%model_5/tf.__operators__.add_14/AddV2?
model_5/embedding_16/CastCast'model_5/tf.math.floormod_5/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_5/embedding_16/Cast?
%model_5/embedding_16/embedding_lookupResourceGather.model_5_embedding_16_embedding_lookup_10264932model_5/embedding_16/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*A
_class7
53loc:@model_5/embedding_16/embedding_lookup/10264932*,
_output_shapes
:??????????*
dtype02'
%model_5/embedding_16/embedding_lookup?
.model_5/embedding_16/embedding_lookup/IdentityIdentity.model_5/embedding_16/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@model_5/embedding_16/embedding_lookup/10264932*,
_output_shapes
:??????????20
.model_5/embedding_16/embedding_lookup/Identity?
0model_5/embedding_16/embedding_lookup/Identity_1Identity7model_5/embedding_16/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????22
0model_5/embedding_16/embedding_lookup/Identity_1?
%model_5/tf.__operators__.add_15/AddV2AddV2)model_5/tf.__operators__.add_14/AddV2:z:09model_5/embedding_16/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2'
%model_5/tf.__operators__.add_15/AddV2?
'model_5/tf.expand_dims_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'model_5/tf.expand_dims_5/ExpandDims/dim?
#model_5/tf.expand_dims_5/ExpandDims
ExpandDimsmodel_5/tf.cast_7/Cast:y:00model_5/tf.expand_dims_5/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2%
#model_5/tf.expand_dims_5/ExpandDims?
model_5/tf.math.multiply_5/MulMul)model_5/tf.__operators__.add_15/AddV2:z:0,model_5/tf.expand_dims_5/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2 
model_5/tf.math.multiply_5/Mul?
2model_5/tf.math.reduce_sum_5/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :24
2model_5/tf.math.reduce_sum_5/Sum/reduction_indices?
 model_5/tf.math.reduce_sum_5/SumSum"model_5/tf.math.multiply_5/Mul:z:0;model_5/tf.math.reduce_sum_5/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2"
 model_5/tf.math.reduce_sum_5/Sum?
(tf.clip_by_value_8/clip_by_value/MinimumMinimuminputs_1*tf_clip_by_value_8_clip_by_value_minimum_y*
T0*'
_output_shapes
:?????????
2*
(tf.clip_by_value_8/clip_by_value/Minimum?
 tf.clip_by_value_8/clip_by_valueMaximum,tf.clip_by_value_8/clip_by_value/Minimum:z:0"tf_clip_by_value_8_clip_by_value_y*
T0*'
_output_shapes
:?????????
2"
 tf.clip_by_value_8/clip_by_value?
tf.cast_8/CastCast(tf.math.greater_equal_8/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
2
tf.cast_8/Castt
tf.concat_6/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_6/concat/axis?
tf.concat_6/concatConcatV2)model_4/tf.math.reduce_sum_4/Sum:output:0)model_5/tf.math.reduce_sum_5/Sum:output:0 tf.concat_6/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_6/concat}
tf.concat_7/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_7/concat/axis?
tf.concat_7/concatConcatV2$tf.clip_by_value_8/clip_by_value:z:0tf.cast_8/Cast:y:0 tf.concat_7/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
tf.concat_7/concat?
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_18/MatMul/ReadVariableOp?
dense_18/MatMulMatMultf.concat_6/concat:output:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_18/MatMul?
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_18/BiasAdd/ReadVariableOp?
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_18/BiasAddt
dense_18/ReluReludense_18/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_18/Relu?
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_21/MatMul/ReadVariableOp?
dense_21/MatMulMatMultf.concat_7/concat:output:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_21/MatMul?
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_21/BiasAdd/ReadVariableOp?
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_21/BiasAdd?
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_19/MatMul/ReadVariableOp?
dense_19/MatMulMatMuldense_18/Relu:activations:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_19/MatMul?
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_19/BiasAdd/ReadVariableOp?
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_19/BiasAddt
dense_19/ReluReludense_19/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_19/Relu?
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_20/MatMul/ReadVariableOp?
dense_20/MatMulMatMuldense_19/Relu:activations:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_20/MatMul?
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_20/BiasAdd/ReadVariableOp?
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_20/BiasAddt
dense_20/ReluReludense_20/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_20/Relu?
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_22/MatMul/ReadVariableOp?
dense_22/MatMulMatMuldense_21/BiasAdd:output:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_22/MatMul?
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_22/BiasAdd/ReadVariableOp?
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_22/BiasAdd}
tf.concat_8/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_8/concat/axis?
tf.concat_8/concatConcatV2dense_20/Relu:activations:0dense_22/BiasAdd:output:0 tf.concat_8/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_8/concat?
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_23/MatMul/ReadVariableOp?
dense_23/MatMulMatMultf.concat_8/concat:output:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_23/MatMul?
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_23/BiasAdd/ReadVariableOp?
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_23/BiasAdd|
tf.nn.relu_6/ReluReludense_23/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_6/Relu?
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_24/MatMul/ReadVariableOp?
dense_24/MatMulMatMultf.nn.relu_6/Relu:activations:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_24/MatMul?
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_24/BiasAdd/ReadVariableOp?
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_24/BiasAdd?
tf.__operators__.add_16/AddV2AddV2dense_24/BiasAdd:output:0tf.nn.relu_6/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_16/AddV2?
tf.nn.relu_7/ReluRelu!tf.__operators__.add_16/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_7/Relu?
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_25/MatMul/ReadVariableOp?
dense_25/MatMulMatMultf.nn.relu_7/Relu:activations:0&dense_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_25/MatMul?
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_25/BiasAdd/ReadVariableOp?
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_25/BiasAdd?
tf.__operators__.add_17/AddV2AddV2dense_25/BiasAdd:output:0tf.nn.relu_7/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_17/AddV2?
tf.nn.relu_8/ReluRelu!tf.__operators__.add_17/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_8/Relu?
2normalize_2/normalization_2/Reshape/ReadVariableOpReadVariableOp;normalize_2_normalization_2_reshape_readvariableop_resource*
_output_shapes	
:?*
dtype024
2normalize_2/normalization_2/Reshape/ReadVariableOp?
)normalize_2/normalization_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2+
)normalize_2/normalization_2/Reshape/shape?
#normalize_2/normalization_2/ReshapeReshape:normalize_2/normalization_2/Reshape/ReadVariableOp:value:02normalize_2/normalization_2/Reshape/shape:output:0*
T0*
_output_shapes
:	?2%
#normalize_2/normalization_2/Reshape?
4normalize_2/normalization_2/Reshape_1/ReadVariableOpReadVariableOp=normalize_2_normalization_2_reshape_1_readvariableop_resource*
_output_shapes	
:?*
dtype026
4normalize_2/normalization_2/Reshape_1/ReadVariableOp?
+normalize_2/normalization_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2-
+normalize_2/normalization_2/Reshape_1/shape?
%normalize_2/normalization_2/Reshape_1Reshape<normalize_2/normalization_2/Reshape_1/ReadVariableOp:value:04normalize_2/normalization_2/Reshape_1/shape:output:0*
T0*
_output_shapes
:	?2'
%normalize_2/normalization_2/Reshape_1?
normalize_2/normalization_2/subSubtf.nn.relu_8/Relu:activations:0,normalize_2/normalization_2/Reshape:output:0*
T0*(
_output_shapes
:??????????2!
normalize_2/normalization_2/sub?
 normalize_2/normalization_2/SqrtSqrt.normalize_2/normalization_2/Reshape_1:output:0*
T0*
_output_shapes
:	?2"
 normalize_2/normalization_2/Sqrt?
%normalize_2/normalization_2/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32'
%normalize_2/normalization_2/Maximum/y?
#normalize_2/normalization_2/MaximumMaximum$normalize_2/normalization_2/Sqrt:y:0.normalize_2/normalization_2/Maximum/y:output:0*
T0*
_output_shapes
:	?2%
#normalize_2/normalization_2/Maximum?
#normalize_2/normalization_2/truedivRealDiv#normalize_2/normalization_2/sub:z:0'normalize_2/normalization_2/Maximum:z:0*
T0*(
_output_shapes
:??????????2%
#normalize_2/normalization_2/truediv?
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_26/MatMul/ReadVariableOp?
dense_26/MatMulMatMul'normalize_2/normalization_2/truediv:z:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_26/MatMul?
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_26/BiasAdd/ReadVariableOp?
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_26/BiasAdd?
IdentityIdentitydense_26/BiasAdd:output:0 ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp ^dense_26/BiasAdd/ReadVariableOp^dense_26/MatMul/ReadVariableOp&^model_4/embedding_12/embedding_lookup&^model_4/embedding_13/embedding_lookup&^model_4/embedding_14/embedding_lookup&^model_5/embedding_15/embedding_lookup&^model_5/embedding_16/embedding_lookup&^model_5/embedding_17/embedding_lookup3^normalize_2/normalization_2/Reshape/ReadVariableOp5^normalize_2/normalization_2/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
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
:?????????
$
_user_specified_name
inputs/0/0:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/0/1:QM
'
_output_shapes
:?????????

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
?	
?
F__inference_dense_21_layer_call_and_return_conditional_losses_10265438

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
F__inference_dense_24_layer_call_and_return_conditional_losses_10265515

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?X
?

L__inference_custom_model_2_layer_call_and_return_conditional_losses_10264196

cards0

cards1
bets*
&tf_math_greater_equal_8_greaterequal_y
model_4_10263865
model_4_10263867
model_4_10263869
model_4_10263871
model_5_10263900
model_5_10263902
model_5_10263904
model_5_10263906.
*tf_clip_by_value_8_clip_by_value_minimum_y&
"tf_clip_by_value_8_clip_by_value_y
dense_18_10263940
dense_18_10263942
dense_21_10263966
dense_21_10263968
dense_19_10263993
dense_19_10263995
dense_20_10264020
dense_20_10264022
dense_22_10264046
dense_22_10264048
dense_23_10264074
dense_23_10264076
dense_24_10264101
dense_24_10264103
dense_25_10264129
dense_25_10264131
normalize_2_10264164
normalize_2_10264166
dense_26_10264190
dense_26_10264192
identity?? dense_18/StatefulPartitionedCall? dense_19/StatefulPartitionedCall? dense_20/StatefulPartitionedCall? dense_21/StatefulPartitionedCall? dense_22/StatefulPartitionedCall? dense_23/StatefulPartitionedCall? dense_24/StatefulPartitionedCall? dense_25/StatefulPartitionedCall? dense_26/StatefulPartitionedCall?model_4/StatefulPartitionedCall?model_5/StatefulPartitionedCall?#normalize_2/StatefulPartitionedCall?
$tf.math.greater_equal_8/GreaterEqualGreaterEqualbets&tf_math_greater_equal_8_greaterequal_y*
T0*'
_output_shapes
:?????????
2&
$tf.math.greater_equal_8/GreaterEqual?
model_4/StatefulPartitionedCallStatefulPartitionedCallcards0model_4_10263865model_4_10263867model_4_10263869model_4_10263871*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_model_4_layer_call_and_return_conditional_losses_102635492!
model_4/StatefulPartitionedCall?
model_5/StatefulPartitionedCallStatefulPartitionedCallcards1model_5_10263900model_5_10263902model_5_10263904model_5_10263906*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_model_5_layer_call_and_return_conditional_losses_102637752!
model_5/StatefulPartitionedCall?
(tf.clip_by_value_8/clip_by_value/MinimumMinimumbets*tf_clip_by_value_8_clip_by_value_minimum_y*
T0*'
_output_shapes
:?????????
2*
(tf.clip_by_value_8/clip_by_value/Minimum?
 tf.clip_by_value_8/clip_by_valueMaximum,tf.clip_by_value_8/clip_by_value/Minimum:z:0"tf_clip_by_value_8_clip_by_value_y*
T0*'
_output_shapes
:?????????
2"
 tf.clip_by_value_8/clip_by_value?
tf.cast_8/CastCast(tf.math.greater_equal_8/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
2
tf.cast_8/Castt
tf.concat_6/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_6/concat/axis?
tf.concat_6/concatConcatV2(model_4/StatefulPartitionedCall:output:0(model_5/StatefulPartitionedCall:output:0 tf.concat_6/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_6/concat}
tf.concat_7/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_7/concat/axis?
tf.concat_7/concatConcatV2$tf.clip_by_value_8/clip_by_value:z:0tf.cast_8/Cast:y:0 tf.concat_7/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
tf.concat_7/concat?
 dense_18/StatefulPartitionedCallStatefulPartitionedCalltf.concat_6/concat:output:0dense_18_10263940dense_18_10263942*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_18_layer_call_and_return_conditional_losses_102639292"
 dense_18/StatefulPartitionedCall?
 dense_21/StatefulPartitionedCallStatefulPartitionedCalltf.concat_7/concat:output:0dense_21_10263966dense_21_10263968*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_21_layer_call_and_return_conditional_losses_102639552"
 dense_21/StatefulPartitionedCall?
 dense_19/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0dense_19_10263993dense_19_10263995*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_19_layer_call_and_return_conditional_losses_102639822"
 dense_19/StatefulPartitionedCall?
 dense_20/StatefulPartitionedCallStatefulPartitionedCall)dense_19/StatefulPartitionedCall:output:0dense_20_10264020dense_20_10264022*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_20_layer_call_and_return_conditional_losses_102640092"
 dense_20/StatefulPartitionedCall?
 dense_22/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0dense_22_10264046dense_22_10264048*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_22_layer_call_and_return_conditional_losses_102640352"
 dense_22/StatefulPartitionedCall}
tf.concat_8/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_8/concat/axis?
tf.concat_8/concatConcatV2)dense_20/StatefulPartitionedCall:output:0)dense_22/StatefulPartitionedCall:output:0 tf.concat_8/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_8/concat?
 dense_23/StatefulPartitionedCallStatefulPartitionedCalltf.concat_8/concat:output:0dense_23_10264074dense_23_10264076*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_23_layer_call_and_return_conditional_losses_102640632"
 dense_23/StatefulPartitionedCall?
tf.nn.relu_6/ReluRelu)dense_23/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_6/Relu?
 dense_24/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_6/Relu:activations:0dense_24_10264101dense_24_10264103*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_24_layer_call_and_return_conditional_losses_102640902"
 dense_24/StatefulPartitionedCall?
tf.__operators__.add_16/AddV2AddV2)dense_24/StatefulPartitionedCall:output:0tf.nn.relu_6/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_16/AddV2?
tf.nn.relu_7/ReluRelu!tf.__operators__.add_16/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_7/Relu?
 dense_25/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_7/Relu:activations:0dense_25_10264129dense_25_10264131*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_25_layer_call_and_return_conditional_losses_102641182"
 dense_25/StatefulPartitionedCall?
tf.__operators__.add_17/AddV2AddV2)dense_25/StatefulPartitionedCall:output:0tf.nn.relu_7/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_17/AddV2?
tf.nn.relu_8/ReluRelu!tf.__operators__.add_17/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_8/Relu?
#normalize_2/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_8/Relu:activations:0normalize_2_10264164normalize_2_10264166*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *R
fMRK
I__inference_normalize_2_layer_call_and_return_conditional_losses_102641532%
#normalize_2/StatefulPartitionedCall?
 dense_26/StatefulPartitionedCallStatefulPartitionedCall,normalize_2/StatefulPartitionedCall:output:0dense_26_10264190dense_26_10264192*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_26_layer_call_and_return_conditional_losses_102641792"
 dense_26/StatefulPartitionedCall?
IdentityIdentity)dense_26/StatefulPartitionedCall:output:0!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall ^model_4/StatefulPartitionedCall ^model_5/StatefulPartitionedCall$^normalize_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
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
:?????????
 
_user_specified_namecards0:OK
'
_output_shapes
:?????????
 
_user_specified_namecards1:MI
'
_output_shapes
:?????????


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
?	
?
F__inference_dense_22_layer_call_and_return_conditional_losses_10264035

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
J__inference_embedding_15_layer_call_and_return_conditional_losses_10263665

inputs
embedding_lookup_10263659
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_10263659Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*,
_class"
 loc:@embedding_lookup/10263659*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@embedding_lookup/10263659*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
5
bets-
serving_default_bets:0?????????

9
cards0/
serving_default_cards0:0?????????
9
cards1/
serving_default_cards1:0?????????<
dense_260
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
??
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
trainable_variables
	variables
regularization_losses
 	keras_api
!
signatures
?_default_save_signature
?__call__
+?&call_and_return_all_conditional_losses"??
_tf_keras_network??{"class_name": "CustomModel", "name": "custom_model_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "custom_model_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cards0"}, "name": "cards0", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cards1"}, "name": "cards1", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "bets"}, "name": "bets", "inbound_nodes": []}, {"class_name": "Functional", "config": {"name": "model_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}, "name": "input_5", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_4", "inbound_nodes": [[["input_5", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_6", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_6", "inbound_nodes": [["flatten_4", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_4", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_4", "inbound_nodes": [["tf.clip_by_value_6", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_14", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_14", "inbound_nodes": [[["tf.clip_by_value_6", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_12", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_12", "inbound_nodes": [[["tf.compat.v1.floor_div_4", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_4", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_4", "inbound_nodes": [["tf.clip_by_value_6", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_6", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_6", "inbound_nodes": [["flatten_4", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_12", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_12", "inbound_nodes": [["embedding_14", 0, 0, {"y": ["embedding_12", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_13", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_13", "inbound_nodes": [[["tf.math.floormod_4", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_6", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_6", "inbound_nodes": [["tf.math.greater_equal_6", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_13", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_13", "inbound_nodes": [["tf.__operators__.add_12", 0, 0, {"y": ["embedding_13", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_4", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_4", "inbound_nodes": [["tf.cast_6", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_4", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_4", "inbound_nodes": [["tf.__operators__.add_13", 0, 0, {"y": ["tf.expand_dims_4", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_4", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_4", "inbound_nodes": [["tf.math.multiply_4", 0, 0, {"axis": 1}]]}], "input_layers": [["input_5", 0, 0]], "output_layers": [["tf.math.reduce_sum_4", 0, 0]]}, "name": "model_4", "inbound_nodes": [[["cards0", 0, 0, {}]]]}, {"class_name": "Functional", "config": {"name": "model_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_6"}, "name": "input_6", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_5", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_5", "inbound_nodes": [[["input_6", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_7", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_7", "inbound_nodes": [["flatten_5", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_5", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_5", "inbound_nodes": [["tf.clip_by_value_7", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_17", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_17", "inbound_nodes": [[["tf.clip_by_value_7", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_15", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_15", "inbound_nodes": [[["tf.compat.v1.floor_div_5", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_5", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_5", "inbound_nodes": [["tf.clip_by_value_7", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_7", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_7", "inbound_nodes": [["flatten_5", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_14", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_14", "inbound_nodes": [["embedding_17", 0, 0, {"y": ["embedding_15", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_16", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_16", "inbound_nodes": [[["tf.math.floormod_5", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_7", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_7", "inbound_nodes": [["tf.math.greater_equal_7", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_15", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_15", "inbound_nodes": [["tf.__operators__.add_14", 0, 0, {"y": ["embedding_16", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_5", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_5", "inbound_nodes": [["tf.cast_7", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_5", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_5", "inbound_nodes": [["tf.__operators__.add_15", 0, 0, {"y": ["tf.expand_dims_5", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_5", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_5", "inbound_nodes": [["tf.math.multiply_5", 0, 0, {"axis": 1}]]}], "input_layers": [["input_6", 0, 0]], "output_layers": [["tf.math.reduce_sum_5", 0, 0]]}, "name": "model_5", "inbound_nodes": [[["cards1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_8", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_8", "inbound_nodes": [["bets", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_6", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_6", "inbound_nodes": [[["model_4", 1, 0, {"axis": 1}], ["model_5", 1, 0, {"axis": 1}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_8", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_8", "inbound_nodes": [["bets", 0, 0, {"clip_value_min": 0.0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_8", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_8", "inbound_nodes": [["tf.math.greater_equal_8", 0, 0, {"dtype": "float32"}]]}, {"class_name": "Dense", "config": {"name": "dense_18", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_18", "inbound_nodes": [[["tf.concat_6", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_7", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_7", "inbound_nodes": [[["tf.clip_by_value_8", 0, 0, {"axis": -1}], ["tf.cast_8", 0, 0, {"axis": -1}]]]}, {"class_name": "Dense", "config": {"name": "dense_19", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_19", "inbound_nodes": [[["dense_18", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_21", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_21", "inbound_nodes": [[["tf.concat_7", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_20", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_20", "inbound_nodes": [[["dense_19", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_22", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_22", "inbound_nodes": [[["dense_21", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_8", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_8", "inbound_nodes": [[["dense_20", 0, 0, {"axis": -1}], ["dense_22", 0, 0, {"axis": -1}]]]}, {"class_name": "Dense", "config": {"name": "dense_23", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_23", "inbound_nodes": [[["tf.concat_8", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_6", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu_6", "inbound_nodes": [["dense_23", 0, 0, {"name": null}]]}, {"class_name": "Dense", "config": {"name": "dense_24", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_24", "inbound_nodes": [[["tf.nn.relu_6", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_16", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_16", "inbound_nodes": [["dense_24", 0, 0, {"y": ["tf.nn.relu_6", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_7", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu_7", "inbound_nodes": [["tf.__operators__.add_16", 0, 0, {"name": null}]]}, {"class_name": "Dense", "config": {"name": "dense_25", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_25", "inbound_nodes": [[["tf.nn.relu_7", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_17", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_17", "inbound_nodes": [["dense_25", 0, 0, {"y": ["tf.nn.relu_7", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_8", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu_8", "inbound_nodes": [["tf.__operators__.add_17", 0, 0, {"name": null}]]}, {"class_name": "Normalize", "config": {"name": "normalize_2", "trainable": true, "dtype": "float32"}, "name": "normalize_2", "inbound_nodes": [[["tf.nn.relu_8", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_26", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Constant", "config": {"value": 0}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_26", "inbound_nodes": [[["normalize_2", 0, 0, {}]]]}], "input_layers": [[["cards0", 0, 0], ["cards1", 0, 0]], ["bets", 0, 0]], "output_layers": [["dense_26", 0, 0]]}, "build_input_shape": [[{"class_name": "TensorShape", "items": [null, 2]}, {"class_name": "TensorShape", "items": [null, 3]}], {"class_name": "TensorShape", "items": [null, 10]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "CustomModel", "config": {"name": "custom_model_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cards0"}, "name": "cards0", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cards1"}, "name": "cards1", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "bets"}, "name": "bets", "inbound_nodes": []}, {"class_name": "Functional", "config": {"name": "model_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}, "name": "input_5", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_4", "inbound_nodes": [[["input_5", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_6", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_6", "inbound_nodes": [["flatten_4", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_4", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_4", "inbound_nodes": [["tf.clip_by_value_6", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_14", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_14", "inbound_nodes": [[["tf.clip_by_value_6", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_12", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_12", "inbound_nodes": [[["tf.compat.v1.floor_div_4", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_4", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_4", "inbound_nodes": [["tf.clip_by_value_6", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_6", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_6", "inbound_nodes": [["flatten_4", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_12", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_12", "inbound_nodes": [["embedding_14", 0, 0, {"y": ["embedding_12", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_13", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_13", "inbound_nodes": [[["tf.math.floormod_4", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_6", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_6", "inbound_nodes": [["tf.math.greater_equal_6", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_13", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_13", "inbound_nodes": [["tf.__operators__.add_12", 0, 0, {"y": ["embedding_13", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_4", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_4", "inbound_nodes": [["tf.cast_6", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_4", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_4", "inbound_nodes": [["tf.__operators__.add_13", 0, 0, {"y": ["tf.expand_dims_4", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_4", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_4", "inbound_nodes": [["tf.math.multiply_4", 0, 0, {"axis": 1}]]}], "input_layers": [["input_5", 0, 0]], "output_layers": [["tf.math.reduce_sum_4", 0, 0]]}, "name": "model_4", "inbound_nodes": [[["cards0", 0, 0, {}]]]}, {"class_name": "Functional", "config": {"name": "model_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_6"}, "name": "input_6", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_5", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_5", "inbound_nodes": [[["input_6", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_7", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_7", "inbound_nodes": [["flatten_5", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_5", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_5", "inbound_nodes": [["tf.clip_by_value_7", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_17", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_17", "inbound_nodes": [[["tf.clip_by_value_7", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_15", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_15", "inbound_nodes": [[["tf.compat.v1.floor_div_5", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_5", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_5", "inbound_nodes": [["tf.clip_by_value_7", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_7", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_7", "inbound_nodes": [["flatten_5", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_14", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_14", "inbound_nodes": [["embedding_17", 0, 0, {"y": ["embedding_15", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_16", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_16", "inbound_nodes": [[["tf.math.floormod_5", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_7", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_7", "inbound_nodes": [["tf.math.greater_equal_7", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_15", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_15", "inbound_nodes": [["tf.__operators__.add_14", 0, 0, {"y": ["embedding_16", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_5", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_5", "inbound_nodes": [["tf.cast_7", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_5", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_5", "inbound_nodes": [["tf.__operators__.add_15", 0, 0, {"y": ["tf.expand_dims_5", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_5", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_5", "inbound_nodes": [["tf.math.multiply_5", 0, 0, {"axis": 1}]]}], "input_layers": [["input_6", 0, 0]], "output_layers": [["tf.math.reduce_sum_5", 0, 0]]}, "name": "model_5", "inbound_nodes": [[["cards1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_8", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_8", "inbound_nodes": [["bets", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_6", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_6", "inbound_nodes": [[["model_4", 1, 0, {"axis": 1}], ["model_5", 1, 0, {"axis": 1}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_8", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_8", "inbound_nodes": [["bets", 0, 0, {"clip_value_min": 0.0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_8", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_8", "inbound_nodes": [["tf.math.greater_equal_8", 0, 0, {"dtype": "float32"}]]}, {"class_name": "Dense", "config": {"name": "dense_18", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_18", "inbound_nodes": [[["tf.concat_6", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_7", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_7", "inbound_nodes": [[["tf.clip_by_value_8", 0, 0, {"axis": -1}], ["tf.cast_8", 0, 0, {"axis": -1}]]]}, {"class_name": "Dense", "config": {"name": "dense_19", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_19", "inbound_nodes": [[["dense_18", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_21", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_21", "inbound_nodes": [[["tf.concat_7", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_20", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_20", "inbound_nodes": [[["dense_19", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_22", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_22", "inbound_nodes": [[["dense_21", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_8", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_8", "inbound_nodes": [[["dense_20", 0, 0, {"axis": -1}], ["dense_22", 0, 0, {"axis": -1}]]]}, {"class_name": "Dense", "config": {"name": "dense_23", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_23", "inbound_nodes": [[["tf.concat_8", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_6", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu_6", "inbound_nodes": [["dense_23", 0, 0, {"name": null}]]}, {"class_name": "Dense", "config": {"name": "dense_24", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_24", "inbound_nodes": [[["tf.nn.relu_6", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_16", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_16", "inbound_nodes": [["dense_24", 0, 0, {"y": ["tf.nn.relu_6", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_7", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu_7", "inbound_nodes": [["tf.__operators__.add_16", 0, 0, {"name": null}]]}, {"class_name": "Dense", "config": {"name": "dense_25", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_25", "inbound_nodes": [[["tf.nn.relu_7", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_17", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_17", "inbound_nodes": [["dense_25", 0, 0, {"y": ["tf.nn.relu_7", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_8", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu_8", "inbound_nodes": [["tf.__operators__.add_17", 0, 0, {"name": null}]]}, {"class_name": "Normalize", "config": {"name": "normalize_2", "trainable": true, "dtype": "float32"}, "name": "normalize_2", "inbound_nodes": [[["tf.nn.relu_8", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_26", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Constant", "config": {"value": 0}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_26", "inbound_nodes": [[["normalize_2", 0, 0, {}]]]}], "input_layers": [[["cards0", 0, 0], ["cards1", 0, 0]], ["bets", 0, 0]], "output_layers": [["dense_26", 0, 0]]}}, "training_config": {"loss": null, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0020000000949949026, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "cards0", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cards0"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "cards1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cards1"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "bets", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "bets"}}
?Q
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
1trainable_variables
2	variables
3regularization_losses
4	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?N
_tf_keras_network?N{"class_name": "Functional", "name": "model_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}, "name": "input_5", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_4", "inbound_nodes": [[["input_5", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_6", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_6", "inbound_nodes": [["flatten_4", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_4", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_4", "inbound_nodes": [["tf.clip_by_value_6", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_14", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_14", "inbound_nodes": [[["tf.clip_by_value_6", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_12", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_12", "inbound_nodes": [[["tf.compat.v1.floor_div_4", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_4", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_4", "inbound_nodes": [["tf.clip_by_value_6", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_6", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_6", "inbound_nodes": [["flatten_4", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_12", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_12", "inbound_nodes": [["embedding_14", 0, 0, {"y": ["embedding_12", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_13", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_13", "inbound_nodes": [[["tf.math.floormod_4", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_6", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_6", "inbound_nodes": [["tf.math.greater_equal_6", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_13", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_13", "inbound_nodes": [["tf.__operators__.add_12", 0, 0, {"y": ["embedding_13", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_4", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_4", "inbound_nodes": [["tf.cast_6", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_4", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_4", "inbound_nodes": [["tf.__operators__.add_13", 0, 0, {"y": ["tf.expand_dims_4", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_4", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_4", "inbound_nodes": [["tf.math.multiply_4", 0, 0, {"axis": 1}]]}], "input_layers": [["input_5", 0, 0]], "output_layers": [["tf.math.reduce_sum_4", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 2]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}, "name": "input_5", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_4", "inbound_nodes": [[["input_5", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_6", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_6", "inbound_nodes": [["flatten_4", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_4", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_4", "inbound_nodes": [["tf.clip_by_value_6", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_14", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_14", "inbound_nodes": [[["tf.clip_by_value_6", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_12", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_12", "inbound_nodes": [[["tf.compat.v1.floor_div_4", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_4", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_4", "inbound_nodes": [["tf.clip_by_value_6", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_6", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_6", "inbound_nodes": [["flatten_4", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_12", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_12", "inbound_nodes": [["embedding_14", 0, 0, {"y": ["embedding_12", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_13", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_13", "inbound_nodes": [[["tf.math.floormod_4", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_6", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_6", "inbound_nodes": [["tf.math.greater_equal_6", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_13", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_13", "inbound_nodes": [["tf.__operators__.add_12", 0, 0, {"y": ["embedding_13", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_4", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_4", "inbound_nodes": [["tf.cast_6", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_4", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_4", "inbound_nodes": [["tf.__operators__.add_13", 0, 0, {"y": ["tf.expand_dims_4", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_4", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_4", "inbound_nodes": [["tf.math.multiply_4", 0, 0, {"axis": 1}]]}], "input_layers": [["input_5", 0, 0]], "output_layers": [["tf.math.reduce_sum_4", 0, 0]]}}}
?Q
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
Dtrainable_variables
E	variables
Fregularization_losses
G	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?N
_tf_keras_network?N{"class_name": "Functional", "name": "model_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_6"}, "name": "input_6", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_5", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_5", "inbound_nodes": [[["input_6", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_7", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_7", "inbound_nodes": [["flatten_5", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_5", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_5", "inbound_nodes": [["tf.clip_by_value_7", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_17", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_17", "inbound_nodes": [[["tf.clip_by_value_7", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_15", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_15", "inbound_nodes": [[["tf.compat.v1.floor_div_5", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_5", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_5", "inbound_nodes": [["tf.clip_by_value_7", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_7", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_7", "inbound_nodes": [["flatten_5", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_14", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_14", "inbound_nodes": [["embedding_17", 0, 0, {"y": ["embedding_15", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_16", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_16", "inbound_nodes": [[["tf.math.floormod_5", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_7", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_7", "inbound_nodes": [["tf.math.greater_equal_7", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_15", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_15", "inbound_nodes": [["tf.__operators__.add_14", 0, 0, {"y": ["embedding_16", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_5", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_5", "inbound_nodes": [["tf.cast_7", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_5", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_5", "inbound_nodes": [["tf.__operators__.add_15", 0, 0, {"y": ["tf.expand_dims_5", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_5", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_5", "inbound_nodes": [["tf.math.multiply_5", 0, 0, {"axis": 1}]]}], "input_layers": [["input_6", 0, 0]], "output_layers": [["tf.math.reduce_sum_5", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 3]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_6"}, "name": "input_6", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_5", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_5", "inbound_nodes": [[["input_6", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_7", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_7", "inbound_nodes": [["flatten_5", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_5", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_5", "inbound_nodes": [["tf.clip_by_value_7", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_17", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_17", "inbound_nodes": [[["tf.clip_by_value_7", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_15", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_15", "inbound_nodes": [[["tf.compat.v1.floor_div_5", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_5", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_5", "inbound_nodes": [["tf.clip_by_value_7", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_7", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_7", "inbound_nodes": [["flatten_5", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_14", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_14", "inbound_nodes": [["embedding_17", 0, 0, {"y": ["embedding_15", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_16", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_16", "inbound_nodes": [[["tf.math.floormod_5", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_7", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_7", "inbound_nodes": [["tf.math.greater_equal_7", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_15", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_15", "inbound_nodes": [["tf.__operators__.add_14", 0, 0, {"y": ["embedding_16", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_5", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_5", "inbound_nodes": [["tf.cast_7", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_5", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_5", "inbound_nodes": [["tf.__operators__.add_15", 0, 0, {"y": ["tf.expand_dims_5", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_5", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_5", "inbound_nodes": [["tf.math.multiply_5", 0, 0, {"axis": 1}]]}], "input_layers": [["input_6", 0, 0]], "output_layers": [["tf.math.reduce_sum_5", 0, 0]]}}}
?
H	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.greater_equal_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.greater_equal_8", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}}
?
I	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.concat_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.concat_6", "trainable": true, "dtype": "float32", "function": "concat"}}
?
J	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.clip_by_value_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.clip_by_value_8", "trainable": true, "dtype": "float32", "function": "clip_by_value"}}
?
K	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.cast_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.cast_8", "trainable": true, "dtype": "float32", "function": "cast"}}
?

Lkernel
Mbias
Ntrainable_variables
O	variables
Pregularization_losses
Q	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_18", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
?
R	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.concat_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.concat_7", "trainable": true, "dtype": "float32", "function": "concat"}}
?

Skernel
Tbias
Utrainable_variables
V	variables
Wregularization_losses
X	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_19", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_19", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?

Ykernel
Zbias
[trainable_variables
\	variables
]regularization_losses
^	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_21", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_21", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20]}}
?

_kernel
`bias
atrainable_variables
b	variables
cregularization_losses
d	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_20", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_20", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?

ekernel
fbias
gtrainable_variables
h	variables
iregularization_losses
j	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_22", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_22", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?
k	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.concat_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.concat_8", "trainable": true, "dtype": "float32", "function": "concat"}}
?

lkernel
mbias
ntrainable_variables
o	variables
pregularization_losses
q	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_23", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_23", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
?
r	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.nn.relu_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.nn.relu_6", "trainable": true, "dtype": "float32", "function": "nn.relu"}}
?

skernel
tbias
utrainable_variables
v	variables
wregularization_losses
x	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_24", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_24", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?
y	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.__operators__.add_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_16", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
?
z	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.nn.relu_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.nn.relu_7", "trainable": true, "dtype": "float32", "function": "nn.relu"}}
?

{kernel
|bias
}trainable_variables
~	variables
regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_25", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_25", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.__operators__.add_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_17", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.nn.relu_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.nn.relu_8", "trainable": true, "dtype": "float32", "function": "nn.relu"}}
?
?	normalize
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Normalize", "name": "normalize_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "normalize_2", "trainable": true, "dtype": "float32"}}
?
?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_26", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_26", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Constant", "config": {"value": 0}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?
	?iter
?beta_1
?beta_2

?decay
?learning_rateLm?Mm?Sm?Tm?Ym?Zm?_m?`m?em?fm?lm?mm?sm?tm?{m?|m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?Lv?Mv?Sv?Tv?Yv?Zv?_v?`v?ev?fv?lv?mv?sv?tv?{v?|v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?"
	optimizer
 "
trackable_dict_wrapper
?
?0
?1
?2
?3
?4
?5
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
?22
?23"
trackable_list_wrapper
?
?0
?1
?2
?3
?4
?5
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
?22
?23
?24
?25
?26"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
trainable_variables
 ?layer_regularization_losses
	variables
regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_5", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}}
?
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.clip_by_value_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.clip_by_value_6", "trainable": true, "dtype": "float32", "function": "clip_by_value"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.compat.v1.floor_div_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.compat.v1.floor_div_4", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}}
?
?
embeddings
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_14", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}}
?
?
embeddings
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_12", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.floormod_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.floormod_4", "trainable": true, "dtype": "float32", "function": "math.floormod"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.greater_equal_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.greater_equal_6", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.__operators__.add_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_12", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
?
?
embeddings
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_13", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.cast_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.cast_6", "trainable": true, "dtype": "float32", "function": "cast"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.__operators__.add_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_13", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.expand_dims_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.expand_dims_4", "trainable": true, "dtype": "float32", "function": "expand_dims"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.multiply_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_4", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.reduce_sum_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.reduce_sum_4", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}}
8
?0
?1
?2"
trackable_list_wrapper
8
?0
?1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
1trainable_variables
 ?layer_regularization_losses
2	variables
3regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_6", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_6"}}
?
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_5", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.clip_by_value_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.clip_by_value_7", "trainable": true, "dtype": "float32", "function": "clip_by_value"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.compat.v1.floor_div_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.compat.v1.floor_div_5", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}}
?
?
embeddings
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_17", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3]}}
?
?
embeddings
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_15", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3]}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.floormod_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.floormod_5", "trainable": true, "dtype": "float32", "function": "math.floormod"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.greater_equal_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.greater_equal_7", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.__operators__.add_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_14", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
?
?
embeddings
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_16", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3]}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.cast_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.cast_7", "trainable": true, "dtype": "float32", "function": "cast"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.__operators__.add_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_15", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.expand_dims_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.expand_dims_5", "trainable": true, "dtype": "float32", "function": "expand_dims"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.multiply_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_5", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.reduce_sum_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.reduce_sum_5", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}}
8
?0
?1
?2"
trackable_list_wrapper
8
?0
?1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
Dtrainable_variables
 ?layer_regularization_losses
E	variables
Fregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
??2dense_18/kernel
:?2dense_18/bias
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
?
?layers
Ntrainable_variables
 ?layer_regularization_losses
O	variables
Pregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
#:!
??2dense_19/kernel
:?2dense_19/bias
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
?
?layers
Utrainable_variables
 ?layer_regularization_losses
V	variables
Wregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	?2dense_21/kernel
:?2dense_21/bias
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
?
?layers
[trainable_variables
 ?layer_regularization_losses
\	variables
]regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!
??2dense_20/kernel
:?2dense_20/bias
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
?
?layers
atrainable_variables
 ?layer_regularization_losses
b	variables
cregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!
??2dense_22/kernel
:?2dense_22/bias
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
?
?layers
gtrainable_variables
 ?layer_regularization_losses
h	variables
iregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
#:!
??2dense_23/kernel
:?2dense_23/bias
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
?
?layers
ntrainable_variables
 ?layer_regularization_losses
o	variables
pregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
#:!
??2dense_24/kernel
:?2dense_24/bias
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
?
?layers
utrainable_variables
 ?layer_regularization_losses
v	variables
wregularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
#:!
??2dense_25/kernel
:?2dense_25/bias
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
?
?layers
}trainable_variables
 ?layer_regularization_losses
~	variables
regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
?
?state_variables
?_broadcast_shape
	?mean
?variance

?count
?	keras_api"?
_tf_keras_layer?{"class_name": "Normalization", "name": "normalization_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "normalization_2", "trainable": true, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [-1]}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
 "
trackable_list_wrapper
8
?0
?1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	?2dense_26/kernel
:2dense_26/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
*:(	4?2embedding_14/embeddings
*:(	?2embedding_12/embeddings
*:(	?2embedding_13/embeddings
*:(	4?2embedding_17/embeddings
*:(	?2embedding_15/embeddings
*:(	?2embedding_16/embeddings
-:+?2 normalize_2/normalization_2/mean
1:/?2$normalize_2/normalization_2/variance
):'	 2!normalize_2/normalization_2/count
?
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
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
?0"
trackable_list_wrapper
8
?0
?1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
?
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
 "
trackable_list_wrapper
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?trainable_variables
 ?layer_regularization_losses
?	variables
?regularization_losses
?layer_metrics
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
?
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
 "
trackable_list_wrapper
F
	?mean
?variance

?count"
trackable_dict_wrapper
 "
trackable_list_wrapper
"
_generic_user_object
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
8
?0
?1
?2"
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
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
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
 "
trackable_list_wrapper
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
(:&
??2Adam/dense_18/kernel/m
!:?2Adam/dense_18/bias/m
(:&
??2Adam/dense_19/kernel/m
!:?2Adam/dense_19/bias/m
':%	?2Adam/dense_21/kernel/m
!:?2Adam/dense_21/bias/m
(:&
??2Adam/dense_20/kernel/m
!:?2Adam/dense_20/bias/m
(:&
??2Adam/dense_22/kernel/m
!:?2Adam/dense_22/bias/m
(:&
??2Adam/dense_23/kernel/m
!:?2Adam/dense_23/bias/m
(:&
??2Adam/dense_24/kernel/m
!:?2Adam/dense_24/bias/m
(:&
??2Adam/dense_25/kernel/m
!:?2Adam/dense_25/bias/m
':%	?2Adam/dense_26/kernel/m
 :2Adam/dense_26/bias/m
/:-	4?2Adam/embedding_14/embeddings/m
/:-	?2Adam/embedding_12/embeddings/m
/:-	?2Adam/embedding_13/embeddings/m
/:-	4?2Adam/embedding_17/embeddings/m
/:-	?2Adam/embedding_15/embeddings/m
/:-	?2Adam/embedding_16/embeddings/m
(:&
??2Adam/dense_18/kernel/v
!:?2Adam/dense_18/bias/v
(:&
??2Adam/dense_19/kernel/v
!:?2Adam/dense_19/bias/v
':%	?2Adam/dense_21/kernel/v
!:?2Adam/dense_21/bias/v
(:&
??2Adam/dense_20/kernel/v
!:?2Adam/dense_20/bias/v
(:&
??2Adam/dense_22/kernel/v
!:?2Adam/dense_22/bias/v
(:&
??2Adam/dense_23/kernel/v
!:?2Adam/dense_23/bias/v
(:&
??2Adam/dense_24/kernel/v
!:?2Adam/dense_24/bias/v
(:&
??2Adam/dense_25/kernel/v
!:?2Adam/dense_25/bias/v
':%	?2Adam/dense_26/kernel/v
 :2Adam/dense_26/bias/v
/:-	4?2Adam/embedding_14/embeddings/v
/:-	?2Adam/embedding_12/embeddings/v
/:-	?2Adam/embedding_13/embeddings/v
/:-	4?2Adam/embedding_17/embeddings/v
/:-	?2Adam/embedding_15/embeddings/v
/:-	?2Adam/embedding_16/embeddings/v
?2?
#__inference__wrapped_model_10263379?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *q?n
l?i
G?D
 ?
cards0?????????
 ?
cards1?????????
?
bets?????????

?2?
1__inference_custom_model_2_layer_call_fn_10265099
1__inference_custom_model_2_layer_call_fn_10264450
1__inference_custom_model_2_layer_call_fn_10265168
1__inference_custom_model_2_layer_call_fn_10264611?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
L__inference_custom_model_2_layer_call_and_return_conditional_losses_10265030
L__inference_custom_model_2_layer_call_and_return_conditional_losses_10264860
L__inference_custom_model_2_layer_call_and_return_conditional_losses_10264196
L__inference_custom_model_2_layer_call_and_return_conditional_losses_10264288?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_model_4_layer_call_fn_10265278
*__inference_model_4_layer_call_fn_10263560
*__inference_model_4_layer_call_fn_10263605
*__inference_model_4_layer_call_fn_10265265?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_model_4_layer_call_and_return_conditional_losses_10265210
E__inference_model_4_layer_call_and_return_conditional_losses_10263514
E__inference_model_4_layer_call_and_return_conditional_losses_10263482
E__inference_model_4_layer_call_and_return_conditional_losses_10265252?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_model_5_layer_call_fn_10265375
*__inference_model_5_layer_call_fn_10263786
*__inference_model_5_layer_call_fn_10265388
*__inference_model_5_layer_call_fn_10263831?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_model_5_layer_call_and_return_conditional_losses_10263740
E__inference_model_5_layer_call_and_return_conditional_losses_10265362
E__inference_model_5_layer_call_and_return_conditional_losses_10263708
E__inference_model_5_layer_call_and_return_conditional_losses_10265320?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_dense_18_layer_call_fn_10265408?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_18_layer_call_and_return_conditional_losses_10265399?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_19_layer_call_fn_10265428?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_19_layer_call_and_return_conditional_losses_10265419?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_21_layer_call_fn_10265447?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_21_layer_call_and_return_conditional_losses_10265438?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_20_layer_call_fn_10265467?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_20_layer_call_and_return_conditional_losses_10265458?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_22_layer_call_fn_10265486?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_22_layer_call_and_return_conditional_losses_10265477?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_23_layer_call_fn_10265505?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_23_layer_call_and_return_conditional_losses_10265496?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_24_layer_call_fn_10265524?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_24_layer_call_and_return_conditional_losses_10265515?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_25_layer_call_fn_10265543?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_25_layer_call_and_return_conditional_losses_10265534?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_normalize_2_layer_call_fn_10265569?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_normalize_2_layer_call_and_return_conditional_losses_10265560?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_26_layer_call_fn_10265588?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_26_layer_call_and_return_conditional_losses_10265579?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
&__inference_signature_wrapper_10264690betscards0cards1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_flatten_4_layer_call_fn_10265599?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_flatten_4_layer_call_and_return_conditional_losses_10265594?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_embedding_14_layer_call_fn_10265616?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_embedding_14_layer_call_and_return_conditional_losses_10265609?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_embedding_12_layer_call_fn_10265633?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_embedding_12_layer_call_and_return_conditional_losses_10265626?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_embedding_13_layer_call_fn_10265650?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_embedding_13_layer_call_and_return_conditional_losses_10265643?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_flatten_5_layer_call_fn_10265661?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_flatten_5_layer_call_and_return_conditional_losses_10265656?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_embedding_17_layer_call_fn_10265678?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_embedding_17_layer_call_and_return_conditional_losses_10265671?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_embedding_15_layer_call_fn_10265695?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_embedding_15_layer_call_and_return_conditional_losses_10265688?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_embedding_16_layer_call_fn_10265712?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_embedding_16_layer_call_and_return_conditional_losses_10265705?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
Const_4?
#__inference__wrapped_model_10263379?.???????????LMYZST_`eflmst{|????{?x
q?n
l?i
G?D
 ?
cards0?????????
 ?
cards1?????????
?
bets?????????

? "3?0
.
dense_26"?
dense_26??????????
L__inference_custom_model_2_layer_call_and_return_conditional_losses_10264196?.???????????LMYZST_`eflmst{|???????
y?v
l?i
G?D
 ?
cards0?????????
 ?
cards1?????????
?
bets?????????

p

 
? "%?"
?
0?????????
? ?
L__inference_custom_model_2_layer_call_and_return_conditional_losses_10264288?.???????????LMYZST_`eflmst{|???????
y?v
l?i
G?D
 ?
cards0?????????
 ?
cards1?????????
?
bets?????????

p 

 
? "%?"
?
0?????????
? ?
L__inference_custom_model_2_layer_call_and_return_conditional_losses_10264860?.???????????LMYZST_`eflmst{|???????
???
x?u
O?L
$?!

inputs/0/0?????????
$?!

inputs/0/1?????????
"?
inputs/1?????????

p

 
? "%?"
?
0?????????
? ?
L__inference_custom_model_2_layer_call_and_return_conditional_losses_10265030?.???????????LMYZST_`eflmst{|???????
???
x?u
O?L
$?!

inputs/0/0?????????
$?!

inputs/0/1?????????
"?
inputs/1?????????

p 

 
? "%?"
?
0?????????
? ?
1__inference_custom_model_2_layer_call_fn_10264450?.???????????LMYZST_`eflmst{|???????
y?v
l?i
G?D
 ?
cards0?????????
 ?
cards1?????????
?
bets?????????

p

 
? "???????????
1__inference_custom_model_2_layer_call_fn_10264611?.???????????LMYZST_`eflmst{|???????
y?v
l?i
G?D
 ?
cards0?????????
 ?
cards1?????????
?
bets?????????

p 

 
? "???????????
1__inference_custom_model_2_layer_call_fn_10265099?.???????????LMYZST_`eflmst{|???????
???
x?u
O?L
$?!

inputs/0/0?????????
$?!

inputs/0/1?????????
"?
inputs/1?????????

p

 
? "???????????
1__inference_custom_model_2_layer_call_fn_10265168?.???????????LMYZST_`eflmst{|???????
???
x?u
O?L
$?!

inputs/0/0?????????
$?!

inputs/0/1?????????
"?
inputs/1?????????

p 

 
? "???????????
F__inference_dense_18_layer_call_and_return_conditional_losses_10265399^LM0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
+__inference_dense_18_layer_call_fn_10265408QLM0?-
&?#
!?
inputs??????????
? "????????????
F__inference_dense_19_layer_call_and_return_conditional_losses_10265419^ST0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
+__inference_dense_19_layer_call_fn_10265428QST0?-
&?#
!?
inputs??????????
? "????????????
F__inference_dense_20_layer_call_and_return_conditional_losses_10265458^_`0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
+__inference_dense_20_layer_call_fn_10265467Q_`0?-
&?#
!?
inputs??????????
? "????????????
F__inference_dense_21_layer_call_and_return_conditional_losses_10265438]YZ/?,
%?"
 ?
inputs?????????
? "&?#
?
0??????????
? 
+__inference_dense_21_layer_call_fn_10265447PYZ/?,
%?"
 ?
inputs?????????
? "????????????
F__inference_dense_22_layer_call_and_return_conditional_losses_10265477^ef0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
+__inference_dense_22_layer_call_fn_10265486Qef0?-
&?#
!?
inputs??????????
? "????????????
F__inference_dense_23_layer_call_and_return_conditional_losses_10265496^lm0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
+__inference_dense_23_layer_call_fn_10265505Qlm0?-
&?#
!?
inputs??????????
? "????????????
F__inference_dense_24_layer_call_and_return_conditional_losses_10265515^st0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
+__inference_dense_24_layer_call_fn_10265524Qst0?-
&?#
!?
inputs??????????
? "????????????
F__inference_dense_25_layer_call_and_return_conditional_losses_10265534^{|0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
+__inference_dense_25_layer_call_fn_10265543Q{|0?-
&?#
!?
inputs??????????
? "????????????
F__inference_dense_26_layer_call_and_return_conditional_losses_10265579_??0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? ?
+__inference_dense_26_layer_call_fn_10265588R??0?-
&?#
!?
inputs??????????
? "???????????
J__inference_embedding_12_layer_call_and_return_conditional_losses_10265626a?/?,
%?"
 ?
inputs?????????
? "*?'
 ?
0??????????
? ?
/__inference_embedding_12_layer_call_fn_10265633T?/?,
%?"
 ?
inputs?????????
? "????????????
J__inference_embedding_13_layer_call_and_return_conditional_losses_10265643a?/?,
%?"
 ?
inputs?????????
? "*?'
 ?
0??????????
? ?
/__inference_embedding_13_layer_call_fn_10265650T?/?,
%?"
 ?
inputs?????????
? "????????????
J__inference_embedding_14_layer_call_and_return_conditional_losses_10265609a?/?,
%?"
 ?
inputs?????????
? "*?'
 ?
0??????????
? ?
/__inference_embedding_14_layer_call_fn_10265616T?/?,
%?"
 ?
inputs?????????
? "????????????
J__inference_embedding_15_layer_call_and_return_conditional_losses_10265688a?/?,
%?"
 ?
inputs?????????
? "*?'
 ?
0??????????
? ?
/__inference_embedding_15_layer_call_fn_10265695T?/?,
%?"
 ?
inputs?????????
? "????????????
J__inference_embedding_16_layer_call_and_return_conditional_losses_10265705a?/?,
%?"
 ?
inputs?????????
? "*?'
 ?
0??????????
? ?
/__inference_embedding_16_layer_call_fn_10265712T?/?,
%?"
 ?
inputs?????????
? "????????????
J__inference_embedding_17_layer_call_and_return_conditional_losses_10265671a?/?,
%?"
 ?
inputs?????????
? "*?'
 ?
0??????????
? ?
/__inference_embedding_17_layer_call_fn_10265678T?/?,
%?"
 ?
inputs?????????
? "????????????
G__inference_flatten_4_layer_call_and_return_conditional_losses_10265594X/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? {
,__inference_flatten_4_layer_call_fn_10265599K/?,
%?"
 ?
inputs?????????
? "???????????
G__inference_flatten_5_layer_call_and_return_conditional_losses_10265656X/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? {
,__inference_flatten_5_layer_call_fn_10265661K/?,
%?"
 ?
inputs?????????
? "???????????
E__inference_model_4_layer_call_and_return_conditional_losses_10263482l????8?5
.?+
!?
input_5?????????
p

 
? "&?#
?
0??????????
? ?
E__inference_model_4_layer_call_and_return_conditional_losses_10263514l????8?5
.?+
!?
input_5?????????
p 

 
? "&?#
?
0??????????
? ?
E__inference_model_4_layer_call_and_return_conditional_losses_10265210k????7?4
-?*
 ?
inputs?????????
p

 
? "&?#
?
0??????????
? ?
E__inference_model_4_layer_call_and_return_conditional_losses_10265252k????7?4
-?*
 ?
inputs?????????
p 

 
? "&?#
?
0??????????
? ?
*__inference_model_4_layer_call_fn_10263560_????8?5
.?+
!?
input_5?????????
p

 
? "????????????
*__inference_model_4_layer_call_fn_10263605_????8?5
.?+
!?
input_5?????????
p 

 
? "????????????
*__inference_model_4_layer_call_fn_10265265^????7?4
-?*
 ?
inputs?????????
p

 
? "????????????
*__inference_model_4_layer_call_fn_10265278^????7?4
-?*
 ?
inputs?????????
p 

 
? "????????????
E__inference_model_5_layer_call_and_return_conditional_losses_10263708l????8?5
.?+
!?
input_6?????????
p

 
? "&?#
?
0??????????
? ?
E__inference_model_5_layer_call_and_return_conditional_losses_10263740l????8?5
.?+
!?
input_6?????????
p 

 
? "&?#
?
0??????????
? ?
E__inference_model_5_layer_call_and_return_conditional_losses_10265320k????7?4
-?*
 ?
inputs?????????
p

 
? "&?#
?
0??????????
? ?
E__inference_model_5_layer_call_and_return_conditional_losses_10265362k????7?4
-?*
 ?
inputs?????????
p 

 
? "&?#
?
0??????????
? ?
*__inference_model_5_layer_call_fn_10263786_????8?5
.?+
!?
input_6?????????
p

 
? "????????????
*__inference_model_5_layer_call_fn_10263831_????8?5
.?+
!?
input_6?????????
p 

 
? "????????????
*__inference_model_5_layer_call_fn_10265375^????7?4
-?*
 ?
inputs?????????
p

 
? "????????????
*__inference_model_5_layer_call_fn_10265388^????7?4
-?*
 ?
inputs?????????
p 

 
? "????????????
I__inference_normalize_2_layer_call_and_return_conditional_losses_10265560[??+?(
!?
?
x??????????
? "&?#
?
0??????????
? ?
.__inference_normalize_2_layer_call_fn_10265569N??+?(
!?
?
x??????????
? "????????????
&__inference_signature_wrapper_10264690?.???????????LMYZST_`eflmst{|???????
? 
???
&
bets?
bets?????????

*
cards0 ?
cards0?????????
*
cards1 ?
cards1?????????"3?0
.
dense_26"?
dense_26?????????